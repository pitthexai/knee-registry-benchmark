from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from orthobench.datasets import VIEW_CLASSES, ViewDataset, normalize_view_label
from orthobench.track_b_models import load_track_b_checkpoint


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(_: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_loader(dataset, batch: int, workers: int, shuffle: bool, seed: int):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator,
    )


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    labels, predictions = [], []
    for images, y, _ in loader:
        pred = model(images.to(device)).argmax(1).cpu().numpy()
        labels.extend(y.numpy().tolist())
        predictions.extend(pred.tolist())
    return np.asarray(labels), np.asarray(predictions)


def sample_fewshot(frame, shots_per_class: int, seed: int, group_col: str):
    """Sample whole patient/study groups until each class budget is reached."""
    work = frame.copy()
    work["view_label_norm"] = work["view_label"].map(normalize_view_label)
    selected = []
    for class_name in VIEW_CLASSES:
        subset = work[work["view_label_norm"] == class_name]
        if subset.empty:
            continue
        groups = (
            subset[group_col]
            .drop_duplicates()
            .sample(frac=1, random_state=int(seed))
            .tolist()
        )
        chosen = []
        for group in groups:
            chosen.append(group)
            if subset[subset[group_col].isin(chosen)].shape[0] >= shots_per_class:
                break
        selected.append(subset[subset[group_col].isin(chosen)])
    if not selected:
        return work.iloc[0:0]
    return (
        pd.concat(selected)
        .drop_duplicates()
        .sample(frac=1, random_state=int(seed))
        .reset_index(drop=True)
    )


def _unfreeze(module) -> None:
    if module is None:
        return
    if isinstance(module, nn.Parameter):
        module.requires_grad = True
        return
    for parameter in module.parameters():
        parameter.requires_grad = True


def configure_partial_finetuning(model) -> list[str]:
    """Freeze the encoder, then unfreeze its last block/stage and final layers."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    _unfreeze(model.head)
    unfrozen = ["head"]
    encoder = model.encoder

    # timm ViT and DINOv2
    if hasattr(encoder, "blocks") and len(encoder.blocks):
        _unfreeze(encoder.blocks[-1])
        unfrozen.append("encoder.blocks[-1]")
    # timm Swin
    elif hasattr(encoder, "layers") and len(encoder.layers):
        _unfreeze(encoder.layers[-1])
        unfrozen.append("encoder.layers[-1]")
    elif hasattr(encoder, "stages") and len(encoder.stages):
        _unfreeze(encoder.stages[-1])
        unfrozen.append("encoder.stages[-1]")
    # BiomedCLIP OpenCLIP visual tower wraps a timm trunk.
    elif hasattr(encoder, "trunk"):
        trunk = encoder.trunk
        if hasattr(trunk, "blocks") and len(trunk.blocks):
            _unfreeze(trunk.blocks[-1])
            unfrozen.append("encoder.trunk.blocks[-1]")
        elif hasattr(trunk, "stages") and len(trunk.stages):
            _unfreeze(trunk.stages[-1])
            unfrozen.append("encoder.trunk.stages[-1]")
    else:
        raise RuntimeError(
            f"Could not locate the final encoder block for {model.model_key}."
        )

    # Allow the final normalization/projection to adapt when present.
    for name in ("norm", "fc_norm", "ln_post", "head"):
        if hasattr(encoder, name):
            _unfreeze(getattr(encoder, name))
            unfrozen.append(f"encoder.{name}")
    for name in ("proj", "projection"):
        if hasattr(encoder, name):
            _unfreeze(getattr(encoder, name))
            unfrozen.append(f"encoder.{name}")

    if hasattr(encoder, "trunk") and hasattr(encoder.trunk, "norm"):
        _unfreeze(encoder.trunk.norm)
        unfrozen.append("encoder.trunk.norm")

    if not any(parameter.requires_grad for parameter in model.encoder.parameters()):
        raise RuntimeError(f"No encoder parameters were unfrozen for {model.model_key}.")
    return sorted(set(unfrozen))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--adapt_name", default="OAI_adapt")
    parser.add_argument("--test_name", default="OAI_test")
    parser.add_argument("--split_col", default="split_view_external_oai")
    parser.add_argument(
        "--shots_per_class", type=int, nargs="+", default=[5, 10, 25, 50, 100]
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--base_seed", type=int, default=None,
        help="Restrict adaptation to the matching base-training seed.",
    )
    parser.add_argument("--path_col", default="png_path")
    parser.add_argument("--group_col", default="group_id")
    args = parser.parse_args()

    seed_everything(args.seed)
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.manifest, low_memory=False)
    adapt = frame[
        (frame[args.split_col] == args.adapt_name) & (frame.has_view_label == 1)
    ].copy()
    test = frame[
        (frame[args.split_col] == args.test_name) & (frame.has_view_label == 1)
    ].copy()
    if adapt.empty or test.empty:
        raise SystemExit("OAI adaptation or test split is empty.")
    if args.group_col not in adapt.columns:
        raise SystemExit(f"Missing group column: {args.group_col}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"
    checkpoint_paths = sorted(Path(args.ckpt_dir).glob("*.pt"))
    if args.base_seed is not None:
        checkpoint_paths = [
            path for path in checkpoint_paths
            if path.stem.endswith(f"_seed{args.base_seed}")
        ]
    if not checkpoint_paths:
        raise SystemExit("No matching Track B checkpoints found.")

    rows = []
    for checkpoint_path in checkpoint_paths:
        base_checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if base_checkpoint.get("track") != "B":
            raise ValueError(f"Not a Track B checkpoint: {checkpoint_path}")
        base_model_id = base_checkpoint["model_id"]
        preprocessing = base_checkpoint["preprocessing"]
        test_dataset = ViewDataset(
            test, False, path_col=args.path_col, preprocessing=preprocessing
        )
        test_loader = make_loader(
            test_dataset, args.batch * 2, args.num_workers, False, args.seed + 20_000
        )

        for shots in args.shots_per_class:
            seed_everything(args.seed)
            fewshot = sample_fewshot(adapt, shots, args.seed, args.group_col)
            if fewshot.empty:
                continue
            model, loaded_checkpoint = load_track_b_checkpoint(checkpoint_path, device)
            unfrozen_modules = configure_partial_finetuning(model)

            normalized = fewshot["view_label"].map(normalize_view_label)
            counts = (
                normalized.value_counts()
                .reindex(VIEW_CLASSES)
                .fillna(0)
                .to_numpy(float)
            )
            weights = counts.sum() / (counts + 1e-6)
            weights /= weights.mean()
            criterion = nn.CrossEntropyLoss(
                weight=torch.tensor(weights, dtype=torch.float32, device=device)
            )
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
            scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
            train_dataset = ViewDataset(
                fewshot, True, path_col=args.path_col, preprocessing=preprocessing
            )
            train_loader = make_loader(
                train_dataset, args.batch, args.num_workers, True, args.seed
            )
            for _ in range(args.epochs):
                # Match Track A: train mode is used globally, while gradients are
                # restricted to the head, final encoder block, and final norm.
                model.train()
                for images, labels, _ in train_loader:
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.cuda.amp.autocast(enabled=amp_enabled):
                        loss = criterion(model(images), labels)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], args.grad_clip
                    )
                    scaler.step(optimizer)
                    scaler.update()

            y_true, y_pred = evaluate(model, test_loader, device)
            model_id = (
                f"fewshot_{base_model_id}_{shots}spc_adaptseed{args.seed}"
            )
            output_checkpoint = output / f"{model_id}.pt"
            torch.save(
                {
                    "track": "B",
                    "model": model.state_dict(),
                    "model_key": loaded_checkpoint["model_key"],
                    "model_source": loaded_checkpoint.get("model_source"),
                    "model_id": model_id,
                    "regime": "view_maintenance_partial_ft",
                    "seed": loaded_checkpoint["seed"],
                    "base_seed": loaded_checkpoint["seed"],
                    "base_model_id": base_model_id,
                    "adapt_seed": args.seed,
                    "img_size": loaded_checkpoint.get("img_size", 224),
                    "preprocessing": preprocessing,
                    "classes": VIEW_CLASSES,
                    "shots_per_class": shots,
                    "n_fewshot": len(fewshot),
                    "epochs": args.epochs,
                    "learning_rate": args.lr,
                    "unfrozen_modules": unfrozen_modules,
                },
                output_checkpoint,
            )
            row = {
                "model": model_id,
                "architecture": loaded_checkpoint["model_key"],
                "base_model_id": base_model_id,
                "base_seed": loaded_checkpoint["seed"],
                "adapt_seed": args.seed,
                "shots_per_class": shots,
                "n_fewshot": len(fewshot),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "macro_f1": float(
                    f1_score(y_true, y_pred, average="macro", zero_division=0)
                ),
                "weighted_f1": float(
                    f1_score(y_true, y_pred, average="weighted", zero_division=0)
                ),
                "ckpt": str(output_checkpoint),
            }
            rows.append(row)
            print(row, flush=True)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    pd.DataFrame(rows).to_csv(
        output / f"view_fewshot_maintenance_seed{args.seed}.csv", index=False
    )


if __name__ == "__main__":
    main()

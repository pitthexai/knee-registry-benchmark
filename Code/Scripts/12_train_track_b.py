from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader

from orthobench.datasets import VIEW_CLASSES, ViewDataset, normalize_view_label
from orthobench.preprocessing import PreprocessingConfig
from orthobench.track_b_models import (
    TRACK_B_SPECS,
    configure_regime,
    create_track_b_model,
    set_training_state,
    track_b_normalization,
)


def seed_everything(seed: int, deterministic: bool):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def seed_worker(_):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed); np.random.seed(seed)


def loader(dataset, batch, workers, shuffle, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset, batch_size=batch, shuffle=shuffle, num_workers=workers,
        pin_memory=torch.cuda.is_available(), worker_init_fn=seed_worker, generator=generator,
    )


@torch.inference_mode()
def evaluate(model, data_loader, device):
    model.eval(); ys, predictions = [], []
    for images, labels, _ in data_loader:
        pred = model(images.to(device)).argmax(1).cpu().numpy()
        ys.extend(labels.numpy().tolist()); predictions.extend(pred.tolist())
    return np.asarray(ys), np.asarray(predictions)


def schedule_factor(step, total_steps, warmup_steps):
    if step < warmup_steps:
        return max((step + 1) / max(warmup_steps, 1), 1e-3)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--models", nargs="+", default=list(TRACK_B_SPECS))
    parser.add_argument("--regimes", nargs="+", choices=["linear_probe", "full_ft"], default=["linear_probe", "full_ft"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--linear_epochs", type=int, default=15)
    parser.add_argument("--full_epochs", type=int, default=20)
    parser.add_argument("--linear_lr", type=float, default=1e-3)
    parser.add_argument("--encoder_lr", type=float, default=1e-5)
    parser.add_argument("--head_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--max_train_images", type=int, default=0, help="Stratified smoke-test limit; 0 uses all images")
    parser.add_argument("--max_eval_images", type=int, default=0, help="Stratified smoke-test limit; 0 uses all images")
    parser.add_argument("--path_col", default="png_path")
    parser.add_argument("--resize_mode", choices=["pad", "stretch"], default="pad")
    parser.add_argument("--intensity_mode", choices=["percentile", "minmax", "none"], default="percentile")
    parser.add_argument("--mask_mode", choices=["none", "border", "corners", "border_corners"], default="none")
    parser.add_argument("--pad_value", type=int, default=0)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    output = Path(args.out_dir); (output / "view_ckpts").mkdir(parents=True, exist_ok=True); (output / "results").mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.manifest, low_memory=False)
    splits = {name: frame[(frame.split_view == name) & (frame.has_view_label == 1)].copy() for name in ["train", "val", "test"]}
    def stratified_limit(data, limit, seed):
        if limit <= 0 or len(data) <= limit:
            return data
        per_class = max(1, limit // len(VIEW_CLASSES))
        pieces = []
        normalized = data.view_label.map(normalize_view_label)
        for class_name in VIEW_CLASSES:
            candidates = data[normalized == class_name]
            pieces.append(candidates.sample(n=min(per_class, len(candidates)), random_state=seed))
        return pd.concat(pieces).sample(frac=1, random_state=seed).reset_index(drop=True)
    splits["train"] = stratified_limit(splits["train"], args.max_train_images, args.seed)
    splits["val"] = stratified_limit(splits["val"], args.max_eval_images, args.seed + 1)
    splits["test"] = stratified_limit(splits["test"], args.max_eval_images, args.seed + 2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"
    rows = []
    for model_key in args.models:
        for regime in args.regimes:
            seed_everything(args.seed, args.deterministic)
            print(f"\n===== Track B {model_key} {regime} seed={args.seed} =====", flush=True)
            model = create_track_b_model(
                model_key, len(VIEW_CLASSES), pretrained=True, img_size=args.img_size
            ).to(device)
            configure_regime(model, regime)
            mean, std = track_b_normalization(model_key, model)
            preprocessing = PreprocessingConfig(
                img_size=args.img_size, resize_mode=args.resize_mode,
                intensity_mode=args.intensity_mode, mask_mode=args.mask_mode,
                pad_value=args.pad_value, mean=mean, std=std,
            ).to_dict()
            datasets = {name: ViewDataset(data, name == "train", path_col=args.path_col, preprocessing=preprocessing) for name, data in splits.items()}
            loaders = {
                "train": loader(datasets["train"], args.batch, args.num_workers, True, args.seed),
                "val": loader(datasets["val"], args.batch * 2, args.num_workers, False, args.seed + 10_000),
                "test": loader(datasets["test"], args.batch * 2, args.num_workers, False, args.seed + 20_000),
            }
            counts = datasets["train"].df.view_label_norm.value_counts().reindex(VIEW_CLASSES).fillna(0).to_numpy(float)
            weights = counts.sum() / (counts + 1e-6); weights /= weights.mean()
            criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
            epochs = args.linear_epochs if regime == "linear_probe" else args.full_epochs
            if regime == "linear_probe":
                groups = [{"params": model.head.parameters(), "lr": args.linear_lr}]
            else:
                groups = [
                    {"params": model.encoder.parameters(), "lr": args.encoder_lr},
                    {"params": model.head.parameters(), "lr": args.head_lr},
                ]
            optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
            total_steps = epochs * len(loaders["train"]); warmup_steps = args.warmup_epochs * len(loaders["train"])
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: schedule_factor(step, total_steps, warmup_steps))
            scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
            model_id = f"{model_key}_{regime}_seed{args.seed}"
            checkpoint_path = output / "view_ckpts" / f"{model_id}.pt"
            best_macro = -1.0; best_epoch = 0; history = []
            for epoch in range(1, epochs + 1):
                set_training_state(model, regime); total_loss = 0.0
                for images, labels, _ in loaders["train"]:
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.cuda.amp.autocast(enabled=amp_enabled):
                        loss = criterion(model(images), labels)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip)
                    scaler.step(optimizer); scaler.update(); scheduler.step()
                    total_loss += float(loss.item()) * images.size(0)
                y_val, p_val = evaluate(model, loaders["val"], device)
                macro = float(f1_score(y_val, p_val, average="macro", zero_division=0))
                weighted = float(f1_score(y_val, p_val, average="weighted", zero_division=0))
                record = {"epoch": epoch, "loss": total_loss / len(datasets["train"]), "val_macro_f1": macro, "val_weighted_f1": weighted, "lr": [g["lr"] for g in optimizer.param_groups]}
                history.append(record); print(record, flush=True)
                if macro > best_macro:
                    best_macro, best_epoch = macro, epoch
                    torch.save({
                        "track": "B", "model": model.state_dict(), "model_key": model_key,
                        "model_source": TRACK_B_SPECS[model_key].source, "model_id": model_id,
                        "regime": regime, "seed": args.seed, "img_size": args.img_size,
                        "preprocessing": preprocessing, "classes": VIEW_CLASSES,
                        "best_epoch": best_epoch, "best_val_macro_f1": best_macro,
                        "deterministic": args.deterministic,
                    }, checkpoint_path)
            checkpoint = torch.load(checkpoint_path, map_location=device); model.load_state_dict(checkpoint["model"])
            y_test, p_test = evaluate(model, loaders["test"], device)
            row = {
                "model": model_id, "architecture": model_key, "regime": regime, "seed": args.seed,
                "split": "UPMC_internal_test", "accuracy": accuracy_score(y_test, p_test),
                "macro_f1": f1_score(y_test, p_test, average="macro", zero_division=0),
                "weighted_f1": f1_score(y_test, p_test, average="weighted", zero_division=0),
                "best_epoch": best_epoch, "best_val_macro_f1": best_macro, "ckpt": str(checkpoint_path),
            }
            rows.append(row); print("TEST", row); print(classification_report(y_test, p_test, target_names=VIEW_CLASSES, digits=4, zero_division=0))
            with open(output / "results" / f"history_{model_id}.json", "w") as handle: json.dump(history, handle, indent=2)
            del model
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(output / "results" / f"track_b_internal_seed{args.seed}.csv", index=False)


if __name__ == "__main__":
    main()

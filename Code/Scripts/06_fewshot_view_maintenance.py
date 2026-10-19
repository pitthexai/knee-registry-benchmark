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
from orthobench.models import create_timm_model, freeze_all, unfreeze_classifier_and_last_block


def seed_everything(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def seeded_generator(seed: int):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def eval_model(model, loader, device):
    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for x, y, _ in loader:
            pred = model(x.to(device)).argmax(1).cpu().numpy()
            ys.extend(y.numpy().tolist())
            preds.extend(pred.tolist())
    return np.asarray(ys), np.asarray(preds)


def metrics(y, pred, prefix):
    return {
        f"{prefix}_accuracy": float(accuracy_score(y, pred)),
        f"{prefix}_macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        f"{prefix}_weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
    }


def sample_fewshot(df, shots_per_class, seed, group_col):
    """Select nested whole-group samples; the same seed gives the same prefix."""
    out = []
    work = df.copy()
    work["view_label_norm"] = work["view_label"].map(normalize_view_label)
    for class_name in VIEW_CLASSES:
        sub = work[work["view_label_norm"] == class_name]
        if len(sub) == 0:
            continue
        groups = sub[group_col].drop_duplicates().sample(frac=1, random_state=int(seed)).tolist()
        chosen = []
        for group in groups:
            chosen.append(group)
            if sub[sub[group_col].isin(chosen)].shape[0] >= shots_per_class:
                break
        out.append(sub[sub[group_col].isin(chosen)])
    if not out:
        return work.iloc[0:0]
    return pd.concat(out).drop_duplicates().sample(frac=1, random_state=int(seed)).reset_index(drop=True)


def class_weights(frame, device):
    normalized = frame["view_label"].map(normalize_view_label)
    counts = normalized.value_counts().reindex(VIEW_CLASSES).fillna(0).to_numpy(dtype=float)
    weights = np.zeros_like(counts)
    present = counts > 0
    weights[present] = counts[present].sum() / counts[present]
    weights[present] /= weights[present].mean()
    return torch.tensor(weights, dtype=torch.float32, device=device), counts.astype(int)


def make_loader(frame, train, img_size, path_col, preprocessing, batch, workers, seed):
    return DataLoader(
        ViewDataset(frame, train, img_size, path_col, preprocessing),
        batch_size=batch, shuffle=train, num_workers=workers, pin_memory=True,
        worker_init_fn=seed_worker, generator=seeded_generator(seed),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--adapt_name", default="OAI_adapt")
    ap.add_argument("--test_name", default="OAI_test")
    ap.add_argument("--split_col", default="split_view_external_oai")
    ap.add_argument("--retention_split_col", default="split_view")
    ap.add_argument("--retention_name", default="test")
    ap.add_argument("--shots_per_class", type=int, nargs="+", default=[5, 10, 25, 50, 100])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base_seed", type=int, default=None)
    ap.add_argument("--path_col", default=None)
    ap.add_argument("--group_col", default="group_id")
    ap.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    seed_everything(args.seed, args.deterministic)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.manifest, low_memory=False)
    adapt = df[(df[args.split_col] == args.adapt_name) & (df["has_view_label"] == 1)].copy()
    external_test = df[(df[args.split_col] == args.test_name) & (df["has_view_label"] == 1)].copy()
    retention_test = df[(df[args.retention_split_col] == args.retention_name) & (df["has_view_label"] == 1)].copy()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    ckpt_paths = sorted(Path(args.ckpt_dir).glob("*.pt"))
    if args.base_seed is not None:
        ckpt_paths = [p for p in ckpt_paths if p.stem.endswith(f"_seed{args.base_seed}")]
    if not ckpt_paths:
        raise SystemExit("No matching base checkpoints found")

    for ckpt_path in ckpt_paths:
        seed_everything(args.seed, args.deterministic)
        base_ckpt = torch.load(ckpt_path, map_location=device)
        model_name = base_ckpt.get("model_name", ckpt_path.stem)
        base_model_id = base_ckpt.get("model_id", ckpt_path.stem)
        img_size = base_ckpt.get("img_size", 224)
        preprocessing = base_ckpt.get("preprocessing")
        external_loader = make_loader(external_test, False, img_size, args.path_col, preprocessing, args.batch, args.num_workers, args.seed + 20_000)
        retention_loader = make_loader(retention_test, False, img_size, args.path_col, preprocessing, args.batch, args.num_workers, args.seed + 30_000)

        baseline_model = create_timm_model(model_name, num_classes=len(VIEW_CLASSES), pretrained=False).to(device)
        baseline_model.load_state_dict(base_ckpt["model"])
        y_ext_base, p_ext_base = eval_model(baseline_model, external_loader, device)
        y_ret_base, p_ret_base = eval_model(baseline_model, retention_loader, device)
        baseline_ext = metrics(y_ext_base, p_ext_base, "baseline_oai")
        baseline_ret = metrics(y_ret_base, p_ret_base, "baseline_retention")
        del baseline_model

        for requested_k in args.shots_per_class:
            seed_everything(args.seed, args.deterministic)
            fewshot = sample_fewshot(adapt, requested_k, args.seed, args.group_col)
            if len(fewshot) == 0:
                continue
            weights, counts = class_weights(fewshot, device)
            model = create_timm_model(model_name, num_classes=len(VIEW_CLASSES), pretrained=False).to(device)
            model.load_state_dict(base_ckpt["model"])
            freeze_all(model)
            unfreeze_classifier_and_last_block(model)
            criterion = nn.CrossEntropyLoss(weight=weights)
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
            train_loader = make_loader(fewshot, True, img_size, args.path_col, preprocessing, args.batch, args.num_workers, args.seed + requested_k)
            for _ in range(args.epochs):
                model.train()
                for x, y, _ in train_loader:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(x), y)
                    loss.backward()
                    optimizer.step()

            y_ext, p_ext = eval_model(model, external_loader, device)
            y_ret, p_ret = eval_model(model, retention_loader, device)
            ext = metrics(y_ext, p_ext, "oai")
            ret = metrics(y_ret, p_ret, "retention")
            model_id = f"fewshot_{base_model_id}_{requested_k}spc_adaptseed{args.seed}"
            out_ckpt = out_dir / f"{model_id}.pt"
            checkpoint = {
                "model": model.state_dict(), "model_name": model_name,
                "model_id": model_id, "base_model_id": base_model_id,
                "base_seed": base_ckpt.get("seed", args.base_seed), "adapt_seed": args.seed,
                "img_size": img_size, "preprocessing": preprocessing,
                "classes": VIEW_CLASSES, "requested_shots_per_class": requested_k,
                "actual_class_counts": dict(zip(VIEW_CLASSES, counts.tolist())),
                "n_fewshot_images": len(fewshot),
                "n_fewshot_groups": int(fewshot[args.group_col].nunique()),
                "deterministic": args.deterministic,
            }
            torch.save(checkpoint, out_ckpt)
            row = {
                "model": model_id, "architecture": model_name,
                "base_model_id": base_model_id, "base_seed": base_ckpt.get("seed", args.base_seed),
                "adapt_seed": args.seed, "requested_shots_per_class": requested_k,
                "n_fewshot_images": len(fewshot),
                "n_fewshot_groups": int(fewshot[args.group_col].nunique()),
                **{f"actual_{name}": int(counts[i]) for i, name in enumerate(VIEW_CLASSES)},
                **baseline_ext, **baseline_ret, **ext, **ret,
                "delta_oai_accuracy": ext["oai_accuracy"] - baseline_ext["baseline_oai_accuracy"],
                "delta_oai_macro_f1": ext["oai_macro_f1"] - baseline_ext["baseline_oai_macro_f1"],
                "delta_oai_weighted_f1": ext["oai_weighted_f1"] - baseline_ext["baseline_oai_weighted_f1"],
                "delta_retention_accuracy": ret["retention_accuracy"] - baseline_ret["baseline_retention_accuracy"],
                "delta_retention_macro_f1": ret["retention_macro_f1"] - baseline_ret["baseline_retention_macro_f1"],
                "delta_retention_weighted_f1": ret["retention_weighted_f1"] - baseline_ret["baseline_retention_weighted_f1"],
                "ckpt": str(out_ckpt),
            }
            rows.append(row)
            print(row)

    pd.DataFrame(rows).to_csv(out_dir / f"view_fewshot_maintenance_seed{args.seed}.csv", index=False)


if __name__ == "__main__":
    main()

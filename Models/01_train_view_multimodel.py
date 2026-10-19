from __future__ import annotations

import argparse, json, random, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from orthobench.datasets import ViewDataset, VIEW_CLASSES
from orthobench.models import create_timm_model, model_normalization
from orthobench.preprocessing import PreprocessingConfig


def seed_everything(seed: int, deterministic: bool = True):
    """Reset all RNGs so a model's result does not depend on model-list order."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def seed_worker(worker_id: int):
    """Seed Python and NumPy from the worker seed assigned by DataLoader."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def seeded_generator(seed: int):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def run_eval(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            pred = model(x).argmax(1).cpu().numpy()
            ys.extend(y.numpy().tolist())
            ps.extend(pred.tolist())
    return np.asarray(ys), np.asarray(ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--models", nargs="+", default=["tf_efficientnet_b0", "resnet50", "densenet121", "convnext_tiny", "vit_base_patch16_224", "swin_tiny_patch4_window7_224"])
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--path_col", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resize_mode", choices=["pad", "stretch"], default="pad")
    ap.add_argument("--intensity_mode", choices=["percentile", "minmax", "none"], default="percentile")
    ap.add_argument("--mask_mode", choices=["none", "border", "corners", "border_corners"], default="none")
    ap.add_argument("--percentile_low", type=float, default=0.5)
    ap.add_argument("--percentile_high", type=float, default=99.5)
    ap.add_argument("--border_fraction", type=float, default=0.04)
    ap.add_argument("--corner_fraction", type=float, default=0.18)
    ap.add_argument("--pad_value", type=int, default=0)
    ap.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()
    seed_everything(args.seed, args.deterministic)

    out_dir = Path(args.out_dir)
    (out_dir / "view_ckpts").mkdir(parents=True, exist_ok=True)
    (out_dir / "results").mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.manifest, low_memory=False)

    train_df = df[(df["split_view"] == "train") & (df["has_view_label"] == 1)].copy()
    val_df = df[(df["split_view"] == "val") & (df["has_view_label"] == 1)].copy()
    test_df = df[(df["split_view"] == "test") & (df["has_view_label"] == 1)].copy()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = []
    for model_name in args.models:
        # This reset is essential: otherwise later architectures inherit RNG
        # state consumed by models trained earlier in the same invocation.
        seed_everything(args.seed, args.deterministic)
        train_generator = seeded_generator(args.seed)
        val_generator = seeded_generator(args.seed + 10_000)
        test_generator = seeded_generator(args.seed + 20_000)
        print(f"\n===== Training {model_name} =====")
        model = create_timm_model(model_name, num_classes=len(VIEW_CLASSES), pretrained=True).to(device)
        mean, std = model_normalization(model)
        preprocessing = PreprocessingConfig(
            img_size=args.img_size, resize_mode=args.resize_mode,
            intensity_mode=args.intensity_mode, percentile_low=args.percentile_low,
            percentile_high=args.percentile_high, mask_mode=args.mask_mode,
            border_fraction=args.border_fraction, corner_fraction=args.corner_fraction,
            pad_value=args.pad_value, mean=mean, std=std,
        ).to_dict()
        train_ds = ViewDataset(train_df, train=True, path_col=args.path_col, preprocessing=preprocessing)
        val_ds = ViewDataset(val_df, train=False, path_col=args.path_col, preprocessing=preprocessing)
        test_ds = ViewDataset(test_df, train=False, path_col=args.path_col, preprocessing=preprocessing)
        train_loader = DataLoader(
            train_ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers,
            pin_memory=True, worker_init_fn=seed_worker, generator=train_generator,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers,
            pin_memory=True, worker_init_fn=seed_worker, generator=val_generator,
        )
        test_loader = DataLoader(
            test_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers,
            pin_memory=True, worker_init_fn=seed_worker, generator=test_generator,
        )
        labels = train_ds.df["view_label_norm"].value_counts().reindex(VIEW_CLASSES).fillna(0).values.astype(float)
        weights = labels.sum() / (labels + 1e-6)
        weights = torch.tensor(weights / weights.mean(), dtype=torch.float32, device=device)
        crit = nn.CrossEntropyLoss(weight=weights)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        best_macro = -1
        best_epoch = None
        model_id = f"{model_name}_seed{args.seed}"
        ckpt_path = out_dir / "view_ckpts" / f"{model_id}.pt"
        history = []
        for ep in range(1, args.epochs + 1):
            model.train()
            total = 0.0
            for x, y, _ in train_loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                loss = crit(model(x), y)
                loss.backward()
                opt.step()
                total += float(loss.item()) * x.size(0)
            yv, pv = run_eval(model, val_loader, device)
            macro = f1_score(yv, pv, average="macro", zero_division=0)
            w = f1_score(yv, pv, average="weighted", zero_division=0)
            print(f"{model_name} ep={ep:03d} loss={total/len(train_ds):.4f} val_macroF1={macro:.4f} val_wF1={w:.4f}")
            history.append({"epoch": ep, "loss": total/len(train_ds), "val_macro_f1": macro, "val_weighted_f1": w})
            if macro > best_macro:
                best_macro = macro
                best_epoch = ep
                torch.save({
                    "model": model.state_dict(), "model_name": model_name,
                    "model_id": model_id, "seed": args.seed, "img_size": args.img_size,
                    "preprocessing": preprocessing, "classes": VIEW_CLASSES,
                    "best_epoch": best_epoch, "best_val_macro_f1": float(best_macro),
                    "deterministic": args.deterministic,
                    "seed_scope": "reset_per_model_with_seeded_dataloaders",
                }, ckpt_path)
        # internal test
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        yt, pt = run_eval(model, test_loader, device)
        row = {
            "model": model_id,
            "architecture": model_name,
            "seed": args.seed,
            "split": "UPMC_internal_test",
            "accuracy": float(accuracy_score(yt, pt)),
            "macro_f1": float(f1_score(yt, pt, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(yt, pt, average="weighted", zero_division=0)),
            "ckpt": str(ckpt_path),
            "resize_mode": preprocessing["resize_mode"],
            "intensity_mode": preprocessing["intensity_mode"],
            "mask_mode": preprocessing["mask_mode"],
            "best_epoch": best_epoch,
            "best_val_macro_f1": float(best_macro),
            "deterministic": args.deterministic,
        }
        print("TEST", row)
        print(classification_report(yt, pt, target_names=VIEW_CLASSES, digits=4, zero_division=0))
        rows.append(row)
        with open(out_dir / "results" / f"view_train_history_{model_id}.json", "w") as f:
            json.dump(history, f, indent=2)
    pd.DataFrame(rows).to_csv(out_dir / "results" / f"view_internal_leaderboard_seed{args.seed}.csv", index=False)


if __name__ == "__main__":
    main()

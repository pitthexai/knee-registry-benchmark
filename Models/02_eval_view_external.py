from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from orthobench.datasets import ViewDataset, VIEW_CLASSES
from orthobench.models import create_timm_model


def predict(model, loader, device):
    model.eval()
    ys, ps, probs, uids = [], [], [], []
    with torch.no_grad():
        for x, y, uid in loader:
            x = x.to(device)
            logits = model(x)
            p = torch.softmax(logits, 1).cpu().numpy()
            probs.append(p)
            ps.extend(p.argmax(1).tolist())
            ys.extend(y.numpy().tolist())
            uids.extend(uid)
    return np.asarray(ys), np.asarray(ps), np.vstack(probs), uids


def eval_one(ckpt_path, df, split_name, batch, workers, path_col):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device)
    model_name = ckpt.get("model_name", Path(ckpt_path).stem)
    model_id = ckpt.get("model_id", Path(ckpt_path).stem)
    img_size = ckpt.get("img_size", 224)
    preprocessing = ckpt.get("preprocessing")
    model = create_timm_model(model_name, num_classes=len(VIEW_CLASSES), pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    ds = ViewDataset(df, train=False, img_size=img_size, path_col=path_col, preprocessing=preprocessing)
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True)
    y, pred, prob, uids = predict(model, loader, device)
    return {
        "model": model_id,
        "architecture": model_name,
        "split": split_name,
        "n": len(y),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "preprocessing_version": (preprocessing or {}).get("version", "2-legacy"),
    }, pd.DataFrame({"uid": uids, "y": y, "pred": pred, **{f"prob_{c}": prob[:, i] for i, c in enumerate(VIEW_CLASSES)}})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--split_col", default="split_view_external_oai")
    ap.add_argument("--split_name", default="OAI_test")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--path_col", default=None)
    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.manifest, low_memory=False)
    sub = df[(df[args.split_col] == args.split_name) & (df["has_view_label"] == 1)].copy()
    rows = []
    for ckpt_path in sorted(Path(args.ckpt_dir).glob("*.pt")):
        row, pred_df = eval_one(str(ckpt_path), sub, args.split_name, args.batch, args.num_workers, args.path_col)
        metadata_cols = [c for c in ["uid", "group_id", "source", "view_label", "side_label"] if c in sub.columns]
        pred_df = pred_df.merge(sub[metadata_cols].drop_duplicates("uid"), on="uid", how="left", validate="one_to_one")
        print(row)
        pred_df.to_csv(out_dir / f"pred_view_{row['model']}_{args.split_name}.csv", index=False)
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / f"view_external_{args.split_name}_leaderboard.csv", index=False)


if __name__ == "__main__":
    main()

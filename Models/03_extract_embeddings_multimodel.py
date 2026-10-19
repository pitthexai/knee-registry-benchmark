from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from orthobench.datasets import DomainDataset, collate_drop_none
from orthobench.models import create_timm_model, extract_features


def rows_for_split(df, which):
    if which in {"train", "val", "calib", "test"}:
        return df[df["split_domain"] == which].copy()
    if which == "fracatlas":
        return df[df["split_domain_external"] == "FracAtlas_test"].copy()
    raise ValueError(which)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--which", nargs="+", default=["train", "val", "calib", "test", "fracatlas"])
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--path_col", default=None)
    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.manifest, low_memory=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for ckpt_path in sorted(Path(args.ckpt_dir).glob("*.pt")):
        ckpt = torch.load(ckpt_path, map_location=device)
        model_name = ckpt.get("model_name", Path(ckpt_path).stem)
        model_id = ckpt.get("model_id", Path(ckpt_path).stem)
        img_size = ckpt.get("img_size", 224)
        preprocessing = ckpt.get("preprocessing")
        model = create_timm_model(model_name, num_classes=len(ckpt.get("classes", [0,1,2,3])), pretrained=False).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        for which in args.which:
            sub = rows_for_split(df, which)
            sub = sub[sub["domain_label"].isin([0, 1])].copy()
            ds = DomainDataset(sub, train=False, img_size=img_size, path_col=args.path_col, preprocessing=preprocessing)
            loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_drop_none)
            Xs, ys, uids = [], [], []
            with torch.no_grad():
                for batch in loader:
                    if batch is None:
                        continue
                    x, y, uid = batch
                    x = x.to(device)
                    emb = extract_features(model, x).cpu().numpy()
                    Xs.append(emb); ys.append(y.numpy()); uids.extend(uid)
            if not Xs:
                print("No embeddings", model_name, which)
                continue
            X = np.vstack(Xs); y = np.concatenate(ys)
            out = out_dir / f"emb_{model_id}_{which}.npz"
            np.savez_compressed(out, X=X, y=y, uid=np.array(uids, dtype=object))
            print("Saved", out, X.shape)


if __name__ == "__main__":
    main()

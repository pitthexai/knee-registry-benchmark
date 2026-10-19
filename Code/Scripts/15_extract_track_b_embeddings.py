from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from orthobench.datasets import DomainDataset, collate_drop_none
from orthobench.track_b_models import load_track_b_checkpoint


def rows_for_split(frame, split):
    if split in {"train", "val", "calib", "test"}: return frame[frame.split_domain == split].copy()
    if split == "fracatlas": return frame[frame.split_domain_external == "FracAtlas_test"].copy()
    raise ValueError(split)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True); parser.add_argument("--ckpt_dir", required=True); parser.add_argument("--out_dir", required=True)
    parser.add_argument("--which", nargs="+", default=["train", "val", "calib", "test", "fracatlas"])
    parser.add_argument("--path_col", default="png_path"); parser.add_argument("--batch", type=int, default=64); parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args(); output = Path(args.out_dir); output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.manifest, low_memory=False); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for checkpoint_path in sorted(Path(args.ckpt_dir).glob("*.pt")):
        model, checkpoint = load_track_b_checkpoint(checkpoint_path, device); model.eval()
        for split in args.which:
            subset = rows_for_split(frame, split); subset = subset[subset.domain_label.isin([0, 1])].copy()
            dataset = DomainDataset(subset, False, path_col=args.path_col, preprocessing=checkpoint["preprocessing"])
            loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), collate_fn=collate_drop_none)
            features, labels, uids = [], [], []
            with torch.inference_mode():
                for batch in loader:
                    if batch is None: continue
                    images, y, uid = batch
                    features.append(model.encode(images.to(device)).cpu().numpy()); labels.append(y.numpy()); uids.extend(uid)
            feature_array = np.vstack(features); label_array = np.concatenate(labels)
            path = output / f"emb_{checkpoint['model_id']}_{split}.npz"
            np.savez_compressed(path, X=feature_array, y=label_array, uid=np.asarray(uids, dtype=object))
            print("Saved", path, feature_array.shape, flush=True)
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == "__main__": main()

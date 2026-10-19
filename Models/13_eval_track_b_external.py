from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from orthobench.datasets import VIEW_CLASSES, ViewDataset
from orthobench.track_b_models import load_track_b_checkpoint


@torch.inference_mode()
def predict(model, loader, device):
    model.eval(); ys, preds, probabilities, uids = [], [], [], []
    for images, labels, batch_uids in loader:
        probability = torch.softmax(model(images.to(device)), dim=1).cpu().numpy()
        ys.extend(labels.numpy().tolist()); preds.extend(probability.argmax(1).tolist())
        probabilities.append(probability); uids.extend(batch_uids)
    return np.asarray(ys), np.asarray(preds), np.vstack(probabilities), uids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True); parser.add_argument("--ckpt_dir", required=True); parser.add_argument("--out_dir", required=True)
    parser.add_argument("--split_col", default="split_view_external_oai"); parser.add_argument("--split_name", default="OAI_test")
    parser.add_argument("--path_col", default="png_path"); parser.add_argument("--batch", type=int, default=64); parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args(); output = Path(args.out_dir); output.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest, low_memory=False)
    subset = manifest[(manifest[args.split_col] == args.split_name) & (manifest.has_view_label == 1)].copy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); rows = []
    for checkpoint_path in sorted(Path(args.ckpt_dir).glob("*.pt")):
        model, checkpoint = load_track_b_checkpoint(checkpoint_path, device)
        dataset = ViewDataset(subset, False, path_col=args.path_col, preprocessing=checkpoint["preprocessing"])
        loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
        y, pred, probability, uids = predict(model, loader, device)
        pa, lateral = VIEW_CLASSES.index("PA"), VIEW_CLASSES.index("LATERAL")
        invalid = [VIEW_CLASSES.index("AP"), VIEW_CLASSES.index("SUNRISE")]
        row = {
            "model": checkpoint["model_id"], "architecture": checkpoint["model_key"], "regime": checkpoint["regime"], "seed": checkpoint["seed"],
            "split": args.split_name, "n": len(y), "accuracy": accuracy_score(y, pred),
            "macro_f1_fixed": f1_score(y, pred, labels=[pa, lateral], average="macro", zero_division=0),
            "weighted_f1_fixed": f1_score(y, pred, labels=[pa, lateral], average="weighted", zero_division=0),
            "pa_f1": f1_score(y, pred, labels=[pa], average="macro", zero_division=0),
            "lateral_f1": f1_score(y, pred, labels=[lateral], average="macro", zero_division=0),
            "invalid_view_prediction_rate": np.isin(pred, invalid).mean(),
        }
        predictions = pd.DataFrame({"uid": uids, "y": y, "pred": pred, **{f"prob_{name}": probability[:, i] for i, name in enumerate(VIEW_CLASSES)}})
        metadata = [column for column in ["uid", "group_id", "source", "view_label", "side_label"] if column in subset]
        predictions = predictions.merge(subset[metadata].drop_duplicates("uid"), on="uid", how="left", validate="one_to_one")
        predictions.to_csv(output / f"pred_{checkpoint['model_id']}_{args.split_name}.csv", index=False)
        rows.append(row); print(row, flush=True)
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(output / f"track_b_external_{args.split_name}.csv", index=False)


if __name__ == "__main__": main()

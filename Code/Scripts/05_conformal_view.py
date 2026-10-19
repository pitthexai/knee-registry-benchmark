from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from orthobench.datasets import ViewDataset, VIEW_CLASSES
from orthobench.models import create_timm_model
from orthobench.conformal import softmax_np, aps_scores, qhat, evaluate_sets, mondrian_qhats, evaluate_class_conditional
from orthobench.metrics import expected_calibration_error


class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_t = nn.Parameter(torch.zeros(()))
    def forward(self, logits):
        return logits / torch.exp(self.log_t).clamp(1e-3, 100.0)


def get_logits(model, loader, device):
    model.eval()
    logits, ys = [], []
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            logits.append(model(x).cpu())
            ys.append(y)
    return torch.cat(logits, 0), torch.cat(ys, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--calib_split_col", default="split_view_external_oai")
    ap.add_argument("--calib_name", default="OAI_calib")
    ap.add_argument("--test_split_col", default="split_view_external_oai")
    ap.add_argument("--test_name", default="OAI_test")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--path_col", default=None)
    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.manifest, low_memory=False)
    calib_df = df[(df[args.calib_split_col] == args.calib_name) & (df["has_view_label"] == 1)].copy()
    test_df = df[(df[args.test_split_col] == args.test_name) & (df["has_view_label"] == 1)].copy()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for ckpt_path in sorted(Path(args.ckpt_dir).glob("*.pt")):
        ckpt = torch.load(ckpt_path, map_location=device)
        model_name = ckpt.get("model_name", Path(ckpt_path).stem)
        model_id = Path(ckpt_path).stem
        img_size = ckpt.get("img_size", 224)
        preprocessing = ckpt.get("preprocessing")
        model = create_timm_model(model_name, num_classes=len(VIEW_CLASSES), pretrained=False).to(device)
        model.load_state_dict(ckpt["model"])
        calib_loader = DataLoader(ViewDataset(calib_df, False, img_size, args.path_col, preprocessing), batch_size=args.batch, shuffle=False, num_workers=args.num_workers)
        test_loader = DataLoader(ViewDataset(test_df, False, img_size, args.path_col, preprocessing), batch_size=args.batch, shuffle=False, num_workers=args.num_workers)
        logits_cal, y_cal_t = get_logits(model, calib_loader, device)
        logits_te, y_te_t = get_logits(model, test_loader, device)
        y_cal = y_cal_t.numpy(); y_te = y_te_t.numpy()

        # Uncalibrated
        probs_te_uncal = softmax_np(logits_te.numpy())
        top_uncal = probs_te_uncal.argmax(1)
        row_uncal = {"model": model_id, "architecture": model_name, "setting": "uncalibrated", "split": args.test_name,
                     "accuracy": float(accuracy_score(y_te, top_uncal)),
                     "macro_f1": float(f1_score(y_te, top_uncal, average="macro", zero_division=0)),
                     "weighted_f1": float(f1_score(y_te, top_uncal, average="weighted", zero_division=0)),
                     "ece": expected_calibration_error(probs_te_uncal, y_te)}
        rows.append(row_uncal)

        # Temperature scaling on calibration split
        temp = TemperatureScaler().to(device)
        opt = torch.optim.LBFGS(temp.parameters(), lr=0.1, max_iter=200)
        nll = nn.CrossEntropyLoss()
        logits_dev = logits_cal.to(device); y_dev = y_cal_t.to(device)
        def closure():
            opt.zero_grad(); loss = nll(temp(logits_dev), y_dev); loss.backward(); return loss
        opt.step(closure)
        T = float(torch.exp(temp.log_t).detach().cpu())
        probs_cal = softmax_np(logits_cal.numpy() / T)
        probs_te = softmax_np(logits_te.numpy() / T)
        top = probs_te.argmax(1)
        scores = aps_scores(probs_cal, y_cal)
        q = qhat(scores, args.alpha)
        aps = evaluate_sets(probs_te, y_te, q)
        mq = mondrian_qhats(probs_cal, y_cal, len(VIEW_CLASSES), args.alpha)
        maps = evaluate_class_conditional(probs_te, y_te, mq)
        base = {"model": model_id, "architecture": model_name, "split": args.test_name, "T": T, "preprocessing_version": (preprocessing or {}).get("version", "2-legacy"),
                "accuracy": float(accuracy_score(y_te, top)),
                "macro_f1": float(f1_score(y_te, top, average="macro", zero_division=0)),
                "weighted_f1": float(f1_score(y_te, top, average="weighted", zero_division=0)),
                "ece": expected_calibration_error(probs_te, y_te)}
        rows.append({**base, "setting": "temp_scaled_APS", **aps})
        rows.append({**base, "setting": "temp_scaled_class_conditional_APS", **maps})
        with open(out_dir / f"conformal_{model_id}.json", "w") as f:
            json.dump({"model": model_id, "architecture": model_name, "T": T, "alpha": args.alpha, "qhat": q, "class_conditional_qhat": mq}, f, indent=2)
        print(model_id, base, aps, maps)
    pd.DataFrame(rows).to_csv(out_dir / f"view_conformal_{args.test_name}.csv", index=False)


if __name__ == "__main__":
    main()

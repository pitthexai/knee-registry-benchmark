from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from orthobench.conformal import aps_scores, evaluate_sets, qhat, softmax_np
from orthobench.datasets import VIEW_CLASSES, ViewDataset
from orthobench.metrics import expected_calibration_error
from orthobench.track_b_models import load_track_b_checkpoint


class TemperatureScaler(nn.Module):
    def __init__(self): super().__init__(); self.log_t = nn.Parameter(torch.zeros(()))
    def forward(self, logits): return logits / torch.exp(self.log_t).clamp(1e-3, 100.0)


@torch.inference_mode()
def logits_for(model, loader, device):
    # inference_mode disables gradients, but it does not disable dropout or
    # stochastic depth. Explicit evaluation mode is required for deterministic
    # logits, particularly for Swin Transformer checkpoints.
    model.eval()
    logits, labels = [], []
    for images, y, _ in loader:
        logits.append(model(images.to(device)).cpu()); labels.append(y)
    return torch.cat(logits), torch.cat(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True); parser.add_argument("--ckpt_dir", required=True); parser.add_argument("--out_dir", required=True)
    parser.add_argument("--split_col", default="split_view_external_oai"); parser.add_argument("--calib_name", default="OAI_calib"); parser.add_argument("--test_name", default="OAI_test")
    parser.add_argument("--path_col", default="png_path"); parser.add_argument("--alpha", type=float, default=0.1); parser.add_argument("--batch", type=int, default=64); parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args(); output = Path(args.out_dir); output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.manifest, low_memory=False)
    calibration = frame[(frame[args.split_col] == args.calib_name) & (frame.has_view_label == 1)].copy()
    test = frame[(frame[args.split_col] == args.test_name) & (frame.has_view_label == 1)].copy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); rows = []
    for checkpoint_path in sorted(Path(args.ckpt_dir).glob("*.pt")):
        model, checkpoint = load_track_b_checkpoint(checkpoint_path, device)
        def make_loader(data):
            dataset = ViewDataset(data, False, path_col=args.path_col, preprocessing=checkpoint["preprocessing"])
            return DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
        cal_logits, cal_y_t = logits_for(model, make_loader(calibration), device); test_logits, test_y_t = logits_for(model, make_loader(test), device)
        scaler = TemperatureScaler().to(device); optimizer = torch.optim.LBFGS(scaler.parameters(), lr=0.1, max_iter=200); loss_fn = nn.CrossEntropyLoss()
        cal_device, y_device = cal_logits.to(device), cal_y_t.to(device)
        def closure():
            optimizer.zero_grad(); loss = loss_fn(scaler(cal_device), y_device); loss.backward(); return loss
        optimizer.step(closure); temperature = float(torch.exp(scaler.log_t).detach().cpu())
        cal_probability = softmax_np(cal_logits.numpy() / temperature); test_probability = softmax_np(test_logits.numpy() / temperature)
        y_cal, y_test = cal_y_t.numpy(), test_y_t.numpy()
        raw_pred = test_logits.numpy().argmax(1)
        pred = test_probability.argmax(1)
        if not np.array_equal(raw_pred, pred):
            raise RuntimeError(
                f"Temperature scaling changed argmax predictions for {checkpoint['model_id']}; "
                "this should be mathematically impossible for a positive scalar temperature."
            )
        threshold = qhat(aps_scores(cal_probability, y_cal), args.alpha); aps = evaluate_sets(test_probability, y_test, threshold)
        pa, lateral = VIEW_CLASSES.index("PA"), VIEW_CLASSES.index("LATERAL"); invalid = [VIEW_CLASSES.index("AP"), VIEW_CLASSES.index("SUNRISE")]
        row = {
            "model": checkpoint["model_id"], "architecture": checkpoint["model_key"], "regime": checkpoint["regime"], "seed": checkpoint["seed"],
            "split": args.test_name, "temperature": temperature, "alpha": args.alpha,
            "accuracy": accuracy_score(y_test, pred),
            "macro_f1_fixed": f1_score(y_test, pred, labels=[pa, lateral], average="macro", zero_division=0),
            "invalid_view_prediction_rate": np.isin(pred, invalid).mean(),
            "ece": expected_calibration_error(test_probability, y_test), **aps,
        }
        rows.append(row); print(row, flush=True); del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(output / "track_b_conformal_OAI_test.csv", index=False)


if __name__ == "__main__": main()

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from orthobench.datasets import VIEW_CLASSES, ViewDataset
from orthobench.models import create_timm_model
from orthobench.preprocessing import PreprocessingConfig
from orthobench.sensitivity import paired_mask_metrics


SPLITS = {
    "UPMC_val": ("split_view", "val"),
    "UPMC_internal_test": ("split_view", "test"),
    "OAI_test": ("split_view_external_oai", "OAI_test"),
}


def predict(model, loader, device):
    model.eval()
    ys, preds, probs, uids = [], [], [], []
    with torch.no_grad():
        for x, y, uid in loader:
            probability = torch.softmax(model(x.to(device)), dim=1).cpu().numpy()
            ys.extend(y.numpy().tolist())
            preds.extend(probability.argmax(1).tolist())
            probs.append(probability)
            uids.extend(uid)
    return np.asarray(ys), np.asarray(preds), np.vstack(probs), np.asarray(uids)


def main():
    ap = argparse.ArgumentParser(description="Evaluate fixed checkpoints under paired peripheral masking.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--path_col", default=None)
    ap.add_argument("--splits", nargs="+", choices=sorted(SPLITS), default=["UPMC_internal_test", "OAI_test"])
    ap.add_argument("--mask_modes", nargs="+", choices=["none", "border", "corners", "border_corners"], default=["none", "border", "corners", "border_corners"])
    ap.add_argument("--border_fraction", type=float, default=None, help="Override checkpoint value for the sensitivity analysis.")
    ap.add_argument("--corner_fraction", type=float, default=None, help="Override checkpoint value for the sensitivity analysis.")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--allow_masked_training", action="store_true")
    args = ap.parse_args()
    if "none" not in args.mask_modes:
        raise ValueError("mask_modes must include none as the paired reference")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest, low_memory=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_rows = []

    for ckpt_path in sorted(Path(args.ckpt_dir).glob("*.pt")):
        ckpt = torch.load(ckpt_path, map_location=device)
        model_name = ckpt.get("model_name", ckpt_path.stem)
        model_id = ckpt.get("model_id", ckpt_path.stem)
        base_cfg = PreprocessingConfig.from_dict(ckpt.get("preprocessing"), ckpt.get("img_size", 224))
        if base_cfg.mask_mode != "none" and not args.allow_masked_training:
            raise ValueError(
                f"{ckpt_path.name} was trained with mask_mode={base_cfg.mask_mode}; "
                "use an unmasked checkpoint or pass --allow_masked_training."
            )
        model = create_timm_model(model_name, num_classes=len(VIEW_CLASSES), pretrained=False).to(device)
        model.load_state_dict(ckpt["model"])

        for split_name in args.splits:
            split_col, split_value = SPLITS[split_name]
            frame = manifest[(manifest[split_col] == split_value) & (manifest["has_view_label"] == 1)].copy()
            outputs = {}
            expected_uids = None
            expected_y = None
            reference_df = None
            for mask_mode in args.mask_modes:
                cfg_dict = base_cfg.to_dict()
                cfg_dict["mask_mode"] = mask_mode
                if args.border_fraction is not None:
                    cfg_dict["border_fraction"] = args.border_fraction
                if args.corner_fraction is not None:
                    cfg_dict["corner_fraction"] = args.corner_fraction
                dataset = ViewDataset(frame, train=False, path_col=args.path_col, preprocessing=cfg_dict)
                loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, pin_memory=True)
                y, pred, prob, uids = predict(model, loader, device)
                if expected_uids is None:
                    expected_uids, expected_y = uids, y
                    reference_df = dataset.df.reset_index(drop=True).copy()
                elif not (np.array_equal(uids, expected_uids) and np.array_equal(y, expected_y)):
                    raise RuntimeError("Evaluation order changed between masking conditions")
                outputs[mask_mode] = (pred, prob)

            pred_none, prob_none = outputs["none"]
            prediction_table = pd.DataFrame({"uid": expected_uids, "y": expected_y, "pred_none": pred_none})
            prediction_table["true_class"] = [VIEW_CLASSES[i] for i in expected_y]
            for column in ["group_id", "source", "view_label", "side_label", dataset.path_col]:
                if column in reference_df.columns and column not in prediction_table.columns:
                    output_name = "image_path" if column == dataset.path_col else column
                    prediction_table[output_name] = reference_df[column].astype(str).values
            for mask_mode in args.mask_modes:
                pred, prob = outputs[mask_mode]
                metrics = paired_mask_metrics(expected_y, pred_none, prob_none, pred, prob)
                metrics.update({
                    "model": model_id,
                    "architecture": model_name,
                    "trained_mask_mode": base_cfg.mask_mode,
                    "evaluation_mask_mode": mask_mode,
                    "split": split_name,
                    "border_fraction": args.border_fraction if args.border_fraction is not None else base_cfg.border_fraction,
                    "corner_fraction": args.corner_fraction if args.corner_fraction is not None else base_cfg.corner_fraction,
                })
                all_rows.append(metrics)
                prediction_table[f"pred_{mask_mode}"] = pred
                prediction_table[f"confidence_{mask_mode}"] = prob.max(1)
                prediction_table[f"true_class_probability_{mask_mode}"] = prob[np.arange(len(expected_y)), expected_y]
                print(metrics)
            prediction_table.to_csv(out_dir / f"paired_predictions_{model_id}_{split_name}.csv", index=False)

    result = pd.DataFrame(all_rows)
    result.to_csv(out_dir / "peripheral_sensitivity_summary.csv", index=False)
    print(f"Wrote {out_dir / 'peripheral_sensitivity_summary.csv'} rows={len(result)}")


if __name__ == "__main__":
    main()

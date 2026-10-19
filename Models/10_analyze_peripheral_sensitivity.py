from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

VIEW_CLASSES = ["AP", "PA", "LATERAL", "SUNRISE"]
MASK_MODES = ["none", "border", "corners", "border_corners"]


def attach_manifest_metadata(frame, manifest, path_col):
    wanted = [c for c in ["uid", "group_id", "source", "view_label", "side_label", path_col] if c in manifest]
    metadata = manifest[wanted].drop_duplicates("uid")
    if path_col in metadata and path_col != "image_path":
        metadata = metadata.rename(columns={path_col: "image_path"})
    missing = [c for c in metadata.columns if c != "uid" and c not in frame]
    return frame.merge(metadata[["uid", *missing]], on="uid", how="left", validate="one_to_one")


def calculate_metrics(frame, mask):
    y = frame["y"].to_numpy(dtype=int)
    pred = frame[f"pred_{mask}"].to_numpy(dtype=int)
    present = np.unique(y)
    base = frame["pred_none"].to_numpy(dtype=int)
    correct_base = base == y
    correct_mask = pred == y
    return {
        "accuracy": accuracy_score(y, pred),
        "macro_f1_present_classes": f1_score(y, pred, labels=present, average="macro", zero_division=0),
        "macro_f1_all_four_classes": f1_score(y, pred, labels=np.arange(len(VIEW_CLASSES)), average="macro", zero_division=0),
        "weighted_f1": f1_score(y, pred, labels=present, average="weighted", zero_division=0),
        "prediction_flip_rate": np.mean(pred != base),
        "correct_to_incorrect_rate": np.mean(correct_base & ~correct_mask),
        "incorrect_to_correct_rate": np.mean(~correct_base & correct_mask),
        "mean_delta_true_class_probability": np.mean(
            frame[f"true_class_probability_{mask}"] - frame["true_class_probability_none"]
        ),
    }


def grouped_bootstrap(frame, masks, replicates, seed):
    rng = np.random.default_rng(seed)
    groups = frame["group_id"].astype(str).unique()
    group_rows = {g: np.flatnonzero(frame["group_id"].astype(str).to_numpy() == g) for g in groups}
    point = {mask: calculate_metrics(frame, mask) for mask in masks}
    values = {(mask, metric): [] for mask in masks for metric in point[mask]}
    delta_values = {(mask, metric): [] for mask in masks if mask != "none" for metric in point[mask]}
    for _ in range(replicates):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        sampled_indices = np.concatenate([group_rows[g] for g in sampled_groups])
        sample = frame.iloc[sampled_indices]
        sampled_metrics = {mask: calculate_metrics(sample, mask) for mask in masks}
        for mask in masks:
            for metric, value in sampled_metrics[mask].items():
                values[(mask, metric)].append(value)
                if mask != "none":
                    delta_values[(mask, metric)].append(value - sampled_metrics["none"][metric])
    rows = []
    for mask in masks:
        for metric, estimate in point[mask].items():
            distribution = np.asarray(values[(mask, metric)])
            rows.append({
                "evaluation_mask_mode": mask,
                "metric": metric,
                "estimate": estimate,
                "ci_lower": np.percentile(distribution, 2.5),
                "ci_upper": np.percentile(distribution, 97.5),
                "delta_vs_none": 0.0 if mask == "none" else estimate - point["none"][metric],
                "delta_ci_lower": 0.0 if mask == "none" else np.percentile(delta_values[(mask, metric)], 2.5),
                "delta_ci_upper": 0.0 if mask == "none" else np.percentile(delta_values[(mask, metric)], 97.5),
                "bootstrap_unit": "group_id",
                "bootstrap_replicates": replicates,
            })
    return pd.DataFrame(rows)


def save_confusions(frame, model_id, split, out_dir, masks):
    y = frame["y"].to_numpy(dtype=int)
    count_rows, class_rows = [], []
    for mask in masks:
        pred = frame[f"pred_{mask}"].to_numpy(dtype=int)
        matrix = confusion_matrix(y, pred, labels=np.arange(len(VIEW_CLASSES)))
        for actual_i, actual in enumerate(VIEW_CLASSES):
            for predicted_i, predicted in enumerate(VIEW_CLASSES):
                count_rows.append({"model": model_id, "split": split, "mask_mode": mask,
                                   "actual": actual, "predicted": predicted,
                                   "count": int(matrix[actual_i, predicted_i])})
        precision, recall, f1, support = precision_recall_fscore_support(
            y, pred, labels=np.arange(len(VIEW_CLASSES)), zero_division=0
        )
        for i, name in enumerate(VIEW_CLASSES):
            class_rows.append({"model": model_id, "split": split, "mask_mode": mask,
                               "class": name, "precision": precision[i], "recall": recall[i],
                               "f1": f1[i], "support": int(support[i])})
        fig, ax = plt.subplots(figsize=(6, 5))
        image = ax.imshow(matrix, cmap="Blues")
        for i in range(len(VIEW_CLASSES)):
            for j in range(len(VIEW_CLASSES)):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
        ax.set(xticks=range(4), yticks=range(4), xticklabels=VIEW_CLASSES, yticklabels=VIEW_CLASSES,
               xlabel="Predicted view", ylabel="True view", title=f"{model_id} | {split} | {mask}")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(out_dir / f"confusion_{model_id}_{split}_{mask}.png", dpi=180)
        plt.close(fig)
    pd.DataFrame(count_rows).to_csv(out_dir / f"confusion_counts_{model_id}_{split}.csv", index=False)
    pd.DataFrame(class_rows).to_csv(out_dir / f"per_class_metrics_{model_id}_{split}.csv", index=False)


def create_audit_sample(frame, n_changed, n_unchanged, seed):
    comparison_masks = [m for m in MASK_MODES if f"pred_{m}" in frame and m != "none"]
    audit = frame.copy()
    for mask in comparison_masks:
        audit[f"changed_{mask}"] = audit[f"pred_{mask}"] != audit["pred_none"]
    audit["changed_any"] = audit[[f"changed_{m}" for m in comparison_masks]].any(axis=1)
    rng = np.random.default_rng(seed)
    pieces = []
    for status, requested in [(True, n_changed), (False, n_unchanged)]:
        eligible = audit[audit["changed_any"] == status].copy()
        eligible["_random"] = rng.random(len(eligible))
        # Prefer one image per patient group so the audit covers more independent cases.
        selected = eligible.sort_values("_random").drop_duplicates("group_id").head(requested)
        selected["audit_sampling_stratum"] = "changed" if status else "unchanged"
        pieces.append(selected)
    result = pd.concat(pieces, ignore_index=True).drop(columns="_random")
    result["audit_reviewer"] = ""
    result["audit_view_text_present"] = ""          # yes / no / uncertain
    result["audit_view_text_content"] = ""
    result["audit_laterality_marker_present"] = "" # yes / no / uncertain
    result["audit_laterality_marker_content"] = ""
    result["audit_prominent_border_present"] = ""  # yes / no / uncertain
    result["audit_mask_removes_annotation"] = ""   # yes / no / uncertain
    result["audit_mask_removes_relevant_anatomy"] = ""  # yes / no / uncertain
    result["audit_peripheral_anatomy_loss"] = ""    # none / mild / moderate / severe
    result["audit_notes"] = ""
    result["audit_sampling_seed"] = seed
    return result


def main():
    ap = argparse.ArgumentParser(description="Analyze paired peripheral-occlusion predictions.")
    ap.add_argument("--sensitivity_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--path_col", default="png_path")
    ap.add_argument("--audit_split", default="OAI_test")
    ap.add_argument("--audit_changed", type=int, default=100)
    ap.add_argument("--audit_unchanged", type=int, default=100)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--bootstrap_replicates", type=int, default=2000)
    args = ap.parse_args()
    sensitivity_dir, out_dir = Path(args.sensitivity_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest, low_memory=False)
    audit_outputs = []
    for prediction_path in sorted(sensitivity_dir.glob("paired_predictions_*.csv")):
        frame = pd.read_csv(prediction_path, low_memory=False)
        frame = attach_manifest_metadata(frame, manifest, args.path_col)
        if "group_id" not in frame or frame["group_id"].isna().any():
            raise ValueError(f"Missing group_id after manifest merge for {prediction_path.name}")
        stem = prediction_path.stem.removeprefix("paired_predictions_")
        split = next((s for s in ["UPMC_internal_test", "OAI_test", "UPMC_val"] if stem.endswith(f"_{s}")), None)
        if split is None:
            raise ValueError(f"Cannot infer split from {prediction_path.name}")
        model_id = stem[:-(len(split) + 1)]
        masks = [m for m in MASK_MODES if f"pred_{m}" in frame]
        save_confusions(frame, model_id, split, out_dir, masks)
        bootstrap = grouped_bootstrap(frame, masks, args.bootstrap_replicates, args.seed)
        bootstrap.insert(0, "split", split)
        bootstrap.insert(0, "model", model_id)
        bootstrap.to_csv(out_dir / f"group_bootstrap_{model_id}_{split}.csv", index=False)
        if split == args.audit_split:
            sample = create_audit_sample(frame, args.audit_changed, args.audit_unchanged, args.seed)
            sample.insert(0, "split", split)
            sample.insert(0, "model", model_id)
            audit_outputs.append(sample)
    if audit_outputs:
        combined = pd.concat(audit_outputs, ignore_index=True)
        combined.to_csv(out_dir / "manual_audit_sample.csv", index=False)
        print(f"Wrote {out_dir / 'manual_audit_sample.csv'} rows={len(combined)}")
    print(f"Peripheral analysis completed: {out_dir}")


if __name__ == "__main__":
    main()

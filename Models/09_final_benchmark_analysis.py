#!/usr/bin/env python3
"""Final analysis for OrthoRegistryBench v3 seed-fixed experiments.

This script:
  1. Recomputes OAI metrics with a fixed PA/LATERAL label set.
  2. Reports invalid AP/SUNRISE prediction rates on OAI.
  3. Runs patient/study-group clustered bootstrap confidence intervals.
  4. Re-evaluates and caches predictions for the 100 adapted checkpoints.
  5. Produces separate baseline, maintenance, conformal, and domain tables.

Run from the OrthoRegistryBench project root so ``orthobench`` is importable.
The analysis never retrains a model.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Iterable

# When this file is installed as ``PROJECT_ROOT/scripts/09_final_benchmark_analysis.py``,
# Python puts ``PROJECT_ROOT/scripts`` on sys.path but not ``PROJECT_ROOT``.  Add the
# first nearby directory containing the local ``orthobench`` package so the script
# works both through run_benchmark.sh and as a direct command.
_SCRIPT_PATH = Path(__file__).resolve()
for _candidate in (_SCRIPT_PATH.parent, _SCRIPT_PATH.parent.parent):
    if (_candidate / "orthobench").is_dir():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break

import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.metrics import accuracy_score, f1_score


VIEW_CLASSES = ["AP", "PA", "LATERAL", "SUNRISE"]
CLASS_TO_INDEX = {name: i for i, name in enumerate(VIEW_CLASSES)}
PA_INDEX = CLASS_TO_INDEX["PA"]
LATERAL_INDEX = CLASS_TO_INDEX["LATERAL"]
OAI_PRESENT = [PA_INDEX, LATERAL_INDEX]
OAI_INVALID = [CLASS_TO_INDEX["AP"], CLASS_TO_INDEX["SUNRISE"]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create publication-ready seed-fixed benchmark summaries."
    )
    parser.add_argument("--out", required=True, help="Seed-fixed benchmark output directory")
    parser.add_argument(
        "--manifest",
        default=None,
        help="Derived v3 manifest; defaults to OUT/orthoregistrybench_manifest_v3.csv",
    )
    parser.add_argument("--analysis_dir", default=None, help="Defaults to OUT/final_analysis")
    parser.add_argument("--path_col", default="png_path")
    parser.add_argument("--group_col", default="group_id")
    parser.add_argument("--oai_split_col", default="split_view_external_oai")
    parser.add_argument("--oai_test_name", default="OAI_test")
    parser.add_argument("--retention_split_col", default="split_view")
    parser.add_argument("--retention_name", default="test")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap_seed", type=int, default=2026)
    parser.add_argument(
        "--force_inference",
        action="store_true",
        help="Overwrite cached per-image predictions",
    )
    parser.add_argument(
        "--skip_inference",
        action="store_true",
        help="Only summarize existing prediction caches",
    )
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, columns: Iterable[str], source: Path | str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns in {source}: {missing}")


def normalize_view_label(value: object) -> str:
    text = str(value).strip().upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "LAT": "LATERAL",
        "LATERAL_VIEW": "LATERAL",
        "SUNRISE_VIEW": "SUNRISE",
        "MERCHANT": "SUNRISE",
        "SKYLINE": "SUNRISE",
        "PATELLOFEMORAL": "SUNRISE",
    }
    return aliases.get(text, text)


def architecture_from_model_id(model_id: str) -> str:
    base = re.sub(r"^fewshot_", "", str(model_id))
    match = re.match(r"(.+?)_seed\d+", base)
    return match.group(1) if match else base


def base_seed_from_model_id(model_id: str) -> int | None:
    match = re.search(r"_seed(\d+)", str(model_id))
    return int(match.group(1)) if match else None


def metrics_from_arrays(y: np.ndarray, pred: np.ndarray, oai: bool) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    labels = OAI_PRESENT if oai else list(range(len(VIEW_CLASSES)))
    result = {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1_fixed": float(
            f1_score(y, pred, labels=labels, average="macro", zero_division=0)
        ),
        "weighted_f1_fixed": float(
            f1_score(y, pred, labels=labels, average="weighted", zero_division=0)
        ),
    }
    for label, name in [(PA_INDEX, "pa_f1"), (LATERAL_INDEX, "lateral_f1")]:
        result[name] = float(
            f1_score(y, pred, labels=[label], average="macro", zero_division=0)
        )
    result["invalid_view_prediction_rate"] = (
        float(np.isin(pred, OAI_INVALID).mean()) if oai else float("nan")
    )
    return result


def confusion_by_group(frame: pd.DataFrame, group_col: str) -> tuple[np.ndarray, np.ndarray]:
    require_columns(frame, [group_col, "y", "pred"], "prediction table")
    work = frame[[group_col, "y", "pred"]].dropna().copy()
    work["y"] = work["y"].astype(int)
    work["pred"] = work["pred"].astype(int)
    groups = work[group_col].astype(str).unique()
    matrices = np.zeros((len(groups), len(VIEW_CLASSES), len(VIEW_CLASSES)), dtype=np.int64)
    group_lookup = {group: i for i, group in enumerate(groups)}
    for group, y_value, pred_value in work[[group_col, "y", "pred"]].itertuples(index=False):
        matrices[group_lookup[str(group)], int(y_value), int(pred_value)] += 1
    return groups, matrices


def metrics_from_confusion(matrix: np.ndarray, oai: bool) -> dict[str, float]:
    matrix = np.asarray(matrix, dtype=float)
    total = matrix.sum()
    accuracy = np.trace(matrix) / total if total else np.nan
    tp = np.diag(matrix)
    true_count = matrix.sum(axis=1)
    pred_count = matrix.sum(axis=0)
    denominator = 2 * tp + (pred_count - tp) + (true_count - tp)
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)
    labels = OAI_PRESENT if oai else list(range(len(VIEW_CLASSES)))
    label_true = true_count[labels]
    weighted = (
        np.average(f1[labels], weights=label_true) if label_true.sum() else np.nan
    )
    invalid = matrix[:, OAI_INVALID].sum() / total if oai and total else np.nan
    return {
        "accuracy": float(accuracy),
        "macro_f1_fixed": float(np.mean(f1[labels])),
        "weighted_f1_fixed": float(weighted),
        "pa_f1": float(f1[PA_INDEX]),
        "lateral_f1": float(f1[LATERAL_INDEX]),
        "invalid_view_prediction_rate": float(invalid),
    }


def clustered_bootstrap(
    frame: pd.DataFrame,
    group_col: str,
    oai: bool,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    groups, matrices = confusion_by_group(frame, group_col)
    if len(groups) < 2:
        return {}
    rng = np.random.default_rng(seed)
    metric_names = list(metrics_from_confusion(matrices.sum(axis=0), oai))
    samples = {name: np.empty(replicates, dtype=float) for name in metric_names}
    # Multinomial group multiplicities are equivalent to resampling groups with replacement.
    for i in range(replicates):
        multiplicities = rng.multinomial(len(groups), np.repeat(1 / len(groups), len(groups)))
        matrix = np.tensordot(multiplicities, matrices, axes=(0, 0))
        values = metrics_from_confusion(matrix, oai)
        for name in metric_names:
            samples[name][i] = values[name]
    output: dict[str, float] = {}
    for name, values in samples.items():
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            # Some metrics are deliberately undefined for a split. For example,
            # invalid AP/SUNRISE predictions apply to the two-class OAI test but
            # not to the four-class UPMC internal test.
            output[f"{name}_ci_low"] = float("nan")
            output[f"{name}_ci_high"] = float("nan")
        else:
            output[f"{name}_ci_low"] = float(np.quantile(finite, 0.025))
            output[f"{name}_ci_high"] = float(np.quantile(finite, 0.975))
    return output


def attach_metadata(predictions: pd.DataFrame, subset: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if group_col in predictions.columns and predictions[group_col].notna().all():
        return predictions
    metadata = [column for column in ["uid", group_col, "source", "view_label", "side_label"] if column in subset]
    return predictions.merge(
        subset[metadata].drop_duplicates("uid"), on="uid", how="left", validate="one_to_one"
    )


def load_baseline_oai_predictions(
    root: Path,
    manifest: pd.DataFrame,
    split_col: str,
    split_name: str,
    group_col: str,
) -> list[tuple[str, pd.DataFrame]]:
    subset = manifest[(manifest[split_col] == split_name) & (manifest["has_view_label"] == 1)].copy()
    paths = sorted((root / "results").glob(f"pred_view_*_{split_name}.csv"))
    if not paths:
        raise FileNotFoundError(f"No baseline OAI prediction CSVs in {root / 'results'}")
    output = []
    suffix = f"_{split_name}.csv"
    for path in paths:
        model_id = path.name[len("pred_view_") : -len(suffix)]
        frame = attach_metadata(pd.read_csv(path, low_memory=False), subset, group_col)
        output.append((model_id, frame))
    return output


def make_prediction_loader(
    frame: pd.DataFrame,
    img_size: int,
    path_col: str,
    preprocessing: dict | None,
    batch: int,
    workers: int,
):
    import torch
    from torch.utils.data import DataLoader
    from orthobench.datasets import ViewDataset

    dataset = ViewDataset(frame, False, img_size, path_col, preprocessing)
    return DataLoader(
        dataset,
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


def infer_checkpoint(
    checkpoint_path: Path,
    subset: pd.DataFrame,
    path_col: str,
    group_col: str,
    batch: int,
    workers: int,
) -> pd.DataFrame:
    import torch
    from orthobench.models import create_timm_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_name = checkpoint.get("model_name", architecture_from_model_id(checkpoint_path.stem))
    model = create_timm_model(model_name, num_classes=len(VIEW_CLASSES), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    loader = make_prediction_loader(
        subset,
        int(checkpoint.get("img_size", 224)),
        path_col,
        checkpoint.get("preprocessing"),
        batch,
        workers,
    )
    ys: list[int] = []
    preds: list[int] = []
    uids: list[str] = []
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for images, labels, batch_uids in loader:
            probability = torch.softmax(model(images.to(device)), dim=1).cpu().numpy()
            ys.extend(labels.numpy().astype(int).tolist())
            preds.extend(probability.argmax(axis=1).astype(int).tolist())
            uids.extend([str(value) for value in batch_uids])
            probabilities.append(probability)
    probability_array = np.vstack(probabilities)
    prediction = pd.DataFrame(
        {
            "uid": uids,
            "y": ys,
            "pred": preds,
            **{
                f"prob_{class_name}": probability_array[:, index]
                for index, class_name in enumerate(VIEW_CLASSES)
            },
        }
    )
    return attach_metadata(prediction, subset, group_col)


def prediction_for_checkpoint(
    checkpoint: Path,
    split_key: str,
    subset: pd.DataFrame,
    cache_dir: Path,
    args: argparse.Namespace,
) -> pd.DataFrame:
    cache_path = cache_dir / f"pred_{checkpoint.stem}_{split_key}.csv"
    if cache_path.exists() and not args.force_inference:
        return pd.read_csv(cache_path, low_memory=False)
    if args.skip_inference:
        raise FileNotFoundError(f"Prediction cache missing with --skip_inference: {cache_path}")
    prediction = infer_checkpoint(
        checkpoint,
        subset,
        args.path_col,
        args.group_col,
        args.batch,
        args.num_workers,
    )
    prediction.to_csv(cache_path, index=False)
    return prediction


def summarize_architectures(
    frame: pd.DataFrame,
    group_columns: list[str],
    metric_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row["n_runs"] = len(group)
        for metric in metric_columns:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            n = len(values)
            mean = float(values.mean()) if n else np.nan
            sd = float(values.std(ddof=1)) if n > 1 else np.nan
            half_width = float(t.ppf(0.975, n - 1) * sd / math.sqrt(n)) if n > 1 else np.nan
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = sd
            row[f"{metric}_ci_low"] = mean - half_width if np.isfinite(half_width) else np.nan
            row[f"{metric}_ci_high"] = mean + half_width if np.isfinite(half_width) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def analyze_baseline(
    root: Path,
    manifest: pd.DataFrame,
    analysis_dir: Path,
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows = []
    predictions = load_baseline_oai_predictions(
        root, manifest, args.oai_split_col, args.oai_test_name, args.group_col
    )
    for model_id, frame in predictions:
        point = metrics_from_arrays(frame["y"].to_numpy(), frame["pred"].to_numpy(), oai=True)
        ci = clustered_bootstrap(
            frame,
            args.group_col,
            True,
            args.bootstrap,
            args.bootstrap_seed + (base_seed_from_model_id(model_id) or 0),
        )
        rows.append(
            {
                "model": model_id,
                "architecture": architecture_from_model_id(model_id),
                "seed": base_seed_from_model_id(model_id),
                "split": args.oai_test_name,
                "n_images": len(frame),
                "n_groups": frame[args.group_col].nunique(),
                **point,
                **ci,
            }
        )
    seed_metrics = pd.DataFrame(rows).sort_values(["architecture", "seed"])
    seed_metrics.to_csv(analysis_dir / "baseline_oai_seed_metrics.csv", index=False)
    metric_columns = [
        "accuracy",
        "macro_f1_fixed",
        "weighted_f1_fixed",
        "pa_f1",
        "lateral_f1",
        "invalid_view_prediction_rate",
    ]
    summary = summarize_architectures(seed_metrics, ["architecture", "split"], metric_columns)
    summary.to_csv(analysis_dir / "baseline_oai_architecture_summary.csv", index=False)
    return seed_metrics


def analyze_baseline_internal(
    root: Path,
    manifest: pd.DataFrame,
    analysis_dir: Path,
    args: argparse.Namespace,
) -> pd.DataFrame:
    subset = manifest[
        (manifest[args.retention_split_col] == args.retention_name)
        & (manifest["has_view_label"] == 1)
    ].copy()
    checkpoints = sorted((root / "view_ckpts").glob("*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No baseline checkpoints in {root / 'view_ckpts'}")
    cache_dir = analysis_dir / "prediction_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for checkpoint in checkpoints:
        model_id = checkpoint.stem
        prediction = prediction_for_checkpoint(
            checkpoint, "UPMC_internal_test", subset, cache_dir, args
        )
        point = metrics_from_arrays(
            prediction["y"].to_numpy(), prediction["pred"].to_numpy(), oai=False
        )
        ci = clustered_bootstrap(
            prediction,
            args.group_col,
            False,
            args.bootstrap,
            args.bootstrap_seed + 25_000 + (base_seed_from_model_id(model_id) or 0),
        )
        rows.append(
            {
                "model": model_id,
                "architecture": architecture_from_model_id(model_id),
                "seed": base_seed_from_model_id(model_id),
                "split": "UPMC_internal_test",
                "n_images": len(prediction),
                "n_groups": prediction[args.group_col].nunique(),
                **point,
                **ci,
            }
        )
    seed_metrics = pd.DataFrame(rows).sort_values(["architecture", "seed"])
    seed_metrics.to_csv(analysis_dir / "baseline_internal_seed_metrics.csv", index=False)
    metric_columns = ["accuracy", "macro_f1_fixed", "weighted_f1_fixed"]
    summary = summarize_architectures(seed_metrics, ["architecture", "split"], metric_columns)
    summary.to_csv(analysis_dir / "baseline_internal_architecture_summary.csv", index=False)
    return seed_metrics


def maintenance_metadata(root: Path) -> pd.DataFrame:
    paths = sorted((root / "view_fewshot").glob("seed*/view_fewshot_maintenance_seed*.csv"))
    if not paths:
        raise FileNotFoundError(f"No view-maintenance CSVs under {root / 'view_fewshot'}")
    return pd.concat([pd.read_csv(path, low_memory=False) for path in paths], ignore_index=True)


def analyze_view_maintenance(
    root: Path,
    manifest: pd.DataFrame,
    analysis_dir: Path,
    baseline: pd.DataFrame,
    baseline_internal: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    metadata = maintenance_metadata(root)
    require_columns(
        metadata,
        ["model", "architecture", "base_model_id", "base_seed", "requested_shots_per_class", "ckpt"],
        "view maintenance results",
    )
    oai_subset = manifest[
        (manifest[args.oai_split_col] == args.oai_test_name) & (manifest["has_view_label"] == 1)
    ].copy()
    retention_subset = manifest[
        (manifest[args.retention_split_col] == args.retention_name)
        & (manifest["has_view_label"] == 1)
    ].copy()
    cache_dir = analysis_dir / "prediction_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    baseline_lookup = baseline.set_index("model").to_dict("index")
    baseline_internal_lookup = baseline_internal.set_index("model").to_dict("index")
    rows = []
    for index, record in metadata.iterrows():
        checkpoint = Path(str(record["ckpt"]))
        if not checkpoint.exists():
            # Support moving the OUT directory after the experiment.
            candidates = list((root / "view_fewshot").glob(f"seed*/{record['model']}.pt"))
            if len(candidates) != 1:
                raise FileNotFoundError(f"Cannot resolve adapted checkpoint: {record['ckpt']}")
            checkpoint = candidates[0]
        oai_prediction = prediction_for_checkpoint(
            checkpoint, "OAI_test", oai_subset, cache_dir, args
        )
        retention_prediction = prediction_for_checkpoint(
            checkpoint, "UPMC_internal_test", retention_subset, cache_dir, args
        )
        oai_point = metrics_from_arrays(
            oai_prediction["y"].to_numpy(), oai_prediction["pred"].to_numpy(), oai=True
        )
        retention_point = metrics_from_arrays(
            retention_prediction["y"].to_numpy(), retention_prediction["pred"].to_numpy(), oai=False
        )
        seed_offset = int(record["base_seed"]) * 10_000 + int(record["requested_shots_per_class"])
        oai_ci = clustered_bootstrap(
            oai_prediction,
            args.group_col,
            True,
            args.bootstrap,
            args.bootstrap_seed + seed_offset,
        )
        retention_ci = clustered_bootstrap(
            retention_prediction,
            args.group_col,
            False,
            args.bootstrap,
            args.bootstrap_seed + 50_000 + seed_offset,
        )
        base = baseline_lookup[str(record["base_model_id"])]
        base_internal = baseline_internal_lookup[str(record["base_model_id"])]
        row = {
            "model": record["model"],
            "architecture": record["architecture"],
            "base_model_id": record["base_model_id"],
            "base_seed": int(record["base_seed"]),
            "adapt_seed": int(record.get("adapt_seed", record["base_seed"])),
            "requested_shots_per_class": int(record["requested_shots_per_class"]),
            "n_fewshot_images": int(record["n_fewshot_images"]),
            "n_fewshot_groups": int(record["n_fewshot_groups"]),
        }
        for name in VIEW_CLASSES:
            column = f"actual_{name}"
            if column in record:
                row[column] = int(record[column])
        for key, value in oai_point.items():
            row[f"oai_{key}"] = value
            row[f"baseline_oai_{key}"] = base[key]
            row[f"delta_oai_{key}"] = value - base[key]
        for key, value in retention_point.items():
            row[f"retention_{key}"] = value
            row[f"baseline_retention_{key}"] = base_internal[key]
            row[f"delta_retention_{key}"] = value - base_internal[key]
        for key, value in oai_ci.items():
            row[f"oai_{key}"] = value
        for key, value in retention_ci.items():
            row[f"retention_{key}"] = value
        rows.append(row)
        print(f"Analyzed {index + 1}/{len(metadata)}: {record['model']}", flush=True)
    seed_metrics = pd.DataFrame(rows).sort_values(
        ["architecture", "requested_shots_per_class", "base_seed"]
    )
    seed_metrics.to_csv(analysis_dir / "view_maintenance_seed_metrics.csv", index=False)
    metric_columns = [
        "oai_accuracy",
        "oai_macro_f1_fixed",
        "oai_weighted_f1_fixed",
        "oai_pa_f1",
        "oai_lateral_f1",
        "oai_invalid_view_prediction_rate",
        "delta_oai_accuracy",
        "delta_oai_macro_f1_fixed",
        "delta_oai_weighted_f1_fixed",
        "delta_oai_invalid_view_prediction_rate",
        "retention_accuracy",
        "retention_macro_f1_fixed",
        "retention_weighted_f1_fixed",
        "delta_retention_accuracy",
        "delta_retention_macro_f1_fixed",
        "delta_retention_weighted_f1_fixed",
    ]
    summary = summarize_architectures(
        seed_metrics,
        ["architecture", "requested_shots_per_class"],
        metric_columns,
    )
    summary.to_csv(analysis_dir / "view_maintenance_architecture_budget_summary.csv", index=False)


def analyze_conformal(root: Path, analysis_dir: Path) -> None:
    path = root / "results" / "view_conformal_OAI_test.csv"
    if not path.exists():
        print(f"WARNING: conformal results not found: {path}", file=sys.stderr)
        return
    frame = pd.read_csv(path, low_memory=False)
    frame["seed"] = frame["model"].map(base_seed_from_model_id)
    frame["architecture"] = frame["model"].map(architecture_from_model_id)
    frame.to_csv(analysis_dir / "conformal_seed_metrics.csv", index=False)
    metric_columns = [
        column
        for column in ["accuracy", "macro_f1", "weighted_f1", "ece", "coverage", "singleton_rate", "avg_set_size"]
        if column in frame.columns
    ]
    summary = summarize_architectures(frame, ["architecture", "setting"], metric_columns)
    summary.to_csv(analysis_dir / "conformal_architecture_summary.csv", index=False)


def analyze_domain_maintenance(root: Path, analysis_dir: Path) -> None:
    path = root / "domain_maintenance" / "results" / "domain_fracatlas_maintenance.csv"
    if not path.exists():
        print(f"WARNING: domain-maintenance results not found: {path}", file=sys.stderr)
        return
    frame = pd.read_csv(path, low_memory=False)
    frame["architecture"] = frame["feature_model"].map(architecture_from_model_id)
    frame["base_seed"] = frame["feature_model"].map(base_seed_from_model_id)
    frame = frame.rename(columns={"seed": "maintenance_seed"})
    frame.to_csv(analysis_dir / "domain_maintenance_run_metrics.csv", index=False)
    metric_columns = [
        column
        for column in [
            "accuracy",
            "macro_f1",
            "weighted_f1",
            "f1_pos1",
            "auroc",
            "auprc",
            "triage_accept_rate",
            "triage_reject_rate",
            "triage_review_rate",
            "triage_accept_purity",
            "triage_auto_accept_knee_recall",
            "triage_reject_contamination_knee_fraction",
            "triage_knee_false_rejection_rate",
        ]
        if column in frame.columns
    ]
    # Five maintenance seeds are crossed with five feature-extractor seeds. They are
    # not 25 independent external cohorts. Average maintenance seeds within each
    # feature checkpoint first, then summarize the five base-model seeds.
    base_seed_means = (
        frame.groupby(
            ["architecture", "feature_model", "domain_model", "setting", "base_seed"],
            as_index=False,
        )[metric_columns]
        .mean(numeric_only=True)
    )
    base_seed_means.to_csv(
        analysis_dir / "domain_maintenance_base_seed_means.csv", index=False
    )
    summary = summarize_architectures(
        base_seed_means, ["architecture", "domain_model", "setting"], metric_columns
    )
    summary.to_csv(analysis_dir / "domain_maintenance_architecture_summary.csv", index=False)
    paired = frame.pivot_table(
        index=["architecture", "feature_model", "domain_model", "base_seed", "maintenance_seed"],
        columns="setting",
        values=metric_columns,
        aggfunc="first",
    )
    delta_rows = []
    if {"baseline_no_injection", "fewshot_injection"}.issubset(paired.columns.get_level_values(1)):
        for metric in metric_columns:
            delta = paired[(metric, "fewshot_injection")] - paired[(metric, "baseline_no_injection")]
            temp = delta.rename(f"delta_{metric}").reset_index()
            delta_rows.append(temp)
    if delta_rows:
        deltas = delta_rows[0]
        for temp in delta_rows[1:]:
            deltas = deltas.merge(
                temp,
                on=["architecture", "feature_model", "domain_model", "base_seed", "maintenance_seed"],
                how="outer",
            )
        deltas.to_csv(analysis_dir / "domain_maintenance_paired_deltas.csv", index=False)
        delta_metrics = [column for column in deltas if column.startswith("delta_")]
        delta_base_seed_means = (
            deltas.groupby(
                ["architecture", "feature_model", "domain_model", "base_seed"],
                as_index=False,
            )[delta_metrics]
            .mean(numeric_only=True)
        )
        delta_base_seed_means.to_csv(
            analysis_dir / "domain_maintenance_paired_delta_base_seed_means.csv",
            index=False,
        )
        delta_summary = summarize_architectures(
            delta_base_seed_means, ["architecture", "domain_model"], delta_metrics
        )
        delta_summary.to_csv(
            analysis_dir / "domain_maintenance_paired_delta_summary.csv", index=False
        )


def write_readme(analysis_dir: Path, args: argparse.Namespace) -> None:
    text = f"""# OrthoRegistryBench final analysis outputs

Generated with group-clustered bootstrap sampling (`B={args.bootstrap}`, seed `{args.bootstrap_seed}`).

Primary OAI metric: `macro_f1_fixed`, calculated over the fixed present-class set
`PA` and `LATERAL`. `invalid_view_prediction_rate` is the proportion of OAI images
predicted as `AP` or `SUNRISE`.

Architecture summaries report mean, sample SD, and a t-based 95% interval across
the five fitted-model seeds. Seed-level confidence intervals are clustered by
`{args.group_col}` and are not treated as five independent patient cohorts.

Files:

- `baseline_oai_seed_metrics.csv`: corrected baseline metrics and clustered CIs.
- `baseline_oai_architecture_summary.csv`: five-seed baseline summary.
- `baseline_internal_seed_metrics.csv`: internal metrics and clustered CIs.
- `baseline_internal_architecture_summary.csv`: five-seed internal summary.
- `view_maintenance_seed_metrics.csv`: corrected adapted metrics, retention, and deltas.
- `view_maintenance_architecture_budget_summary.csv`: maintenance curves by budget.
- `conformal_seed_metrics.csv` and `conformal_architecture_summary.csv`.
- `domain_maintenance_run_metrics.csv`: all domain-maintenance fits.
- `domain_maintenance_base_seed_means.csv`: maintenance seeds averaged within each base model.
- `domain_maintenance_architecture_summary.csv`: domain summaries.
- `domain_maintenance_paired_deltas.csv`: within-run injection improvements.
- `prediction_cache/`: reusable per-image predictions for adapted checkpoints.

Do not use `all_results_long.csv` as the manuscript table; it mixes per-image and
aggregate result schemas.
"""
    (analysis_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path(args.out).resolve()
    manifest_path = (
        Path(args.manifest).resolve()
        if args.manifest
        else root / "orthoregistrybench_manifest_v3.csv"
    )
    analysis_dir = (
        Path(args.analysis_dir).resolve() if args.analysis_dir else root / "final_analysis"
    )
    analysis_dir.mkdir(parents=True, exist_ok=True)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = pd.read_csv(manifest_path, low_memory=False)
    require_columns(
        manifest,
        ["uid", args.group_col, "has_view_label", args.oai_split_col, args.retention_split_col],
        manifest_path,
    )
    print("Analyzing baseline OAI predictions...", flush=True)
    baseline = analyze_baseline(root, manifest, analysis_dir, args)
    print("Analyzing baseline internal predictions...", flush=True)
    baseline_internal = analyze_baseline_internal(root, manifest, analysis_dir, args)
    print("Analyzing view maintenance (inference is cached)...", flush=True)
    analyze_view_maintenance(root, manifest, analysis_dir, baseline, baseline_internal, args)
    print("Summarizing conformal results...", flush=True)
    analyze_conformal(root, analysis_dir)
    print("Summarizing domain maintenance...", flush=True)
    analyze_domain_maintenance(root, analysis_dir)
    write_readme(analysis_dir, args)
    print(f"Final analysis written to: {analysis_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Harmonized final analysis for OrthoRegistryBench Tracks A, B, and C.

The script reads saved per-image prediction CSV files; it does not rerun models.
It separates invalid ground truth from invalid predictions and computes metrics
from one common definition across CNNs, transformers, and zero-shot VLMs.

View label convention: 0=AP, 1=PA, 2=LATERAL, 3=SUNRISE.
Domain label convention: 0=non-knee, 1=knee (override if your project differs).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


VIEW_CLASSES = [0, 1, 2, 3]
VIEW_NAMES = {0: "AP", 1: "PA", 2: "LATERAL", 3: "SUNRISE"}
DOMAIN_CLASSES = [0, 1]
DOMAIN_NAMES = {0: "non_knee", 1: "knee"}

KNOWN_MODELS = sorted(
    [
        "tf_efficientnet_b0", "convnext_tiny", "densenet121", "resnet50",
        "dinov2_vitb14", "biomedclip_vitb16", "vit_b16", "swin_t",
        "qwen2_5_vl_3b", "clip_vitb32",
    ],
    key=len,
    reverse=True,
)

TRUE_ALIASES = ["y", "y_true", "target", "label", "true_label", "ground_truth"]
PRED_ALIASES = ["pred", "y_pred", "prediction", "pred_label", "predicted_label"]
GROUP_ALIASES = ["group_id", "patient_id", "subject_id", "case_id", "uid"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "--input", action="append", required=True, metavar="TRACK=PATH",
        help="Prediction root or CSV; repeat for A, B, and C, e.g. A=/path/prediction_cache",
    )
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_boot", type=int, default=2000,
                   help="Clustered bootstrap replicates; use 200 for a smoke test")
    p.add_argument("--bootstrap_seed", type=int, default=20260921)
    p.add_argument("--ci", type=float, default=0.95)
    p.add_argument("--strict", action="store_true",
                   help="Stop rather than skip an unreadable/unrecognized CSV")
    return p.parse_args()


def parse_inputs(items: list[str]) -> list[tuple[str, Path]]:
    result = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"--input must be TRACK=PATH, received: {item}")
        track, raw = item.split("=", 1)
        track = track.strip().upper()
        if track not in {"A", "B", "C"}:
            raise ValueError(f"Track must be A, B, or C: {track}")
        path = Path(raw).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        result.append((track, path))
    return result


def prediction_files(path: Path) -> list[Path]:
    # Per-image prediction exports in this project use the pred*.csv convention.
    # Restricting directory scans to that convention prevents long-form result
    # tables (for example all_results_long.csv) from being misread as one run.
    if path.is_file():
        files = [path]
    else:
        files = list(path.rglob("pred*.csv"))
    # Reject summaries even when a future summary happens to start with "pred".
    bad = re.compile(r"(summary|metrics|aggregate|architecture|paired|manifest|audit|confusion)", re.I)
    return sorted(p for p in files if not bad.search(p.name))


def first_column(df: pd.DataFrame, aliases: list[str]) -> str | None:
    lower = {str(c).lower(): str(c) for c in df.columns}
    for alias in aliases:
        if alias.lower() in lower:
            return lower[alias.lower()]
    return None


def numeric_labels(series: pd.Series) -> pd.Series:
    """Convert numeric or common textual labels; unknown labels become NaN."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == series.notna().sum():
        return numeric
    mapping = {
        "ap": 0, "anteroposterior": 0,
        "pa": 1, "posteroanterior": 1,
        "lateral": 2, "lat": 2,
        "sunrise": 3, "merchant": 3, "skyline": 3,
        "non_knee": 0, "non-knee": 0, "nonknee": 0, "other": 0,
        "knee": 1,
        "invalid": -1, "unknown": -1, "unparseable": -1,
    }
    text = series.astype("string").str.strip().str.lower()
    return numeric.fillna(text.map(mapping))


def infer_task(path: Path, df: pd.DataFrame) -> str:
    text = str(path).lower()
    if "domain" in text:
        return "domain"
    if "view" in text or any(str(c).lower().startswith("prob_ap") for c in df.columns):
        return "view"
    raise ValueError("cannot infer task (put 'view' or 'domain' in path/filename)")


def infer_model(path: Path, df: pd.DataFrame) -> str:
    for col in ["model", "model_key", "architecture", "encoder"]:
        if col in df.columns and df[col].notna().any():
            vals = df[col].dropna().astype(str).unique()
            if len(vals) == 1:
                return vals[0]
    text = str(path).lower()
    for model in KNOWN_MODELS:
        if model in text:
            return model
    stem = path.stem
    stem = re.sub(r"^pred(?:_track_[abc])?_", "", stem, flags=re.I)
    stem = re.split(r"_(?:seed\d+|view|domain|OAI|UPMC|internal|fracatlas)", stem, maxsplit=1, flags=re.I)[0]
    return stem


def unique_text(df: pd.DataFrame, column: str) -> str | None:
    if column not in df.columns or not df[column].notna().any():
        return None
    values = df[column].dropna().astype(str).unique()
    return values[0] if len(values) == 1 else None


def normalized_encoder(value: str, path: Path) -> str:
    """Remove seed/regime suffixes while retaining the canonical encoder key."""
    text = f"{value} {path}".lower()
    for model in KNOWN_MODELS:
        if model in text:
            return model
    value = re.sub(r"__(?:logreg|hgb|rf)$", "", value, flags=re.I)
    value = re.sub(r"_(?:full_ft|linear_probe)?_?seed\d+$", "", value, flags=re.I)
    return value


def seed_from_text(value: str | None) -> int | None:
    if not value:
        return None
    match = re.search(r"(?:^|_)seed(\d+)(?:_|\.|$)", value.lower())
    return int(match.group(1)) if match else None


def infer_metadata(track: str, path: Path, df: pd.DataFrame, task: str) -> dict:
    text = str(path)
    low = text.lower()
    seed_match = re.search(r"(?:^|_)seed(\d+)(?:_|\.|$)", low)
    budget_match = re.search(r"(?:^|_)(5|10|25|50|100)spc(?:_|\.|$)", low)
    if "oai_test" in low:
        split = "OAI_test"
    elif "fracatlas" in low:
        split = "FracAtlas_test"
    elif "upmc_internal_test" in low:
        split = "UPMC_internal_test"
    elif "internal_test" in low:
        split = "internal_test"
    else:
        split = "unspecified"
    setting = None
    if "setting" in df.columns and df["setting"].notna().any():
        settings = df["setting"].dropna().astype(str).unique()
        if len(settings) == 1:
            setting = settings[0]
    row_seed = None
    if "seed" in df.columns and df["seed"].notna().any():
        row_seeds = pd.to_numeric(df["seed"], errors="coerce").dropna().unique()
        if len(row_seeds) == 1:
            row_seed = int(row_seeds[0])

    if task == "domain" and setting in {
        "baseline_no_injection", "fewshot_injection"
    }:
        # Keep the matched no-injection control and few-shot adaptation arms
        # separate. Both are evaluated on the same seed-specific target test
        # subset, but only the latter receives FracAtlas training examples.
        regime = f"domain_maintenance_{setting}"
    elif track == "C" or "track_c" in low:
        regime = "zero_shot"
    elif "fewshot" in low or "maintenance" in low:
        regime = "maintenance"
    elif "linear_probe" in low:
        regime = "linear_probe"
    elif "full_ft" in low:
        regime = "full_ft"
    else:
        regime = "baseline"
    raw_model = infer_model(path, df)
    feature_model = unique_text(df, "feature_model")
    domain_model = unique_text(df, "domain_model") if task == "domain" else None
    if task == "domain" and feature_model is None and "__" in raw_model:
        feature_model, suffix = raw_model.rsplit("__", 1)
        if domain_model is None and suffix.lower() in {"logreg", "hgb", "rf"}:
            domain_model = suffix.lower()
    encoder = normalized_encoder(feature_model or raw_model, path)
    inferred_seed = int(seed_match.group(1)) if seed_match else None
    base_seed = seed_from_text(feature_model)
    if base_seed is None:
        base_seed = seed_from_text(raw_model)
    run_seed = row_seed if row_seed is not None else inferred_seed
    if base_seed is None:
        base_seed = run_seed
    is_domain_maintenance = task == "domain" and setting in {
        "baseline_no_injection", "fewshot_injection"
    }
    adapt_seed = run_seed if is_domain_maintenance else np.nan
    return {
        "track": track,
        "task": task,
        "split": split,
        "model": encoder,
        "encoder": encoder,
        "raw_model_id": raw_model,
        "feature_model": feature_model,
        "domain_model": domain_model,
        "regime": regime,
        "seed": run_seed if run_seed is not None else np.nan,
        "base_seed": base_seed if base_seed is not None else np.nan,
        "adaptation_seed": adapt_seed,
        "shots_per_class": int(budget_match.group(1)) if budget_match else np.nan,
        "maintenance_setting": setting,
        "prediction_file": str(path),
    }


def safe_div(num: np.ndarray | float, den: np.ndarray | float):
    return np.divide(num, den, out=np.zeros_like(np.asarray(num, dtype=float)), where=np.asarray(den) != 0)


def metrics_from_cm(cm: np.ndarray, official: list[int], observed: list[int]) -> dict[str, float]:
    """cm rows=true official classes; columns=official classes plus INVALID."""
    n = float(cm.sum())
    tp = np.diag(cm[:, : len(official)])
    support = cm.sum(axis=1)
    predicted = cm[:, : len(official)].sum(axis=0)
    recall = safe_div(tp, support)
    precision = safe_div(tp, predicted)
    f1 = safe_div(2 * precision * recall, precision + recall)
    obs_idx = [official.index(c) for c in observed]
    accuracy = float(tp.sum() / n) if n else np.nan
    out = {
        "accuracy": accuracy,
        "balanced_accuracy_supported": float(np.mean(recall[obs_idx])) if obs_idx else np.nan,
        "macro_f1_official": float(np.mean(f1)),
        "macro_f1_supported": float(np.mean(f1[obs_idx])) if obs_idx else np.nan,
        "weighted_f1_supported": float(np.average(f1[obs_idx], weights=support[obs_idx]))
        if obs_idx and support[obs_idx].sum() else np.nan,
        "invalid_prediction_rate": float(cm[:, -1].sum() / n) if n else np.nan,
    }
    for i, cls in enumerate(official):
        out[f"recall_class_{cls}"] = float(recall[i]) if support[i] else np.nan
        out[f"f1_class_{cls}"] = float(f1[i]) if support[i] or predicted[i] else np.nan
    return out


def build_cm(y: np.ndarray, pred: np.ndarray, official: list[int]) -> np.ndarray:
    pos = {c: i for i, c in enumerate(official)}
    cm = np.zeros((len(official), len(official) + 1), dtype=np.int64)
    for truth, guess in zip(y, pred):
        row = pos[int(truth)]
        col = pos.get(int(guess), len(official)) if np.isfinite(guess) else len(official)
        cm[row, col] += 1
    return cm


def cluster_bootstrap(
    y: np.ndarray, pred: np.ndarray, groups: np.ndarray, official: list[int],
    observed: list[int], n_boot: int, ci: float, seed: int,
) -> dict[str, float]:
    unique, inverse = np.unique(groups.astype(str), return_inverse=True)
    group_cms = np.zeros((len(unique), len(official), len(official) + 1), dtype=np.int64)
    for g in range(len(unique)):
        take = inverse == g
        group_cms[g] = build_cm(y[take], pred[take], official)
    rng = np.random.default_rng(seed)
    keep = ["accuracy", "balanced_accuracy_supported", "macro_f1_official",
            "macro_f1_supported", "weighted_f1_supported", "invalid_prediction_rate"]
    draws = defaultdict(list)
    # Multinomial counts are equivalent to sampling clusters with replacement.
    for _ in range(n_boot):
        counts = rng.multinomial(len(unique), np.full(len(unique), 1.0 / len(unique)))
        cm = np.tensordot(counts, group_cms, axes=(0, 0))
        m = metrics_from_cm(cm, official, observed)
        for key in keep:
            if np.isfinite(m[key]):
                draws[key].append(m[key])
    alpha = (1.0 - ci) / 2.0
    out = {}
    for key in keep:
        vals = np.asarray(draws[key], dtype=float)
        out[f"{key}_ci_low"] = float(np.quantile(vals, alpha)) if len(vals) else np.nan
        out[f"{key}_ci_high"] = float(np.quantile(vals, 1 - alpha)) if len(vals) else np.nan
    return out


def stable_seed(base: int, path: Path) -> int:
    digest = hashlib.sha256(str(path).encode()).digest()
    return (base + int.from_bytes(digest[:4], "little")) % (2**32 - 1)


def analyze_file(track: str, path: Path, args: argparse.Namespace):
    df = pd.read_csv(path, low_memory=False)
    true_col = first_column(df, TRUE_ALIASES)
    pred_col = first_column(df, PRED_ALIASES)
    if true_col is None or pred_col is None:
        raise ValueError("missing true/prediction columns")
    task = infer_task(path, df)
    official = VIEW_CLASSES if task == "view" else DOMAIN_CLASSES
    names = VIEW_NAMES if task == "view" else DOMAIN_NAMES
    meta = infer_metadata(track, path, df, task)

    y_all = numeric_labels(df[true_col]).to_numpy(dtype=float)
    p_all = numeric_labels(df[pred_col]).to_numpy(dtype=float)
    true_valid = np.isfinite(y_all) & np.isin(y_all, official)
    n_invalid_truth = int((~true_valid).sum())
    work = df.loc[true_valid].copy()
    y = y_all[true_valid].astype(int)
    pred = p_all[true_valid]
    pred_valid = np.isfinite(pred) & np.isin(pred, official)
    n_invalid_pred = int((~pred_valid).sum())
    observed = sorted(set(y.tolist()))
    if not len(y):
        raise ValueError("no rows with valid ground truth")

    group_col = first_column(work, GROUP_ALIASES)
    if group_col is None:
        groups = np.arange(len(work)).astype(str)
        grouping = "row_fallback"
    else:
        raw_group = work[group_col].astype("string")
        missing = raw_group.isna() | (raw_group.str.strip() == "")
        raw_group.loc[missing] = [f"missing-row-{i}" for i in np.flatnonzero(missing)]
        groups = raw_group.to_numpy(dtype=str)
        grouping = group_col

    cm = build_cm(y, pred, official)
    result = dict(meta)
    result.update({
        "n_rows_raw": len(df),
        "n_evaluable": len(y),
        "n_groups": len(np.unique(groups)),
        "group_column": grouping,
        "n_invalid_ground_truth": n_invalid_truth,
        "invalid_ground_truth_rate": n_invalid_truth / len(df) if len(df) else np.nan,
        "n_invalid_predictions": n_invalid_pred,
        "observed_classes": "|".join(str(x) for x in observed),
        "observed_class_names": "|".join(names[x] for x in observed),
        "missing_official_classes": "|".join(str(x) for x in official if x not in observed),
    })
    result.update(metrics_from_cm(cm, official, observed))
    result.update(cluster_bootstrap(
        y, pred, groups, official, observed, args.n_boot, args.ci,
        stable_seed(args.bootstrap_seed, path),
    ))

    confusion = []
    pred_names = [names[x] for x in official] + ["INVALID"]
    for i, true_cls in enumerate(official):
        for j, pred_name in enumerate(pred_names):
            confusion.append({**meta, "true_class": true_cls,
                              "true_class_name": names[true_cls],
                              "pred_class": official[j] if j < len(official) else -1,
                              "pred_class_name": pred_name, "count": int(cm[i, j])})
    invalid_rows = []
    if n_invalid_pred:
        uid_col = first_column(work, ["uid", "image_id", "file_id", "path", "png_path"])
        bad_positions = np.flatnonzero(~pred_valid)
        for pos in bad_positions:
            invalid_rows.append({
                **meta,
                "row_in_prediction_file": int(np.flatnonzero(true_valid)[pos]),
                "uid_or_path": str(work.iloc[pos][uid_col]) if uid_col else "",
                "true_label": int(y[pos]),
                "raw_prediction": "" if not np.isfinite(pred[pos]) else pred[pos],
            })
    return result, confusion, invalid_rows


SUMMARY_METRICS = [
    "accuracy", "balanced_accuracy_supported", "macro_f1_official",
    "macro_f1_supported", "weighted_f1_supported", "invalid_prediction_rate",
]


def seed_level_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """Collapse maintenance adaptation repeats within each checkpoint seed."""
    keys = [
        "track", "task", "split", "model", "encoder", "domain_model", "regime",
        "maintenance_setting", "shots_per_class", "base_seed",
    ]
    rows = []
    for values, g in runs.groupby(keys, dropna=False, sort=True):
        row = dict(zip(keys, values))
        is_maintenance = (
            row["task"] == "domain"
            and row["maintenance_setting"] in {
                "baseline_no_injection", "fewshot_injection"
            }
        )
        use = g if is_maintenance else g.iloc[:1]
        row["summary_unit"] = (
            "checkpoint_seed_mean_over_adaptation_seeds"
            if is_maintenance else "checkpoint_seed"
        )
        row["n_component_runs"] = len(g)
        row["n_adaptation_seeds"] = g["adaptation_seed"].nunique(dropna=True)
        for metric in SUMMARY_METRICS:
            row[metric] = pd.to_numeric(use[metric], errors="coerce").mean()
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_runs(seed_runs: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "track", "task", "split", "model", "encoder", "domain_model", "regime",
        "maintenance_setting", "shots_per_class",
    ]
    rows = []
    for values, g in seed_runs.groupby(keys, dropna=False, sort=True):
        row = dict(zip(keys, values))
        row["n_runs"] = len(g)
        row["n_unique_base_seeds"] = g["base_seed"].nunique(dropna=True)
        row["n_prediction_files"] = int(g["n_component_runs"].sum())
        row["summary_unit"] = "checkpoint_seed"
        for metric in SUMMARY_METRICS:
            vals = pd.to_numeric(g[metric], errors="coerce")
            row[f"{metric}_mean"] = vals.mean()
            row[f"{metric}_sd"] = vals.std(ddof=1) if vals.notna().sum() > 1 else np.nan
            row[f"{metric}_min"] = vals.min()
            row[f"{metric}_max"] = vals.max()
        rows.append(row)
    return pd.DataFrame(rows)


def maintenance_deltas(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Paired few-shot minus matched no-injection effects."""
    maintenance = runs[runs["maintenance_setting"].isin(
        ["baseline_no_injection", "fewshot_injection"]
    )].copy()
    pair_keys = [
        "track", "task", "split", "model", "encoder", "domain_model",
        "shots_per_class", "base_seed", "adaptation_seed",
    ]
    control = maintenance[maintenance["maintenance_setting"] == "baseline_no_injection"]
    treated = maintenance[maintenance["maintenance_setting"] == "fewshot_injection"]
    cols = pair_keys + SUMMARY_METRICS
    paired = treated[cols].merge(
        control[cols], on=pair_keys, how="inner", validate="one_to_one",
        suffixes=("_fewshot", "_control"),
    )
    for metric in SUMMARY_METRICS:
        paired[f"{metric}_delta"] = (
            paired[f"{metric}_fewshot"] - paired[f"{metric}_control"]
        )
    if paired.empty:
        return paired, pd.DataFrame()
    delta_cols = [f"{m}_delta" for m in SUMMARY_METRICS]
    base_keys = [
        "track", "task", "split", "model", "encoder", "domain_model",
        "shots_per_class", "base_seed",
    ]
    base = paired.groupby(base_keys, dropna=False)[delta_cols].mean().reset_index()
    summary_keys = base_keys[:-1]
    rows = []
    for values, g in base.groupby(summary_keys, dropna=False, sort=True):
        row = dict(zip(summary_keys, values))
        row["n_base_seeds"] = g["base_seed"].nunique(dropna=True)
        for col in delta_cols:
            vals = pd.to_numeric(g[col], errors="coerce")
            row[f"{col}_mean"] = vals.mean()
            row[f"{col}_sd"] = vals.std(ddof=1) if vals.notna().sum() > 1 else np.nan
        rows.append(row)
    return paired, pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if not (0 < args.ci < 1):
        raise ValueError("--ci must be between 0 and 1")
    roots = parse_inputs(args.input)
    out = Path(args.out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    candidates = []
    seen = set()
    for track, root in roots:
        for path in prediction_files(root):
            key = (track, path.resolve())
            if key not in seen:
                candidates.append((track, path))
                seen.add(key)
    if not candidates:
        raise RuntimeError("No candidate CSV files found")

    results, confusion, invalid_rows, skipped = [], [], [], []
    for i, (track, path) in enumerate(candidates, 1):
        try:
            result, cells, invalid = analyze_file(track, path, args)
            results.append(result)
            confusion.extend(cells)
            invalid_rows.extend(invalid)
            print(f"[{i}/{len(candidates)}] OK {track}: {path}")
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            skipped.append({"track": track, "file": str(path), "reason": message})
            if args.strict:
                raise RuntimeError(f"Failed {path}: {message}") from exc
            warnings.warn(f"Skipping {path}: {message}")

    if not results:
        raise RuntimeError("No valid prediction files were analyzed")
    runs = pd.DataFrame(results).sort_values(
        ["track", "task", "split", "model", "regime", "seed"], na_position="last"
    )
    seed_runs = seed_level_runs(runs)
    aggregate = aggregate_runs(seed_runs)
    paired_deltas, delta_summary = maintenance_deltas(runs)
    conf = pd.DataFrame(confusion)
    audit_cols = [
        "track", "task", "split", "model", "encoder", "domain_model",
        "raw_model_id", "feature_model", "regime", "maintenance_setting",
        "seed", "base_seed", "adaptation_seed", "shots_per_class",
        "prediction_file", "n_rows_raw", "n_evaluable", "n_groups", "group_column",
        "n_invalid_ground_truth", "invalid_ground_truth_rate",
        "n_invalid_predictions", "invalid_prediction_rate", "observed_classes",
        "observed_class_names", "missing_official_classes",
    ]
    runs.to_csv(out / "harmonized_run_metrics.csv", index=False)
    seed_runs.to_csv(out / "harmonized_seed_level_metrics.csv", index=False)
    aggregate.to_csv(out / "harmonized_model_summary.csv", index=False)
    paired_deltas.to_csv(out / "harmonized_maintenance_paired_deltas.csv", index=False)
    delta_summary.to_csv(out / "harmonized_maintenance_delta_summary.csv", index=False)
    conf.to_csv(out / "harmonized_confusion_counts.csv", index=False)
    pd.DataFrame(
        invalid_rows,
        columns=[
            "track", "task", "split", "model", "encoder", "domain_model",
            "regime", "seed", "base_seed", "adaptation_seed",
            "shots_per_class", "prediction_file", "row_in_prediction_file",
            "uid_or_path", "true_label", "raw_prediction",
        ],
    ).to_csv(out / "invalid_prediction_rows.csv", index=False)
    runs[audit_cols].to_csv(out / "harmonized_data_audit.csv", index=False)
    pd.DataFrame(skipped, columns=["track", "file", "reason"]).to_csv(
        out / "skipped_files.csv", index=False
    )
    coverage_keys = [
        "track", "task", "split", "regime", "maintenance_setting", "model"
    ]
    coverage = (
        runs.groupby(coverage_keys, dropna=False)
        .agg(
            n_prediction_files=("prediction_file", "size"),
            n_seeds=("seed", lambda x: x.nunique(dropna=True)),
            n_budgets=("shots_per_class", lambda x: x.nunique(dropna=True)),
            total_evaluable_images=("n_evaluable", "sum"),
            total_invalid_predictions=("n_invalid_predictions", "sum"),
        )
        .reset_index()
        .sort_values(coverage_keys)
    )
    coverage.to_csv(out / "analysis_coverage.csv", index=False)
    config = {
        "view_classes": VIEW_NAMES,
        "domain_classes": DOMAIN_NAMES,
        "invalid_ground_truth_policy": "excluded and reported",
        "invalid_prediction_policy": "retained as error and reported",
        "bootstrap": "cluster resampling with replacement",
        "n_boot": args.n_boot,
        "ci": args.ci,
        "bootstrap_seed": args.bootstrap_seed,
        "inputs": [{"track": t, "path": str(p)} for t, p in roots],
        "n_candidate_files": len(candidates),
        "n_analyzed_files": len(results),
        "n_skipped_files": len(skipped),
        "summary_unit": "independent checkpoint seeds",
        "domain_maintenance_hierarchy": (
            "adaptation seeds averaged within checkpoint seed before model summary"
        ),
    }
    (out / "analysis_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"\nAnalyzed {len(results)} prediction files; skipped {len(skipped)}.")
    print(f"Saved harmonized outputs to: {out}")


if __name__ == "__main__":
    main()

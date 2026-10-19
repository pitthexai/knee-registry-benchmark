from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


CANONICAL_VIEWS = {"AP", "PA", "LATERAL", "SUNRISE"}


def split_oai_calibration_groups(
    rows: pd.DataFrame, group_col: str, seed: int
) -> tuple[set[str], set[str]]:
    """Split the old OAI calibration pool into disjoint adapt/calibration groups.

    Lateral-containing groups are divided first so that both subsets contain
    minority-class examples. Remaining groups are divided to make the number
    of groups as even as possible.
    """
    rng = np.random.default_rng(seed)
    groups = rows[group_col].astype(str)
    all_groups = np.asarray(sorted(groups.unique()), dtype=object)
    lateral_groups = np.asarray(
        sorted(rows.loc[rows["view_label"].eq("LATERAL"), group_col].astype(str).unique()),
        dtype=object,
    )
    rng.shuffle(lateral_groups)
    n_lat_adapt = len(lateral_groups) // 2
    adapt = set(lateral_groups[:n_lat_adapt].tolist())
    calib = set(lateral_groups[n_lat_adapt:].tolist())

    remaining = np.asarray(
        [g for g in all_groups if g not in adapt and g not in calib], dtype=object
    )
    rng.shuffle(remaining)
    target_adapt = len(all_groups) // 2
    need = max(target_adapt - len(adapt), 0)
    adapt.update(remaining[:need].tolist())
    calib.update(remaining[need:].tolist())
    if adapt & calib or adapt | calib != set(all_groups.tolist()):
        raise RuntimeError("Failed to construct exhaustive disjoint OAI groups")
    return adapt, calib


def assert_group_disjoint(df: pd.DataFrame, split_col: str, group_col: str) -> None:
    sub = df[df[split_col].notna() & df[split_col].astype(str).ne("")]
    leaking = sub.groupby(group_col)[split_col].nunique()
    if (leaking > 1).any():
        examples = leaking[leaking > 1].index.astype(str).tolist()[:5]
        raise ValueError(f"Group leakage in {split_col}; examples: {examples}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Prepare the locked manuscript manifest for OrthoRegistryBench v2."
    )
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--group_col", default="group_id")
    ap.add_argument("--path_col", default="png_path")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--skip_path_check", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest, low_memory=False)
    required = {
        "uid", "source", args.path_col, args.group_col, "domain_label",
        "view_label", "has_view_label", "split_view", "split_domain",
        "split_domain_external", "split_view_external_oai",
    }
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")
    if df["uid"].duplicated().any():
        raise SystemExit("uid must be unique")
    if df[args.group_col].isna().any():
        raise SystemExit(f"{args.group_col} contains missing values")
    if not args.skip_path_check:
        check_rows = df[df["png_ok"].eq(1)] if "png_ok" in df else df
        missing_paths = (~check_rows[args.path_col].fillna("").map(lambda x: Path(str(x)).is_file())).sum()
        if missing_paths:
            raise SystemExit(
                f"{missing_paths} image paths are unavailable. Run on the data-vault server "
                "or use --skip_path_check only for manifest preparation."
            )

    # Preserve all original assignments for provenance.
    for col in ["split_view", "split_domain", "split_domain_external", "split_view_external_oai"]:
        df[f"original_{col}"] = df[col]

    # Standardize only the named locked OAI test. Never resample test groups.
    df["split_view_external_oai"] = df["split_view_external_oai"].replace(
        {"OAI_test2": "OAI_test"}
    )
    old_calib = df[
        df["source"].astype(str).str.upper().eq("OAI")
        & df["split_view_external_oai"].eq("OAI_calib")
        & df["has_view_label"].eq(1)
        & df["view_label"].isin(CANONICAL_VIEWS)
    ].copy()
    if old_calib.empty:
        raise SystemExit("No eligible OAI_calib rows were found")
    adapt_groups, calib_groups = split_oai_calibration_groups(
        old_calib, args.group_col, args.seed
    )
    old_mask = df["split_view_external_oai"].eq("OAI_calib")
    adapt_mask = old_mask & df[args.group_col].astype(str).isin(adapt_groups)
    calib_mask = old_mask & df[args.group_col].astype(str).isin(calib_groups)
    df.loc[adapt_mask, "split_view_external_oai"] = "OAI_adapt"
    df.loc[calib_mask, "split_view_external_oai"] = "OAI_calib"

    # Explicit benchmark eligibility avoids silent inclusion of Unknown/OtherView.
    df["benchmark_view_eligible"] = (
        df["has_view_label"].eq(1) & df["view_label"].isin(CANONICAL_VIEWS)
    ).astype(int)
    df["benchmark_manifest_version"] = "2.1"
    df["benchmark_split_seed"] = args.seed

    for col in ["split_view", "split_domain", "split_view_external_oai"]:
        assert_group_disjoint(df, col, args.group_col)

    test_original = set(
        df.loc[df["original_split_view_external_oai"].eq("OAI_test2"), "uid"]
    )
    test_derived = set(df.loc[df["split_view_external_oai"].eq("OAI_test"), "uid"])
    if test_original != test_derived:
        raise RuntimeError("Locked OAI test membership changed")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {out} with {len(df):,} rows")
    print("OAI external split summary")
    summary = (
        df[df["split_view_external_oai"].notna()]
        .groupby(["split_view_external_oai", "view_label"])
        .agg(images=("uid", "size"), groups=(args.group_col, "nunique"))
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()

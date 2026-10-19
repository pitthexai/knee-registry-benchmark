from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold


def norm(x):
    return str(x).strip().upper()


def file_ok(p):
    s = str(p).strip()
    if s == "" or s.lower() == "nan":
        return 0
    return int(Path(s).exists())


def group_split(df, group_col, test_size, seed):
    if len(df) == 0:
        raise ValueError("No rows to split. Check filters.")
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    keep_idx, test_idx = next(gss.split(df, groups=df[group_col].astype(str)))
    return df.iloc[keep_idx].copy(), df.iloc[test_idx].copy()


def assign(base, subset, col, value):
    base.loc[subset.index, col] = value


def assert_disjoint_groups(df, split_col, group_col):
    sub = df[df[split_col].astype(str) != ""]
    counts = sub.groupby(group_col)[split_col].nunique()
    if (counts > 1).any():
        raise RuntimeError(f"Group leakage detected in {split_col}")


def main():
    ap = argparse.ArgumentParser(description="Create fixed group-level splits for OrthoRegistryBench.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--path_col", default="png_path")
    ap.add_argument("--group_col", default="group_id")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--upmc_source", default="UPMC")
    ap.add_argument("--oai_source", default="OAI")
    ap.add_argument("--lera_source", default="LERA")
    ap.add_argument("--frac_source", default="FRACATLAS")
    ap.add_argument("--knee_label", default="XR_KNEE")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest, low_memory=False)
    if args.path_col not in df.columns:
        raise SystemExit(f"Missing path column: {args.path_col}")
    if args.group_col not in df.columns:
        raise SystemExit(f"Missing group column: {args.group_col}")

    df["source_norm"] = df["source"].map(norm)
    df["label_norm"] = df["label"].map(norm) if "label" in df.columns else ""
    df["image_ok"] = df[args.path_col].apply(file_ok)

    for col in ["split_view", "split_view_external", "split_view_external_oai", "split_domain", "split_domain_external"]:
        df[col] = ""

    has_view = pd.to_numeric(df.get("has_view_label", 0), errors="coerce").fillna(0).astype(int)

    # View: train/internal on UPMC knee only
    view_upmc = df[(df["source_norm"] == norm(args.upmc_source)) &
                   (df["label_norm"] == norm(args.knee_label)) &
                   (has_view == 1) & (df["image_ok"] == 1)].copy()
    print("UPMC view eligible:", len(view_upmc))
    view_trainval, view_test = group_split(view_upmc, args.group_col, 0.10, args.seed)
    view_train, view_temp = group_split(view_trainval, args.group_col, 0.2222, args.seed)
    view_val, view_calib = group_split(view_temp, args.group_col, 0.50, args.seed)
    assign(df, view_train, "split_view", "train")
    assign(df, view_val, "split_view", "val")
    assign(df, view_calib, "split_view", "calib")
    assign(df, view_test, "split_view", "test")

    # OAI external split into mutually exclusive adaptation/calibration/test sets.
    # Adaptation images update weights; calibration images fit temperature/conformal
    # parameters; test images remain untouched until final evaluation.
    view_oai = df[(df["source_norm"] == norm(args.oai_source)) &
                  (df["label_norm"] == norm(args.knee_label)) &
                  (has_view == 1) & (df["image_ok"] == 1)].copy()
    print("OAI view external eligible:", len(view_oai))
    if len(view_oai) > 0:
        oai_test, oai_dev = group_split(view_oai, args.group_col, 0.20, args.seed)
        oai_adapt, oai_calib = group_split(oai_dev, args.group_col, 0.50, args.seed + 1)
        assign(df, oai_adapt, "split_view_external_oai", "OAI_adapt")
        assign(df, oai_calib, "split_view_external_oai", "OAI_calib")
        assign(df, oai_test, "split_view_external_oai", "OAI_test")
        assign(df, view_oai, "split_view_external", "OAI_all")

    # Domain: seen sources UPMC + OAI + LERA
    if "domain_label" not in df.columns:
        raise SystemExit("Missing domain_label column.")
    df["domain_label"] = pd.to_numeric(df["domain_label"], errors="coerce")
    seen_sources = {norm(args.upmc_source), norm(args.oai_source), norm(args.lera_source)}
    domain_df = df[(df["source_norm"].isin(seen_sources)) &
                   (df["domain_label"].isin([0, 1])) & (df["image_ok"] == 1)].copy()
    print("Domain seen-source eligible:", len(domain_df))
    dom_trainval, dom_test = group_split(domain_df, args.group_col, 0.10, args.seed)
    dom_train, dom_temp = group_split(dom_trainval, args.group_col, 0.2222, args.seed)
    dom_val, dom_calib = group_split(dom_temp, args.group_col, 0.50, args.seed)
    assign(df, dom_train, "split_domain", "train")
    assign(df, dom_val, "split_domain", "val")
    assign(df, dom_calib, "split_domain", "calib")
    assign(df, dom_test, "split_domain", "test")

    frac = df[(df["source_norm"] == norm(args.frac_source)) & (df["domain_label"].isin([0, 1])) & (df["image_ok"] == 1)].copy()
    print("FracAtlas external domain eligible:", len(frac))
    assign(df, frac, "split_domain_external", "FracAtlas_test")

    for col in ["split_view", "split_view_external_oai", "split_domain"]:
        assert_disjoint_groups(df, col, args.group_col)

    df.drop(columns=["source_norm", "label_norm"], errors="ignore").to_csv(args.out, index=False)
    print("Wrote", args.out)
    for c in ["split_view", "split_view_external_oai", "split_domain", "split_domain_external"]:
        print("\n", c)
        print(df[c].value_counts(dropna=False))


if __name__ == "__main__":
    main()

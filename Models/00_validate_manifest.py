from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

REQUIRED = {"uid", "source", "domain_label", "has_view_label", "view_label", "group_id"}


def main():
    ap = argparse.ArgumentParser(description="Fail fast on benchmark manifest errors.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--path_col", default="png_path")
    ap.add_argument("--group_col", default="group_id")
    args = ap.parse_args()
    df = pd.read_csv(args.manifest, low_memory=False)
    missing = (REQUIRED | {args.path_col, args.group_col}) - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns: {sorted(missing)}")
    if df[args.group_col].isna().any():
        raise SystemExit("Missing group identifiers are not allowed")
    if df["uid"].duplicated().any():
        raise SystemExit("Duplicate uid values found")
    check_rows = df[df["png_ok"].eq(1)].copy() if "png_ok" in df else df.copy()
    valid_paths = check_rows[args.path_col].dropna().astype(str).str.strip()
    valid_paths = valid_paths[valid_paths.ne("")]
    if valid_paths.duplicated().any():
        raise SystemExit("Duplicate image paths found; resolve duplicate records first")
    missing_files = [p for p in valid_paths if not Path(p).is_file()]
    if missing_files:
        raise SystemExit(f"Missing/unreadable image files: {len(missing_files)}")
    for split_col in ["split_view", "split_domain", "split_view_external_oai"]:
        if split_col not in df:
            continue
        sub = df[df[split_col].notna() & df[split_col].astype(str).ne("")]
        n_by_group = sub.groupby(args.group_col)[split_col].nunique()
        if (n_by_group > 1).any():
            raise SystemExit(f"Group leakage detected in {split_col}")
    print(f"VALID rows={len(df)} groups={df[args.group_col].nunique()} sources={df.source.nunique()}")


if __name__ == "__main__":
    main()

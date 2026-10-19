#!/usr/bin/env python3
"""Audit Track C failed images, build a repair manifest, and merge rerun rows.

Use `audit` first. Rerun the Track C zero-shot evaluator using the generated
repair manifest and a separate repair output directory. Then use `merge`; it
requires every replacement prediction to be valid and preserves backups.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("audit")
    a.add_argument("--manifest", required=True)
    a.add_argument("--invalid_rows", required=True)
    a.add_argument("--out_dir", required=True)
    a.add_argument("--path_col", default="png_path")
    m = sub.add_parser("merge")
    m.add_argument("--original_pred_dir", required=True)
    m.add_argument("--repair_pred_dir", required=True)
    m.add_argument("--out_dir", required=True,
                   help="Write corrected complete prediction CSVs here; originals remain untouched")
    return p.parse_args()


def image_audit(path):
    p = Path(str(path))
    row = {"path_exists": p.is_file(), "size_bytes": p.stat().st_size if p.is_file() else 0,
           "pil_verify_ok": False, "pil_load_ok": False, "width": np.nan, "height": np.nan,
           "mode": "", "sha256": "", "error": ""}
    if not p.is_file():
        row["error"] = "missing"; return row
    try:
        h = hashlib.sha256()
        with p.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(chunk)
        row["sha256"] = h.hexdigest()
        with Image.open(p) as im:
            im.verify()
        row["pil_verify_ok"] = True
        with Image.open(p) as im:
            im.load()
            row.update({"pil_load_ok": True, "width": im.width, "height": im.height, "mode": im.mode})
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def audit(args):
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest, low_memory=False)
    bad = pd.read_csv(args.invalid_rows)
    uids = bad["uid_or_path"].dropna().astype(str).drop_duplicates()
    selected = manifest[manifest["uid"].astype(str).isin(set(uids))].copy()
    if len(selected) != len(uids):
        raise RuntimeError(f"Matched {len(selected)} of {len(uids)} failed UIDs")
    audits = pd.DataFrame([image_audit(p) for p in selected[args.path_col]])
    report = pd.concat([selected.reset_index(drop=True), audits], axis=1)
    report.to_csv(out / "track_c_failed_image_decode_audit.csv", index=False)
    readable = report[report["pil_load_ok"]].copy()
    readable[manifest.columns].to_csv(out / "track_c_repair_manifest.csv", index=False)
    unreadable = report[~report["pil_load_ok"]].copy()
    unreadable.to_csv(out / "track_c_unreadable_images.csv", index=False)
    print(f"Matched={len(report)} readable={len(readable)} unreadable={len(unreadable)}")
    print("Repair manifest:", out / "track_c_repair_manifest.csv")
    if len(unreadable):
        print("Do not rerun unreadable files; restore/regenerate them first:", out / "track_c_unreadable_images.csv")


def valid_prediction_mask(series):
    p = pd.to_numeric(series, errors="coerce")
    return p.isin([0, 1])


def merge(args):
    original = Path(args.original_pred_dir)
    repair = Path(args.repair_pred_dir)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    repairs = sorted(repair.glob("pred_track_c_*_domain_*FracAtlas*.csv"))
    if not repairs:
        raise SystemExit("No Track C FracAtlas domain repair prediction files found")
    for repaired_path in repairs:
        candidates = list(original.glob(repaired_path.name))
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one original named {repaired_path.name}; found {len(candidates)}")
        original_path = candidates[0]
        old = pd.read_csv(original_path, low_memory=False)
        new = pd.read_csv(repaired_path, low_memory=False)
        if "uid" not in old or "uid" not in new or "pred" not in new:
            raise ValueError(f"uid/pred missing: {repaired_path}")
        if new["uid"].duplicated().any():
            raise ValueError(f"Duplicate repair UIDs: {repaired_path}")
        if not valid_prediction_mask(new["pred"]).all():
            raise ValueError(f"Repair still contains invalid predictions: {repaired_path}")
        old_idx = old.set_index(old["uid"].astype(str), drop=False)
        new_idx = new.set_index(new["uid"].astype(str), drop=False)
        missing = new_idx.index.difference(old_idx.index)
        if len(missing):
            raise ValueError(f"Repair UIDs absent from original: {missing[:5].tolist()}")
        common_cols = [c for c in new.columns if c in old.columns and c != "uid"]
        old_idx.loc[new_idx.index, common_cols] = new_idx[common_cols]
        merged = old_idx.reset_index(drop=True)
        if len(merged) != len(old):
            raise AssertionError("Row count changed during merge")
        destination = out / original_path.name
        merged.to_csv(destination, index=False)
        print("MERGED:", destination, "replaced rows=", len(new))


def main():
    args = parse_args()
    audit(args) if args.command == "audit" else merge(args)


if __name__ == "__main__":
    main()

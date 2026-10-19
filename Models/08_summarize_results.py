from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def maybe_read(p):
    return pd.read_csv(p) if Path(p).exists() else None


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--out", required=True)
    args=ap.parse_args()
    root=Path(args.results_dir)
    blocks=[]
    for p in sorted(root.rglob("*.csv")):
        try:
            df=pd.read_csv(p)
        except Exception:
            continue
        df.insert(0,"source_file",str(p.relative_to(root)))
        blocks.append(df)
    if not blocks:
        raise SystemExit("No CSV files found")
    out=pd.concat(blocks, ignore_index=True, sort=False)
    out.to_csv(args.out, index=False)
    print("Wrote", args.out, "rows", len(out))

if __name__=="__main__":
    main()

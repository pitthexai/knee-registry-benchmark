from __future__ import annotations

import argparse
import json

import torch

from orthobench.track_b_models import TRACK_B_SPECS, create_track_b_model, track_b_normalization


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(TRACK_B_SPECS))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip_forward", action="store_true")
    args = parser.parse_args()
    results = []
    for key in args.models:
        row = {"model_key": key, "source": TRACK_B_SPECS[key].source, "status": "failed"}
        try:
            model = create_track_b_model(key, 4, pretrained=True, img_size=224).to(args.device).eval()
            mean, std = track_b_normalization(key, model)
            row.update(feature_dim=model.feature_dim, mean=mean, std=std)
            if not args.skip_forward:
                with torch.inference_mode():
                    output = model(torch.zeros(1, 3, 224, 224, device=args.device))
                if tuple(output.shape) != (1, 4):
                    raise RuntimeError(f"unexpected output shape {tuple(output.shape)}")
            row["status"] = "ok"
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        print(json.dumps(row), flush=True)
        results.append(row)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    failed = [row for row in results if row["status"] != "ok"]
    if failed:
        raise SystemExit(f"Preflight failed for {[row['model_key'] for row in failed]}")


if __name__ == "__main__":
    main()

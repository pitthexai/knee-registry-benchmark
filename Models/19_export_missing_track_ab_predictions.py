#!/usr/bin/env python3
"""Export missing Track A/B per-image predictions without retraining.

Subcommands
-----------
view
    Evaluate saved view checkpoints on a manifest split. Intended for Track B
    internal-test and view-maintenance checkpoints.
domain-baseline
    Apply saved domain classifiers to saved embedding NPZ files.
domain-maintenance
    Recreate the original disjoint FracAtlas test indices and apply the saved
    maintenance classifiers to those rows.

Run from the OrthoRegistryBench repository root with PYTHONPATH=. if needed.
Existing prediction files are never overwritten unless --overwrite is passed.
"""

from __future__ import annotations

import argparse
import inspect
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


VIEW_NAMES = ["AP", "PA", "LATERAL", "SUNRISE"]


def add_common_output(p):
    p.add_argument("--out_dir", required=True)
    p.add_argument("--overwrite", action="store_true")


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    v = sub.add_parser("view", help="Export predictions from saved view checkpoints")
    v.add_argument("--manifest", required=True)
    v.add_argument("--ckpt_dir", action="append", required=True,
                   help="Checkpoint directory; repeat for baseline and maintenance roots")
    v.add_argument("--split_col", default="split_view")
    v.add_argument("--split_value", default="test")
    v.add_argument("--split_tag", default="UPMC_internal_test")
    v.add_argument("--path_col", default="png_path")
    v.add_argument("--batch", type=int, default=64)
    v.add_argument("--num_workers", type=int, default=4)
    v.add_argument("--device", default="cuda")
    add_common_output(v)

    b = sub.add_parser("domain-baseline", help="Export saved baseline domain predictions")
    b.add_argument("--manifest", required=True)
    b.add_argument("--emb_dir", required=True)
    b.add_argument("--model_dir", required=True)
    b.add_argument("--splits", nargs="+", default=["test", "fracatlas"])
    add_common_output(b)

    m = sub.add_parser("domain-maintenance", help="Export saved maintenance predictions")
    m.add_argument("--manifest", required=True)
    m.add_argument("--emb_dir", required=True)
    m.add_argument("--model_dir", required=True)
    m.add_argument("--inj_pos", type=int, default=200)
    m.add_argument("--inj_neg", type=int, default=200)
    m.add_argument("--tune_pos", type=int, default=200)
    m.add_argument("--tune_neg", type=int, default=200)
    add_common_output(m)
    return p.parse_args()


def save_csv(df, path: Path, overwrite: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        print("EXISTS (not overwritten):", path)
        return False
    df.to_csv(path, index=False)
    print("SAVED:", path, "rows=", len(df))
    return True


def metadata_for_uids(manifest, uids):
    cols = [c for c in ["uid", "group_id", "source", "domain_label", "view_label", "side_label"]
            if c in manifest.columns]
    meta = manifest[cols].drop_duplicates("uid")
    base = pd.DataFrame({"uid": np.asarray(uids).astype(str)})
    return base.merge(meta, on="uid", how="left", validate="one_to_one")


def instantiate_view_model(architecture, ckpt):
    """Use the project's own factory so Track B special encoders stay identical."""
    from orthobench import models as model_module
    try:
        from orthobench import track_b_models as track_b_module
    except ImportError:
        track_b_module = None

    # Track B must be attempted first. In particular, BiomedCLIP is an
    # OpenCLIP/timm visual tower wrapped by the Track B classifier class and
    # cannot be reconstructed by orthobench.models.create_timm_model.
    candidates = []
    if track_b_module is not None:
        candidates.append((track_b_module, "create_track_b_model"))
    candidates.extend([
        (model_module, "create_timm_model"),
        (model_module, "create_model"),
    ])
    errors = []
    for module, name in candidates:
        factory = getattr(module, name, None)
        if factory is None:
            continue
        try:
            sig = inspect.signature(factory)
            kwargs = {}
            if "num_classes" in sig.parameters:
                kwargs["num_classes"] = 4
            if "pretrained" in sig.parameters:
                # Match orthobench.track_b_models.load_track_b_checkpoint:
                # BiomedCLIP's HF-Hub definition is a timm-backed visual trunk,
                # architecturally different from generic OpenCLIP ViT-B/16.
                # The other Track B encoders can be built without downloading
                # pretrained weights because the full saved state follows.
                kwargs["pretrained"] = (
                    module is track_b_module
                    and architecture == "biomedclip_vitb16"
                )
            if "img_size" in sig.parameters:
                kwargs["img_size"] = int(ckpt.get("img_size", 224))
            model = factory(architecture, **kwargs)
            # Permit factories that return (model, metadata/spec) while keeping
            # the normal nn.Module return unchanged.
            if isinstance(model, (tuple, list)):
                model = model[0]
            return model
        except Exception as exc:
            errors.append(f"{module.__name__}.{name}: {type(exc).__name__}: {exc}")
    raise RuntimeError("No project model factory could rebuild checkpoint " + architecture + "; " + " | ".join(errors))


def normalize_state_dict(state):
    if state and all(str(k).startswith("module.") for k in state):
        return {str(k)[7:]: v for k, v in state.items()}
    return state


def export_view(args):
    import torch
    from torch.utils.data import DataLoader
    from orthobench.datasets import ViewDataset

    manifest = pd.read_csv(args.manifest, low_memory=False)
    sub = manifest[manifest[args.split_col].astype(str) == args.split_value].copy()
    if "has_view_label" in sub:
        sub = sub[pd.to_numeric(sub["has_view_label"], errors="coerce") == 1]
    if sub.empty:
        raise SystemExit("No view rows selected")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    paths = []
    for root in args.ckpt_dir:
        paths.extend(Path(root).rglob("*.pt"))
    paths = sorted(set(p.resolve() for p in paths))
    if not paths:
        raise SystemExit("No .pt checkpoints found")

    for ckpt_path in paths:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model_id = str(ckpt.get("model_id", ckpt_path.stem))
        architecture = str(ckpt.get("model_key", ckpt.get("architecture", ckpt.get("model_name", model_id))))
        model = instantiate_view_model(architecture, ckpt)
        state = ckpt.get("model", ckpt.get("state_dict", ckpt.get("model_state_dict")))
        if state is None:
            print("SKIP no state dict:", ckpt_path)
            continue
        model.load_state_dict(normalize_state_dict(state), strict=True)
        model.to(device).eval()
        ds = ViewDataset(
            sub, train=False, img_size=int(ckpt.get("img_size", 224)),
            path_col=args.path_col, preprocessing=ckpt.get("preprocessing"),
        )
        loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.num_workers, pin_memory=device.type == "cuda")
        ys, preds, probs, uids = [], [], [], []
        with torch.inference_mode():
            for x, y, uid in loader:
                logits = model(x.to(device))
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                if isinstance(logits, dict):
                    logits = logits.get("logits")
                probability = torch.softmax(logits, dim=1).cpu().numpy()
                ys.append(y.numpy()); preds.append(probability.argmax(1)); probs.append(probability)
                uids.extend(map(str, uid))
        y = np.concatenate(ys); pred = np.concatenate(preds); probability = np.vstack(probs)
        out = metadata_for_uids(manifest, uids)
        out.insert(1, "y", y); out.insert(2, "pred", pred)
        for j, name in enumerate(VIEW_NAMES):
            out[f"prob_{name}"] = probability[:, j]
        out["model"] = model_id
        out["architecture"] = architecture
        filename = f"pred_{model_id}_{args.split_tag}.csv"
        save_csv(out, Path(args.out_dir) / filename, args.overwrite)


def load_embedding(path):
    d = np.load(path, allow_pickle=True)
    if "uid" not in d.files:
        raise ValueError(f"Embedding lacks uid array: {path}")
    return d["X"], d["y"].astype(int), d["uid"].astype(str)


def positive_probability(clf, X):
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X)
        classes = list(getattr(clf, "classes_", [0, 1]))
        return proba[:, classes.index(1)]
    score = clf.decision_function(X)
    return 1.0 / (1.0 + np.exp(-score))


def parse_baseline_model(path):
    match = re.match(r"domain_(.+)_(logreg|hgb|rf)$", path.stem)
    if not match:
        return None
    return match.group(1), match.group(2)


def export_domain_baseline(args):
    manifest = pd.read_csv(args.manifest, low_memory=False)
    split_tag = {"test": "internal_test", "fracatlas": "FracAtlas_test"}
    model_paths = sorted(Path(args.model_dir).rglob("domain_*.joblib"))
    for model_path in model_paths:
        parsed = parse_baseline_model(model_path)
        if parsed is None:
            continue
        feature, domain_model = parsed
        clf = joblib.load(model_path)
        for split in args.splits:
            emb = Path(args.emb_dir) / f"emb_{feature}_{split}.npz"
            if not emb.exists():
                print("SKIP missing embedding:", emb); continue
            X, y, uid = load_embedding(emb)
            pred = np.asarray(clf.predict(X)).astype(int)
            prob = positive_probability(clf, X)
            out = metadata_for_uids(manifest, uid)
            out.insert(1, "y", y); out.insert(2, "pred", pred); out.insert(3, "prob_knee", prob)
            out["model"] = f"{feature}__{domain_model}"
            out["feature_model"] = feature; out["domain_model"] = domain_model
            name = f"pred_domain_{feature}_{domain_model}_baseline_{split_tag.get(split, split)}.csv"
            save_csv(out, Path(args.out_dir) / name, args.overwrite)


def sample_disjoint(y, npi, nni, npt, nnt, seed):
    rng = np.random.default_rng(seed)
    pos = rng.permutation(np.where(y == 1)[0]); neg = rng.permutation(np.where(y == 0)[0])
    if len(pos) < npi + npt or len(neg) < nni + nnt:
        raise ValueError("Insufficient FracAtlas class counts for original injection/tuning design")
    inj = np.r_[pos[:npi], neg[:nni]]
    tune = np.r_[pos[npi:npi+npt], neg[nni:nni+nnt]]
    used = np.zeros(len(y), dtype=bool); used[inj] = True; used[tune] = True
    return np.flatnonzero(~used)


def parse_maintenance_model(path):
    match = re.match(
        r"domain_(.+)_(logreg|hgb|rf)_(baseline_no_injection|fewshot_injection)_seed(\d+)$",
        path.stem,
    )
    if not match:
        return None
    return match.group(1), match.group(2), match.group(3), int(match.group(4))


def export_domain_maintenance(args):
    manifest = pd.read_csv(args.manifest, low_memory=False)
    cache = {}
    for model_path in sorted(Path(args.model_dir).rglob("domain_*.joblib")):
        parsed = parse_maintenance_model(model_path)
        if parsed is None:
            continue
        feature, domain_model, setting, seed = parsed
        if feature not in cache:
            cache[feature] = load_embedding(Path(args.emb_dir) / f"emb_{feature}_fracatlas.npz")
        X, y, uid = cache[feature]
        test = sample_disjoint(y, args.inj_pos, args.inj_neg, args.tune_pos, args.tune_neg, seed)
        clf = joblib.load(model_path)
        pred = np.asarray(clf.predict(X[test])).astype(int)
        prob = positive_probability(clf, X[test])
        out = metadata_for_uids(manifest, uid[test])
        out.insert(1, "y", y[test]); out.insert(2, "pred", pred); out.insert(3, "prob_knee", prob)
        out["model"] = f"{feature}__{domain_model}"
        out["feature_model"] = feature; out["domain_model"] = domain_model
        out["setting"] = setting; out["seed"] = seed
        name = f"pred_domain_{feature}_{domain_model}_{setting}_seed{seed}_maintenance_FracAtlas_test.csv"
        save_csv(out, Path(args.out_dir) / name, args.overwrite)


def main():
    args = parse_args()
    if args.command == "view":
        export_view(args)
    elif args.command == "domain-baseline":
        export_domain_baseline(args)
    else:
        export_domain_maintenance(args)


if __name__ == "__main__":
    main()

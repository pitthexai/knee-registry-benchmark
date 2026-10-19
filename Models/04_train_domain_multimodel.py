from __future__ import annotations

import argparse, joblib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from orthobench.metrics import binary_metrics


def load_npz(p):
    d = np.load(p, allow_pickle=True)
    return d["X"], d["y"].astype(int)


def make_clf(name):
    if name == "logreg":
        return LogisticRegression(max_iter=3000, class_weight="balanced", n_jobs=-1)
    if name == "hgb":
        return HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05, max_iter=400)
    if name == "rf":
        return RandomForestClassifier(n_estimators=500, class_weight="balanced", n_jobs=-1, random_state=42)
    raise ValueError(name)


def prob(clf, X):
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    z = clf.decision_function(X)
    return 1 / (1 + np.exp(-z))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--feature_models", nargs="+", default=None)
    ap.add_argument("--domain_models", nargs="+", default=["logreg", "hgb", "rf"])
    ap.add_argument("--calibrate", action="store_true")
    args = ap.parse_args()
    out_dir = Path(args.out_dir); (out_dir / "domain_models").mkdir(parents=True, exist_ok=True); (out_dir / "results").mkdir(parents=True, exist_ok=True)
    feature_models = args.feature_models or sorted(
        p.name[len("emb_"):-len("_train.npz")] for p in Path(args.emb_dir).glob("emb_*_train.npz")
    )
    if not feature_models:
        raise SystemExit("No training embedding files found")
    rows = []
    for feat in feature_models:
        Xtr, ytr = load_npz(Path(args.emb_dir) / f"emb_{feat}_train.npz")
        Xva, yva = load_npz(Path(args.emb_dir) / f"emb_{feat}_val.npz")
        Xte, yte = load_npz(Path(args.emb_dir) / f"emb_{feat}_test.npz")
        Xfa, yfa = load_npz(Path(args.emb_dir) / f"emb_{feat}_fracatlas.npz")
        for dm in args.domain_models:
            clf = make_clf(dm)
            clf.fit(Xtr, ytr)
            if args.calibrate:
                clf = CalibratedClassifierCV(clf, method="sigmoid", cv="prefit")
                clf.fit(Xva, yva)
            model_path = out_dir / "domain_models" / f"domain_{feat}_{dm}.joblib"
            joblib.dump(clf, model_path)
            for split, X, y in [("val", Xva, yva), ("internal_test", Xte, yte), ("fracatlas_external", Xfa, yfa)]:
                pr = clf.predict(X)
                pp = prob(clf, X)
                row = {"feature_model": feat, "domain_model": dm, "split": split, "model_path": str(model_path)}
                row.update(binary_metrics(y, pr, pp))
                rows.append(row)
                print(row)
    pd.DataFrame(rows).to_csv(out_dir / "results" / "domain_leaderboard.csv", index=False)


if __name__ == "__main__":
    main()

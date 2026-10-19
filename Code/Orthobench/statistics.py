from __future__ import annotations

from typing import Callable, Dict, Iterable, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def _metric(name: str, y: np.ndarray, pred: np.ndarray, prob=None) -> float:
    if name == "accuracy":
        return float(accuracy_score(y, pred))
    if name == "macro_f1":
        return float(f1_score(y, pred, average="macro", zero_division=0))
    if name == "weighted_f1":
        return float(f1_score(y, pred, average="weighted", zero_division=0))
    if name == "auroc":
        if prob is None:
            raise ValueError("auroc requires probabilities")
        return float(roc_auc_score(y, prob))
    raise ValueError(f"Unknown metric: {name}")


def grouped_bootstrap_ci(y, pred, groups, metric="accuracy", prob=None,
                         n_boot=2000, seed=2026, confidence=0.95) -> Dict[str, float]:
    """Cluster bootstrap CI that keeps all images from a sampled patient/study."""
    y, pred, groups = map(np.asarray, (y, pred, groups))
    prob = None if prob is None else np.asarray(prob)
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == g) for g in sampled])
        vals.append(_metric(metric, y[idx], pred[idx], None if prob is None else prob[idx]))
    tail = (1 - confidence) / 2
    return {"estimate": _metric(metric, y, pred, prob),
            "ci_low": float(np.quantile(vals, tail)),
            "ci_high": float(np.quantile(vals, 1 - tail)),
            "n_groups": int(len(unique)), "n_boot": int(n_boot)}


def paired_grouped_bootstrap_delta(y, pred_a, pred_b, groups, metric="accuracy",
                                   prob_a=None, prob_b=None, n_boot=2000, seed=2026):
    """Paired cluster bootstrap for model A minus model B on identical cases."""
    y, pred_a, pred_b, groups = map(np.asarray, (y, pred_a, pred_b, groups))
    prob_a = None if prob_a is None else np.asarray(prob_a)
    prob_b = None if prob_b is None else np.asarray(prob_b)
    unique = np.unique(groups); rng = np.random.default_rng(seed); delta = []
    for _ in range(n_boot):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == g) for g in sampled])
        a = _metric(metric, y[idx], pred_a[idx], None if prob_a is None else prob_a[idx])
        b = _metric(metric, y[idx], pred_b[idx], None if prob_b is None else prob_b[idx])
        delta.append(a - b)
    point = _metric(metric, y, pred_a, prob_a) - _metric(metric, y, pred_b, prob_b)
    lo, hi = np.quantile(delta, [0.025, 0.975])
    return {"delta": float(point), "ci_low": float(lo), "ci_high": float(hi),
            "p_superiority": float(np.mean(np.asarray(delta) > 0))}

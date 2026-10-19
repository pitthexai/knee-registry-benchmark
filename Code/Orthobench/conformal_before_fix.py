from __future__ import annotations

import numpy as np


def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def aps_scores(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    order = np.argsort(-probs, axis=1)
    sorted_probs = np.take_along_axis(probs, order, axis=1)
    cum = np.cumsum(sorted_probs, axis=1)
    inv = np.argsort(order, axis=1)
    true_rank = inv[np.arange(len(y)), y]
    return cum[np.arange(len(y)), true_rank]


def qhat(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample split-conformal quantile.

    Uses k=ceil((n+1)(1-alpha)), clipped at n. This is deliberately explicit
    because a plain empirical (1-alpha) quantile can undercover in small sets.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or len(scores) == 0:
        raise ValueError("scores must be a non-empty one-dimensional array")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between 0 and 1")
    k = min(int(np.ceil((len(scores) + 1) * (1 - alpha))), len(scores))
    return float(np.partition(scores, k - 1)[k - 1])


def aps_set(probs_row: np.ndarray, q: float):
    order = np.argsort(-probs_row)
    cumulative = np.cumsum(probs_row[order])
    # Calibration scores include the candidate label probability. Prediction
    # sets must therefore use the identical score definition: include a label
    # iff its cumulative score is <= q. The old implementation always added
    # the first label crossing q, which was inconsistent and over-covered.
    return [int(label) for label, score in zip(order, cumulative) if score <= q]


def evaluate_sets(probs: np.ndarray, y: np.ndarray, q: float):
    sizes, cover = [], []
    for i in range(len(y)):
        s = aps_set(probs[i], q)
        sizes.append(len(s))
        cover.append(int(int(y[i]) in s))
    sizes = np.asarray(sizes)
    return {
        "coverage": float(np.mean(cover)),
        "singleton_rate": float(np.mean(sizes == 1)),
        "avg_set_size": float(np.mean(sizes)),
    }


def mondrian_qhats(probs: np.ndarray, y: np.ndarray, n_classes: int, alpha: float):
    out = {}
    for c in range(n_classes):
        mask = y == c
        if mask.sum() == 0:
            continue
        out[c] = qhat(aps_scores(probs[mask], y[mask]), alpha)
    return out


def class_conditional_set(probs_row: np.ndarray, q_by_class: dict, fallback_q=None):
    """APS set with a threshold attached to each candidate label.

    Thresholding by the predicted class (as the v1 code did) is not Mondrian
    conformal prediction and does not yield true-class conditional coverage.
    """
    order = np.argsort(-probs_row)
    cumulative = np.cumsum(probs_row[order])
    scores = np.empty_like(cumulative)
    scores[order] = cumulative
    if fallback_q is None:
        fallback_q = max(q_by_class.values())
    return [int(c) for c in range(len(probs_row))
            if scores[c] <= q_by_class.get(c, fallback_q)]


def evaluate_class_conditional(probs: np.ndarray, y: np.ndarray, q_by_class: dict):
    sizes, cover = [], []
    for i in range(len(y)):
        s = class_conditional_set(probs[i], q_by_class)
        sizes.append(len(s))
        cover.append(int(int(y[i]) in s))
    sizes = np.asarray(sizes)
    return {
        "coverage": float(np.mean(cover)),
        "singleton_rate": float(np.mean(sizes == 1)),
        "avg_set_size": float(np.mean(sizes)),
    }


# Backward-compatible name; behavior is now the statistically valid variant.
evaluate_mondrian_by_predicted_class = evaluate_class_conditional

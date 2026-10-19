from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_auc_score, average_precision_score


def classification_metrics(y_true, y_pred, labels=None, target_names=None, prefix=""):
    out = {
        prefix + "accuracy": float(accuracy_score(y_true, y_pred)),
        prefix + "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        prefix + "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    return out


def binary_metrics(y_true, y_pred, y_prob=None, prefix=""):
    out = classification_metrics(y_true, y_pred, prefix=prefix)
    out[prefix + "f1_pos1"] = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    if y_prob is not None and len(np.unique(y_true)) == 2:
        out[prefix + "auroc"] = float(roc_auc_score(y_true, y_prob))
        out[prefix + "auprc"] = float(average_precision_score(y_true, y_prob))
    return out


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.mean()) * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)

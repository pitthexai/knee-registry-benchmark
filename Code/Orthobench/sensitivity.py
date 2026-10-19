from __future__ import annotations

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score


def paired_mask_metrics(y, pred_base, prob_base, pred_mask, prob_mask):
    """Compare masked and unmasked predictions for the same ordered images."""
    y = np.asarray(y)
    pred_base = np.asarray(pred_base)
    pred_mask = np.asarray(pred_mask)
    prob_base = np.asarray(prob_base)
    prob_mask = np.asarray(prob_mask)
    n = len(y)
    if not (len(pred_base) == len(pred_mask) == len(prob_base) == len(prob_mask) == n):
        raise ValueError("Paired arrays must have equal length")
    if n == 0:
        raise ValueError("Cannot compare an empty evaluation set")

    base_correct = pred_base == y
    mask_correct = pred_mask == y
    correct_to_incorrect = base_correct & ~mask_correct
    incorrect_to_correct = ~base_correct & mask_correct
    b = int(correct_to_incorrect.sum())
    c = int(incorrect_to_correct.sum())
    discordant = b + c
    p_value = float(binomtest(min(b, c), discordant, 0.5).pvalue) if discordant else 1.0
    row = np.arange(n)
    base_acc = accuracy_score(y, pred_base)
    mask_acc = accuracy_score(y, pred_mask)
    base_macro = f1_score(y, pred_base, average="macro", zero_division=0)
    mask_macro = f1_score(y, pred_mask, average="macro", zero_division=0)
    base_weighted = f1_score(y, pred_base, average="weighted", zero_division=0)
    mask_weighted = f1_score(y, pred_mask, average="weighted", zero_division=0)
    return {
        "n": n,
        "accuracy": float(mask_acc),
        "delta_accuracy_vs_none": float(mask_acc - base_acc),
        "macro_f1": float(mask_macro),
        "delta_macro_f1_vs_none": float(mask_macro - base_macro),
        "weighted_f1": float(mask_weighted),
        "delta_weighted_f1_vs_none": float(mask_weighted - base_weighted),
        "prediction_flip_rate": float(np.mean(pred_mask != pred_base)),
        "correct_to_incorrect_n": b,
        "correct_to_incorrect_rate": float(correct_to_incorrect.mean()),
        "incorrect_to_correct_n": c,
        "incorrect_to_correct_rate": float(incorrect_to_correct.mean()),
        "mean_delta_true_class_probability": float(np.mean(prob_mask[row, y] - prob_base[row, y])),
        "mean_delta_max_confidence": float(np.mean(prob_mask.max(1) - prob_base.max(1))),
        "mcnemar_exact_p": p_value,
    }

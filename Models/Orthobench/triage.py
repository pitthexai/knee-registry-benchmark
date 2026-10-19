from __future__ import annotations

from math import sqrt
import numpy as np


def wilson_lower_bound(successes: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.0
    phat = successes / n
    denom = 1 + z * z / n
    center = phat + z * z / (2 * n)
    margin = z * sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    return float((center - margin) / denom)


def choose_t_accept_wilson(p, y, target_precision=0.90, min_accept=25):
    best = None
    for t in np.linspace(0, 1, 1001):
        acc = p >= t
        n = int(acc.sum())
        if n < min_accept:
            continue
        tp = int((acc & (y == 1)).sum())
        lb = wilson_lower_bound(tp, n)
        if lb >= target_precision:
            best = {"t_accept": float(t), "n_accept": n, "accept_precision": float(tp / n), "wilson_lb": lb}
            break
    return best


def choose_t_reject(p, y, max_false_reject=0.02):
    best = None
    for t in np.linspace(0, 1, 1001):
        rej = p <= t
        if rej.sum() == 0:
            continue
        false_knee = float((y[rej] == 1).mean())
        if false_knee <= max_false_reject:
            best = {"t_reject": float(t), "reject_false_knee_rate": false_knee, "reject_rate": float(rej.mean())}
    return best


def triage_report(p, y, t_reject, t_accept):
    if t_reject >= t_accept:
        raise ValueError("t_reject must be smaller than t_accept")
    reject = p <= t_reject
    accept = p >= t_accept
    review = ~(reject | accept)
    out = {
        "accept_rate": float(accept.mean()),
        "reject_rate": float(reject.mean()),
        "review_rate": float(review.mean()),
    }
    out["accept_purity"] = float((y[accept] == 1).mean()) if accept.sum() else None
    out["auto_accept_knee_recall"] = float((accept & (y == 1)).sum() / max((y == 1).sum(), 1)) if accept.sum() else None
    # Two different safety questions; report both and name them unambiguously.
    out["reject_contamination_knee_fraction"] = float((y[reject] == 1).mean()) if reject.sum() else None
    out["knee_false_rejection_rate"] = float((reject & (y == 1)).sum() / max((y == 1).sum(), 1))
    return out

"""The organizer metric: macro-averaged F0.5 over Source-1 entities (singletons included)."""
import numpy as np


def f05(pred, truth):
    if not truth:
        return 1.0 if not pred else 0.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(truth)
    return 1.25 * p * r / (0.25 * p + r)


def macro_f05(pred_by_s1, truth_by_s1, s1_ids):
    return float(np.mean([f05(pred_by_s1.get(s, set()), truth_by_s1.get(s, set())) for s in s1_ids]))

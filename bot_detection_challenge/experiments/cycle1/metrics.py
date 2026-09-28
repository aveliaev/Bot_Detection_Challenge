"""Metric utilities. Main metric = official implementation from bot_detection_challenge/metric.py (tie-aware)."""
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # bot_detection_challenge/metric.py
from metric import pr_curve, precision_at_recall, recall_at_fpr  # noqa: E402


def my_precision_at_recall(y, s, r=0.70):
    """Independent re-implementation (group ties via np.unique) to cross-check the official one."""
    y = np.asarray(y, int); s = np.asarray(s, float)
    u, inv = np.unique(-s, return_inverse=True)          # groups ordered by descending score
    tp = np.bincount(inv, weights=y, minlength=len(u)).cumsum()
    k = np.bincount(inv, minlength=len(u)).cumsum()
    rec = tp / y.sum(); prec = tp / k
    ok = rec >= r
    return float(prec[ok].max()) if ok.any() else 0.0


def operating_point(y, s, r=0.70):
    prec, rec = pr_curve(y, s)
    y = np.asarray(y); s = np.asarray(s)
    ok = rec >= r
    i = np.flatnonzero(ok)[np.argmax(prec[ok])]
    order = np.sort(np.unique(s))[::-1]
    thr = order[i]
    return dict(threshold=float(thr), precision=float(prec[i]), recall=float(rec[i]), n_flagged=int((s >= thr).sum()))


def report(y, s):
    return dict(p_at_r70=precision_at_recall(y, s), pr_auc=average_precision_score(y, s),
                roc_auc=roc_auc_score(y, s), r_at_fpr1=recall_at_fpr(y, s, 0.01))


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    cases = [([1, 0, 1, 1, 1], [5, 4, 3, 2, 1], 0.8), ([1, 1, 0, 0], [.5] * 4, 0.5), ([0, 0, 1, 1], [.5] * 4, 0.5),
             ([1, 0, 0, 1, 0, 1], [.9, .9, .8, .8, .1, .1], 0.5),   # recall 0.67 at .8 -> must go to .1
             ([1, 1, 1, 0, 0, 1, 1, 1, 1, 1], [9, 8, 7, 6, 6, 5, 4, 3, 2, 1], 0.8)]
    for y, s, exp in cases:
        a, b = precision_at_recall(y, s), my_precision_at_recall(y, s)
        assert abs(a - exp) < 1e-12 and abs(b - exp) < 1e-12, (y, s, a, b, exp)
    for _ in range(2000):
        n = rng.randint(2, 40); y = rng.randint(0, 2, n)
        if y.sum() == 0: continue
        s = rng.randint(0, 5, n) / 4.0
        assert abs(precision_at_recall(y, s) - my_precision_at_recall(y, s)) < 1e-12
        perm = rng.permutation(n)
        assert precision_at_recall(y[perm], s[perm]) == precision_at_recall(y, s)
    print("metric tests passed (official == independent impl, tie & order invariance)")

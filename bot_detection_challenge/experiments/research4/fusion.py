"""C4 decision-level fusion of independent channel experts (Iliou et al. 2021 idea), evaluated on the B0 folds.

Parameter-free rules are applied directly. Fitted rules (weighted mean, Iliou thresholds on a grid, meta-models) are
cross-fitted: in P0..P4 fold k is scored with parameters fit on the OOF of the other folds of the same partition;
in fwd7 they are fit on a nested inner OOF built from the fwd7 training days only (experts_inner.py).
Missing mouse (no pointer events): fusion over the available experts only (behavior, item) - never scored as 0.
"""
import sys, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common4 import *
from scipy.special import logit, expit
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier

EPS = 1e-4
FOLDS = {}


def folds_of(sch):
    if sch not in FOLDS:
        from rcv import partitions, fwd7
        P = {f'P{i}': p for i, p in enumerate(partitions(DAY, 5))}
        P['fwd7'] = fwd7(DAY)
        FOLDS.update(P)
    return FOLDS[sch]


# ------------------------------------------------------------------ scaling
def scaled(E, mask, how):
    """how='prob' raw probabilities; how='rank' label-free rank percentile inside `mask` (mouse among its available rows)."""
    out = {}
    for e in EXP:
        x = E[e]
        if how == 'prob':
            out[e] = x
        else:
            m = mask & ~np.isnan(x)
            out[e] = rankpct(x, m)
    return out


def trio(S):
    """stack mouse/behavior/item with NaN for missing mouse."""
    return np.c_[S['mouse'], S['behavior'], S['item']]


def nanmean(A): return np.nanmean(A, axis=1)
def nanmin(A): return np.nanmin(A, axis=1)
def nanmax(A): return np.nanmax(A, axis=1)
def nanstd(A): return np.nanstd(A, axis=1)
def geo(A): return np.exp(np.nanmean(np.log(np.clip(A, EPS, 1)), axis=1))
def prod(A): return np.exp(np.nansum(np.log(np.clip(A, EPS, 1)), axis=1) * 3 / np.sum(~np.isnan(A), axis=1))  # renormalised to 3 experts
def harm(A): return np.sum(~np.isnan(A), axis=1) / np.nansum(1 / np.clip(A, EPS, 1), axis=1)


# ------------------------------------------------------------------ parameter-free rules
def iliou(S, th=0.7, tl=0.3, web='behavior'):
    """Iliou et al.: trust the mouse module when it is confident, otherwise average mouse and the web-log module.
    'web' = our analogue of their web-log module ('behavior' or mean of behavior+item). No mouse -> web module only."""
    m = S['mouse']
    w = S['behavior'] if web == 'behavior' else np.nanmean(np.c_[S['behavior'], S['item']], axis=1)
    avg = 0.5 * m + 0.5 * w
    out = np.where(m >= th, m, np.where(m <= tl, m, avg))
    return np.where(np.isnan(m), w, out)


def rules_free(S, B0r):
    A = trio(S)
    cons = geo(A)
    return {
        'mean': nanmean(A), 'geo_mean': geo(A), 'min': nanmin(A), 'product': prod(A), 'harmonic': harm(A),
        'max': nanmax(A),
        'mean+ctx': nanmean(np.c_[A, S['context']]),
        'S4_B0xconsensus': B0r * cons,
        'S4b_B0xmin': B0r * nanmin(A),
        'count>0.9': (A > 0.9).sum(1) + 1e-3 * nanmean(A),
        'count>0.8': (A > 0.8).sum(1) + 1e-3 * nanmean(A),
    }


# ------------------------------------------------------------------ meta features
def meta_X(Sp, B0p):
    """probability-scale inputs -> logits; missing mouse filled with 0 logit-offset + indicator."""
    A = trio(Sp)
    L = logit(np.clip(A, EPS, 1 - EPS))
    avail = ~np.isnan(A[:, 0])
    Lm = np.where(avail, L[:, 0], 0.0)
    b, i = L[:, 1], L[:, 2]
    X = np.c_[logit(np.clip(B0p, EPS, 1 - EPS)), Lm, b, i, avail.astype(float),
              np.nanmin(L, 1), np.nanmax(L, 1), np.nanmean(L, 1), np.nanstd(L, 1),
              np.where(avail, Lm - b, 0), np.where(avail, Lm - i, 0), b - i,
              np.where(avail, Lm * b, 0), np.where(avail, Lm * i, 0), b * i]
    return X


META_NAMES = ['B0', 'mouse', 'behavior', 'item', 'mouse_avail', 'min', 'max', 'mean', 'std', 'm-b', 'm-i', 'b-i', 'm*b', 'm*i', 'b*i']


def fit_predict(kind, Xtr, ytr, Xva, trio_tr=None, trio_va=None):
    if kind == 'meta_lr':
        m = LogisticRegression(C=0.3, max_iter=3000).fit(Xtr, ytr); return m.predict_proba(Xva)[:, 1]
    if kind == 'meta_lr_noB0':
        m = LogisticRegression(C=0.3, max_iter=3000).fit(Xtr[:, 1:], ytr); return m.predict_proba(Xva[:, 1:])[:, 1]
    if kind.startswith('meta_cat'):
        d = int(kind[-1])
        m = CatBoostClassifier(iterations=300, depth=d, learning_rate=0.05, l2_leaf_reg=10, verbose=0, random_seed=0, thread_count=4)
        m.fit(Xtr, ytr); return m.predict_proba(Xva)[:, 1]
    raise ValueError(kind)


def best_weighted(Str, ytr):
    """grid over simplex weights for mean of (mouse, behavior, item) on rank scale; missing mouse -> renormalised."""
    best, bw = -1, None
    grid = [(a, b, 1 - a - b) for a in np.arange(0, 1.01, .1) for b in np.arange(0, 1.01 - a, .1)]
    for w in grid:
        s = wmean(Str, w)
        v = pr(ytr, s)
        if v > best: best, bw = v, w
    return bw


def wmean(S, w):
    A = trio(S); W = np.array(w)[None, :] * ~np.isnan(A)
    return np.nansum(np.nan_to_num(A) * W, 1) / np.clip(W.sum(1), 1e-9, None)


def best_iliou(Str, ytr, web):
    best, bp = -1, None
    for th in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        for tl in [0.05, 0.1, 0.2, 0.3, 0.4]:
            if tl >= th: continue
            v = pr(ytr, iliou(Str, th, tl, web))
            if v > best: best, bp = v, (th, tl)
    return bp

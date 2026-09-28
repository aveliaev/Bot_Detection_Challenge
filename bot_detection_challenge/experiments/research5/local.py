"""Cycle 5: LOCAL reranking of a narrow band around the B0 operating point. B0 is frozen.

Inputs are only the existing OOF scores of cycle 4 (B0, mouse, behavior, item, context), platform, mouse_available and
disagreement terms. Outside the band score_new = B0. Inside the band score_new = rank(B0) + lambda * correction, clipped
to the band's own rank interval (so nothing outside the band moves and band members can only swap with each other).

Methods (all trained only on OOF rows of training folds):
  local_lr  (band, delta, lambda): for a candidate with B0 prob p, logistic regression on training rows of the band with
            |B0_train - p| < delta; correction = centred local posterior (2*(p_hat - window_rate)).
  pair      (band, eps, lambda):   pairwise logistic model on X(TP) - X(neg) for training pairs in the band with
            |B0_TP - B0_neg| < eps; correction = tanh(w . x_standardised).
  pair_sub  same, separate model per subtype (web+mouse / web no mouse / mobile), mouse inputs only for web+mouse.
  rule      (band, tm, ti, tb, shift): mouse > tm & item > ti & behavior < tb (mouse available) -> shift down `shift` ranks.
"""
from __future__ import annotations
import sys, itertools, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
R5 = Path(__file__).resolve().parent
sys.path.insert(0, str(R5.parent / 'research4'))
from common4 import *   # noqa
from fusion import folds_of
from scipy.special import logit
from sklearn.linear_model import LogisticRegression

EPS = 1e-4
PL = pd.read_parquet(ROOT / 'research3' / 'cache' / 'traits.parquet').pl.reindex(IDS).values
BANDS = {'top4-9%': (.04, .09), 'top4-10%': (.04, .10), 'top5-10%': (.05, .10), 'top5-12%': (.05, .12),
         'rank300-800': (300 / 9522, 800 / 9522), 'rank400-800': (400 / 9522, 800 / 9522)}
LAMBDAS = [0.01, 0.02, 0.05, 0.1]
DELTAS = [0.02, 0.05, 0.10]
EPSS = [0.02, 0.05, 0.10]
RULE_GRID = list(itertools.product([.90, .95], [.85, .90, .95], [.2, .3, .4], [5, 10, 20, 30]))
FEATS = ['B0', 'mouse', 'behavior', 'item', 'context', 'mouse_avail', 'web_mouse', 'web_nomouse', 'mobile',
         'm-i', 'b-i', 'm*i', 'std', 'min', 'max']
MOUSE_F = [FEATS.index(c) for c in ['mouse', 'm-i', 'm*i']]


class Rows:
    """a scored population (one scheme's pooled OOF, or a nested inner OOF): features, B0 prob, percentile score."""
    def __init__(self, E, mask, fold):
        idx = np.flatnonzero(mask)
        self.idx, self.y, self.ids, self.fold = idx, Y[idx], IDS[idx], fold[idx]
        g = lambda e: E[e][idx]
        m, b, i, c, p = g('mouse'), g('behavior'), g('item'), g('context'), g('B0')
        av = ~np.isnan(m)
        L = lambda x: logit(np.clip(x, EPS, 1 - EPS))
        A = np.c_[m, b, i]
        pl = PL[idx]
        self.sub = np.where(pl == 'web', np.where(av, 'web_mouse', 'web_nomouse'), 'mobile')
        X = np.c_[L(p), np.where(av, L(m), 0), L(b), L(i), L(c), av, self.sub == 'web_mouse', self.sub == 'web_nomouse',
                  self.sub == 'mobile', np.where(av, m - i, 0), b - i, np.where(av, m * i, 0),
                  np.nanstd(A, 1), np.nanmin(A, 1), np.nanmax(A, 1)].astype(float)
        self.X, self.p, self.m, self.b, self.i, self.av = X, p, m, b, i, av
        self.n = len(idx)
        self.s = pd.Series(p).rank(pct=True).values           # higher = more bot-like
        self.top = 1 - self.s                                   # fraction from the top


def in_band(R, band, rows=None):
    lo, hi = BANDS[band]
    t = R.top if rows is None else R.top[rows]
    return (t > lo) & (t <= hi)


# ------------------------------------------------------------------ corrections (training rows tr of TR, applied to rows ap of AP)
def _std(Xtr):
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd < 1e-9] = 1
    return mu, sd


def corr_local_lr(TR, tr, AP, ap, delta):
    tr = tr[in_band(TR, BAND_CUR, tr)]
    c = np.zeros(len(ap))
    if len(tr) < 30: return c
    ptr = TR.p[tr]; step = delta / 2
    grid = np.round(AP.p[ap] / step) * step
    for g in np.unique(grid):
        w = tr[np.abs(ptr - g) < delta]
        if len(w) < 20 or TR.y[w].min() == TR.y[w].max(): continue
        mu, sd = _std(TR.X[w])
        lr = LogisticRegression(C=0.5, max_iter=2000).fit((TR.X[w] - mu) / sd, TR.y[w])
        k = grid == g
        c[k] = 2 * (lr.predict_proba((AP.X[ap[k]] - mu) / sd)[:, 1] - TR.y[w].mean())
    return np.clip(c, -1, 1)


def _pair_model(TR, tr, eps, cols, rng):
    pos, neg = tr[TR.y[tr] == 1], tr[TR.y[tr] == 0]
    if len(pos) < 5 or len(neg) < 5: return None
    o = np.argsort(TR.p[neg]); pn = TR.p[neg][o]
    P = []
    for a in pos:
        lo, hi = np.searchsorted(pn, TR.p[a] - eps), np.searchsorted(pn, TR.p[a] + eps)
        P += [(a, neg[o[j]]) for j in range(lo, hi)]
    if len(P) < 50: return None
    P = np.array(P)
    if len(P) > 30000: P = P[rng.choice(len(P), 30000, replace=False)]
    mu, sd = _std(TR.X[tr][:, cols])
    D = (TR.X[P[:, 0]][:, cols] - TR.X[P[:, 1]][:, cols]) / sd
    lr = LogisticRegression(C=0.3, fit_intercept=False, max_iter=3000).fit(np.r_[D, -D], np.r_[np.ones(len(D)), np.zeros(len(D))])
    return lr.coef_[0], mu, sd


def corr_pair(TR, tr, AP, ap, eps, by_sub=False):
    tr = tr[in_band(TR, BAND_CUR, tr)]
    rng = np.random.RandomState(0)
    c = np.zeros(len(ap))
    groups = [('all', None)] if not by_sub else [('web_mouse', 'web_mouse'), ('web_nomouse', 'web_nomouse'), ('mobile', 'mobile')]
    for name, sub in groups:
        cols = list(range(len(FEATS))) if (sub in (None, 'web_mouse')) else [j for j in range(len(FEATS)) if j not in MOUSE_F]
        t = tr if sub is None else tr[TR.sub[tr] == sub]
        a = np.arange(len(ap)) if sub is None else np.flatnonzero(AP.sub[ap] == sub)
        if not len(a): continue
        mdl = _pair_model(TR, t, eps, cols, rng)
        if mdl is None: continue
        w, mu, sd = mdl
        z = ((AP.X[ap[a]][:, cols] - mu) / sd) @ w
        z = z - np.median(((TR.X[t][:, cols] - mu) / sd) @ w)
        c[a] = np.tanh(z)
    return c


def apply_rule(AP, ap, tm, ti, tb):
    return -(AP.av[ap] & (np.nan_to_num(AP.m[ap]) > tm) & (AP.i[ap] > ti) & (AP.b[ap] < tb)).astype(float)


# ------------------------------------------------------------------ config handling
def configs():
    C = [('none',)]
    for b in BANDS:
        C += [('local_lr', b, d) for d in DELTAS]
        C += [('pair', b, e) for e in EPSS]
        C += [('pair_sub', b, e) for e in EPSS]
        C += [('rule', b) + g[:3] for g in {g[:3] for g in RULE_GRID}]
    return C


def corrections(cfg, TR, tr, AP, ap):
    """correction in [-1,1] for AP rows ap (0 outside the band)."""
    global BAND_CUR
    if cfg[0] == 'none': return np.zeros(len(ap))
    BAND_CUR = cfg[1]
    inb = in_band(AP, cfg[1], ap)
    c = np.zeros(len(ap))
    if not inb.any(): return c
    sel = ap[inb]
    if cfg[0] == 'local_lr': c[inb] = corr_local_lr(TR, tr, AP, sel, cfg[2])
    elif cfg[0] == 'pair': c[inb] = corr_pair(TR, tr, AP, sel, cfg[2])
    elif cfg[0] == 'pair_sub': c[inb] = corr_pair(TR, tr, AP, sel, cfg[2], by_sub=True)
    elif cfg[0] == 'rule': c[inb] = apply_rule(AP, sel, *cfg[2:5])
    return c


def new_score(AP, ap, cfg, c, strength):
    """strength = lambda (percentile units) for learned corrections, or rank shift (positions) for rules."""
    s = AP.s[ap].copy()
    if cfg[0] == 'none': return s
    lo, hi = BANDS[cfg[1]]
    inb = in_band(AP, cfg[1], ap)
    step = strength / AP.n if cfg[0] == 'rule' else strength
    s[inb] = np.clip(s[inb] + step * c[inb], 1 - hi + 1e-9, 1 - lo)
    return s


def strengths(cfg):
    return [0] if cfg[0] == 'none' else ([5, 10, 20, 30] if cfg[0] == 'rule' else LAMBDAS)

"""Cycle 6.5 cross-modal conditional mappings (label-free) and residual summaries.

A mapping predicts a TARGET channel representation from SOURCE channel(s) + context:
  CB : context -> behaviour (baseline)        MB  : mouse + context -> behaviour   (mouse rows only)
  IB : item + context -> behaviour            MIB : mouse + item + context -> behaviour (mouse rows only)
  CI : context -> item (baseline)             BI  : behaviour + context -> item
All preprocessing (median imputation, input scaling, target median/IQR standardisation) is fitted on the rows the
mapping is trained on; residual = standardised(actual) - prediction, per target dimension, sign kept."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from representations import BEH, ITEM, TIMING_SIGN, CTX

MAPS = {'CB': (['context'], 'behavior'), 'MB': (['mouse', 'context'], 'behavior'), 'IB': (['item', 'context'], 'behavior'),
        'MIB': (['mouse', 'item', 'context'], 'behavior'), 'CI': (['context'], 'item'), 'BI': (['behavior', 'context'], 'item'),
        'CB_m': (['context'], 'behavior'), 'IB_m': (['item', 'context'], 'behavior')}   # baselines on the mouse rows
NEEDS_MOUSE = {'MB', 'MIB', 'CB_m', 'IB_m'}


class Mapping:
    def __init__(self, name, alpha=1.0):
        self.name, self.alpha = name, alpha
        self.src, self.tgt = MAPS[name]

    def _X(self, R, rows):
        return pd.concat([R[s].loc[rows] for s in self.src], axis=1)

    def rows(self, R, idx):
        if self.name in NEEDS_MOUSE:
            return idx[R['context'].loc[idx, 'c_mouse'].values == 1]
        return idx

    def fit(self, R, idx):
        idx = self.rows(R, idx)
        X = self._X(R, idx); Y = R[self.tgt].loc[idx]
        self.med_x = X.median(); Xf = X.fillna(self.med_x)
        self.mu, self.sd = Xf.mean(), Xf.std().replace(0, 1)
        self.c, self.s = Y.median(), (Y.quantile(.75) - Y.quantile(.25)).replace(0, 1)
        Z = (Y - self.c) / self.s
        Xs = ((Xf - self.mu) / self.sd).values
        self.models = {}
        for col in Y.columns:
            m = Z[col].notna().values
            self.models[col] = Ridge(alpha=self.alpha).fit(Xs[m], Z[col].values[m])
        return self

    def std_target(self, R, idx):
        return (R[self.tgt].loc[idx] - self.c) / self.s

    def predict(self, R, idx, X_override=None):
        X = self._X(R, idx) if X_override is None else X_override
        Xs = ((X.fillna(self.med_x) - self.mu) / self.sd).values
        return pd.DataFrame({c: m.predict(Xs) for c, m in self.models.items()}, index=idx)

    def residual(self, R, idx, X_override=None):
        idx = self.rows(R, idx)
        return self.std_target(R, idx) - self.predict(R, idx, X_override)


def summarize(name, r: pd.DataFrame) -> pd.DataFrame:
    """compact residual summaries (<= 7 per mapping)."""
    a = r.abs()
    S = pd.DataFrame(index=r.index)
    S[f'{name}_L1'] = a.mean(1); S[f'{name}_L2'] = np.sqrt((r ** 2).mean(1)); S[f'{name}_max'] = a.max(1)
    if set(r.columns) <= set(BEH):
        t = [c for c, g in BEH.items() if g == 'timing' and TIMING_SIGN[c] != 0]
        S[f'{name}_timing'] = sum(r[c] * TIMING_SIGN[c] for c in t) / len(t)            # + = pauses longer than expected
        S[f'{name}_dtmed'] = r['b_ldt_median']                                           # actual - expected median pause
        S[f'{name}_activity'] = r[[c for c, g in BEH.items() if g == 'activity']].mean(1)  # + = more activity than expected
        S[f'{name}_eventmix'] = r[[c for c, g in BEH.items() if g == 'eventmix']].abs().mean(1)
    else:
        for grp in ('pool', 'diversity', 'navigation'):
            S[f'{name}_{grp}'] = r[[c for c, g in ITEM.items() if g == grp]].mean(1)     # + = more than expected
    return S


def quality(mp: Mapping, R, idx, X_override=None):
    """per-dimension MAE / R2 / Spearman of the mapping on rows idx (standardised target space)."""
    idx = mp.rows(R, idx)
    Z = mp.std_target(R, idx); P = mp.predict(R, idx, X_override)
    out = {}
    for c in Z.columns:
        m = Z[c].notna()
        z, p = Z[c][m], P[c][m]
        out[c] = dict(mae=float((z - p).abs().mean()), mae_const=float(z.abs().mean()),
                      r2=float(1 - ((z - p) ** 2).sum() / ((z - z.mean()) ** 2).sum()), spearman=float(z.corr(p, method='spearman')))
    return pd.DataFrame(out).T


def shuffled_source(mp: Mapping, R, idx, rng):
    """negative control: permute the non-context source channels between cookies inside strata
    (platform x mouse availability x event-count quintile); context stays with its cookie."""
    idx = mp.rows(R, idx)
    X = mp._X(R, idx).copy()
    ctx = R['context'].loc[idx]
    nb = pd.qcut(R['behavior'].loc[idx, 'b_log_n'].rank(method='first'), 5, labels=False)
    strata = ctx.c_web.astype(int).astype(str) + ctx.c_android.astype(int).astype(str) + ctx.c_mouse.astype(int).astype(str) + nb.astype(str)
    cols = [c for s in mp.src if s != 'context' for c in R[s].columns]
    for _, g in X.groupby(strata.values).groups.items():
        g = list(g)
        X.loc[g, cols] = X.loc[g, cols].values[rng.permutation(len(g))]
    return X

"""C7.6 cookie-level chunk-score features (locked list) from window scores."""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
R7 = Path(__file__).resolve().parent
LOCKED = ['seq_full', 'chunk_mean', 'chunk_std', 'chunk_q10', 'chunk_median', 'chunk_q90', 'chunk_min', 'chunk_max',
          'len5_mean', 'len10_mean', 'len20_mean', 'len5_std', 'position_std_L5', 'frac_gt07']
_W = None


def geom():
    global _W
    if _W is None:
        _W = pd.read_parquet(R7 / 'cache' / 'windows.parquet', columns=['cookie_id', 'L', 'kind', 'posk'])
    return _W


def aggregate(scores: pd.Series) -> pd.DataFrame:
    """scores indexed by window id (deterministic inference windows) -> cookie features."""
    G = geom().loc[scores.index].assign(s=scores.values)
    full = G[G.L == -1].groupby('cookie_id').s.first()
    C = G[G.L > 0]
    g = C.groupby('cookie_id').s
    q = lambda p: g.quantile(p)
    F = pd.DataFrame({'seq_full': full})
    F = F.join(pd.DataFrame({'chunk_mean': g.mean(), 'chunk_std': g.std(), 'chunk_min': g.min(), 'chunk_max': g.max(),
                             'chunk_q10': q(.1), 'chunk_q25': q(.25), 'chunk_median': q(.5), 'chunk_q75': q(.75), 'chunk_q90': q(.9),
                             'frac_gt05': (C.s > .5).groupby(C.cookie_id).mean(), 'frac_gt07': (C.s > .7).groupby(C.cookie_id).mean(),
                             'frac_gt09': (C.s > .9).groupby(C.cookie_id).mean()}), how='left')
    F['chunk_range'] = F.chunk_max - F.chunk_min
    for L in (3, 5, 10, 20, 40):
        F[f'len{L}_mean'] = C[C.L == L].groupby('cookie_id').s.mean()
    for L in (5, 10, 20):
        F[f'len{L}_std'] = C[C.L == L].groupby('cookie_id').s.std()
    for L in (5, 10):
        P = C[C.L == L]
        first = P.sort_values('posk').groupby('cookie_id').s.first(); last = P.sort_values('posk').groupby('cookie_id').s.last()
        F[f'early_minus_late_L{L}'] = first - last
        F[f'position_std_L{L}'] = P.groupby('cookie_id').s.std()
    F.index.name = 'cookie_id'
    return F


def scheme_features(sch, k_list, col='cat', part='val'):
    out = []
    for k in k_list:
        t = pd.read_parquet(R7 / 'oof' / f'winscores_{sch}_k{k}.parquet')
        t = t[t.part == part]
        out.append(aggregate(t[col]))
    return pd.concat(out)

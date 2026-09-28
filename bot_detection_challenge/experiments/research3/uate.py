"""C3.12 UA-token temporal target encoding (fold-aware, same temporal rule as the item pool):
cookie of day D uses labels (from `lab`) of days [E-hist, E-1], E = min(D, last labelled day + 1). Constant prior."""
from __future__ import annotations
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from common import CACHE, meta

_C = None


def _tokens():
    global _C
    if _C is None:
        _C = pd.read_parquet(CACHE / 'ua_tokens.parquet')[['full', 'pl_dev', 'browser', 'day']]
    return _C


def ua_te_features(D, lab, ids, cols=('full', 'pl_dev', 'browser'), hist=7, a=20.0, p0=0.08):
    C = _tokens(); ids = pd.Index(ids)
    L = C.loc[lab.index].assign(y=lab.astype(int).values)
    last = int(L.day.max())
    out = pd.DataFrame(index=ids, columns=[f'uate_{c}' for c in cols], dtype=float)
    days = C.loc[ids, 'day']
    for d in np.sort(days.unique()):
        E = min(d, last + 1)
        h = L[(L.day < E) & (L.day >= E - hist)]
        q = ids[days.values == d]
        if not len(h):
            continue
        for c in cols:
            g = h.groupby(c).y.agg(['sum', 'size'])
            r = (g['sum'] + a * p0) / (g['size'] + a)
            out.loc[q, f'uate_{c}'] = C.loc[q, c].map(r).fillna(p0).values
    out.index.name = 'cookie_id'
    return out

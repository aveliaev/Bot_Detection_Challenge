"""Cycle 6.5 final integration: original B0 features + strict-OOF cross-modal residual family, same CatBoost as B0.
Training rows receive their inner leave-one-day-out residuals of that outer fold ('train' part), validation rows the
outer residuals ('val' part). Usage: python research65/integrate.py NAME PREFIX [PREFIX ...]   (e.g. IB BI MIB)."""
import sys, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
R65 = Path(__file__).resolve().parent; ROOT = R65.parent
sys.path.insert(0, str(ROOT / 'research2')); sys.path.insert(0, str(ROOT / 'research6'))
import numpy as np, pandas as pd
from rcv import Harness
import rcv
from cv import make_model

if __name__ == '__main__':
    name, prefixes = sys.argv[1], sys.argv[2:]
    rcv.OOF2 = ROOT / 'research6' / 'oof'; rcv.LOG = R65 / 'experiments65.jsonl'   # B0 reference OOF lives in research6/oof
    H = Harness()
    cache = {}

    def fam(sch, k):
        if (sch, k) not in cache:
            t = pd.read_parquet(R65 / 'oof' / f'resid_{sch}_k{k}.parquet')
            cols = [c for c in t.columns if c.split('_')[0] in prefixes]
            cache[(sch, k)] = (t[t.part == 'train'][cols], t[t.part == 'val'][cols])
        return cache[(sch, k)]

    def fit_fn(Xtr, ytr, Xva, seed, tri, vai, sch, k):
        Rtr, Rva = fam(sch, k)
        A, B = Xtr.join(Rtr), Xva.join(Rva)
        m = make_model('cat', None, seed).fit(A, ytr)
        return m.predict_proba(B[A.columns])[:, 1]

    H.run(name, fit_fn=fit_fn, hypothesis=f'C6.5 B0 + cross-modal residuals {prefixes}', leak_risk='strict OOF (inner LODO for training rows)')
    sys.argv = ['x']
    from r6_integrate import report
    report(name)

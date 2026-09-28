"""C7 B0 + compact chunk-score features (locked list or a subset), same CatBoost as B0, strict OOF:
training cookies get aggregates of their INNER window scores (4 day-group CatBoost models of the outer training part),
validation cookies get aggregates of the OUTER window scores. Usage: python research7/integrate_b0.py NAME [feature ...]"""
import sys, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
R7 = Path(__file__).resolve().parent; ROOT = R7.parent
sys.path.insert(0, str(R7)); sys.path.insert(0, str(ROOT / 'research2')); sys.path.insert(0, str(ROOT / 'research6'))
import numpy as np, pandas as pd
from rcv import Harness
import rcv
from cv import make_model
from build_seq_features import aggregate, LOCKED

if __name__ == '__main__':
    name = sys.argv[1]; feats = sys.argv[2:] or LOCKED
    rcv.OOF2 = ROOT / 'research6' / 'oof'; rcv.LOG = R7 / 'experiments7.jsonl'
    H = Harness(); cache = {}

    def fam(sch, k):
        if (sch, k) not in cache:
            t = pd.read_parquet(R7 / 'oof' / f'winscores_{sch}_k{k}.parquet')
            tr, va = aggregate(t[t.part == 'train'].cat), aggregate(t[t.part == 'val'].cat)
            cache[(sch, k)] = (tr[feats].add_prefix('s7_'), va[feats].add_prefix('s7_'))
        return cache[(sch, k)]

    def fit_fn(Xtr, ytr, Xva, seed, tri, vai, sch, k):
        Rtr, Rva = fam(sch, k)
        A, B = Xtr.join(Rtr), Xva.join(Rva)
        return make_model('cat', None, seed).fit(A, ytr).predict_proba(B[A.columns])[:, 1]

    H.run(name, fit_fn=fit_fn, hypothesis=f'C7 B0 + chunk features {feats}', leak_risk='strict OOF (inner day-group window models)')
    sys.argv = ['x']
    from r6_integrate import report
    report(name)

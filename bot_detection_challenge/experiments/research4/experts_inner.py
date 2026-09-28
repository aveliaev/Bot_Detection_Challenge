"""C4 nested inner OOF for the fwd7 folds (training days only) -> used to fit fusion weights / meta-models for fwd7
without touching the fwd7 validation labels. Inner split: leave-one-day-out over days 2..k-1; the item pool of each inner
fold is rebuilt from the inner training labels only."""
import sys, json, time, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'research2'))
import numpy as np
from rcv import Harness, BASE_COLS
from cv import make_model, POOL_COLS
from history import pool_features
from channels import R4
H = Harness(n_rep=1); D = H.D
CH = json.load(open(R4 / 'channels.json'))
mouse_ok = (D.F.loc[D.ids, 'ptr_n'].fillna(0) > 0).values
B0C = [c for c in BASE_COLS if c not in POOL_COLS] + [c for c in BASE_COLS if c in POOL_COLS]
EXPERTS = {'B0': (B0C, True), 'mouse': (CH['mouse'], False), 'behavior': (CH['behavior'], False),
           'item': ([c for c in CH['item'] if c not in POOL_COLS], True), 'context': (CH['context'], False)}
def X(cols, pool, idx, P):
    Xd = D.F.loc[D.ids[idx], [c for c in cols if c in D.F.columns and c not in POOL_COLS]]
    return Xd.join(P[POOL_COLS]) if pool else Xd
for k in (5, 7):
    t0 = time.time()
    out = {e: np.full(len(D.y), np.nan) for e in EXPERTS}
    for d in range(2, k):
        tri = np.flatnonzero((D.day < k) & (D.day != d)); vai = np.flatnonzero(D.day == d)
        P = pool_features(D.ci, D.labels.iloc[tri], D.ids, 7)
        for e, (cols, pool) in EXPERTS.items():
            tr, va = (tri[mouse_ok[tri]], vai[mouse_ok[vai]]) if e == 'mouse' else (tri, vai)
            m = make_model('cat', None, 0).fit(X(cols, pool, tr, P), D.y[tr])
            out[e][va] = m.predict_proba(X(cols, pool, va, P))[:, 1]
    np.savez(R4 / 'oof' / f'experts_fwd7inner_k{k}.npz', **out)
    print('inner k', k, 'done', round(time.time() - t0), 's', flush=True)

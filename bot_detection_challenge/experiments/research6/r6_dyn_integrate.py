"""C6.9 integration test: B0 features + prefix score-dynamics family, same CatBoost as B0, same folds.
Training rows of the final model receive dynamics computed from OOF prefix predictions only:
  P0..P4 : cross-fitted within the partition (the row's own fold model never saw it; standard 2-level stacking)
  fwd7   : nested inner OOF on the training days only (leave-one-day-out over days 2..k-1), so no validation label is
           ever used to build a training feature.
Validation rows receive the dynamics of their outer OOF prefix predictions."""
import sys, time, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
R6 = Path(__file__).resolve().parent; ROOT = R6.parent
sys.path.insert(0, str(R6)); sys.path.insert(0, str(ROOT / 'research2'))
import numpy as np, pandas as pd
from rcv import Harness, BASE_COLS, precision_at_recall
import rcv
from cv import make_model, POOL_COLS
from prefix_target import pool_prefix, FRACS
from r6_dynamics import dyn

COLS = ['dyn_d25_50', 'dyn_d50_75', 'dyn_d75_100', 'dyn_total', 'dyn_max_jump', 'dyn_slope', 'dyn_var',
        'dyn_first_gt0.5', 'dyn_first_gt0.7', 'dyn_first_gt0.9', 'dyn_P25', 'dyn_P50']


def inner_fwd7(H, k):
    """nested inner prefix OOF (P25/P50/P75 + inner B0 as P100) for training days of fwd7 fold k."""
    p = R6 / 'oof' / f'prefix_fwd7inner_k{k}.npz'
    if p.exists(): return dict(np.load(p))
    D = H.D; ci_full = pd.read_parquet(ROOT / 'work' / 'cookie_items.parquet')
    FF = {f: pd.read_parquet(R6 / 'cache' / f'features_f{f}.parquet') for f in FRACS}
    CI = {f: pd.read_parquet(R6 / 'cache' / f'ci_f{f}.parquet') for f in FRACS}
    B0C = [c for c in BASE_COLS if c not in POOL_COLS]; PC = [c for c in BASE_COLS if c in POOL_COLS]
    out = {f'P{f}': np.full(len(D.y), np.nan) for f in FRACS}
    out['P100'] = np.load(ROOT / 'research4' / 'oof' / f'experts_fwd7inner_k{k}.npz')['B0']
    for d in range(2, k):
        tri = np.flatnonzero((D.day < k) & (D.day != d)); vai = np.flatnonzero(D.day == d)
        lab = D.labels.iloc[tri]
        for f in FRACS:
            pool = pool_prefix(ci_full, CI[f], lab, D.ids, 7)
            X = lambda idx: FF[f].loc[D.ids[idx], B0C].join(pool[PC])[B0C + PC]
            out[f'P{f}'][vai] = make_model('cat', None, 0).fit(X(tri), D.y[tri]).predict_proba(X(vai))[:, 1]
    np.savez(p, **out)
    return out


def dyn_frame(z, ids):
    P = np.c_[z['P25'], z['P50'], z['P75'], z['P100']]
    m = ~np.isnan(P).any(1)
    Fz = dyn(P[m]); Fz.index = ids[m]
    return Fz[COLS]


if __name__ == '__main__':
    t0 = time.time()
    H = Harness(); D = H.D
    rcv.OOF2 = R6 / 'oof'; rcv.LOG = R6 / 'experiments6.jsonl'
    Zs = {s: dyn_frame(np.load(R6 / 'oof' / f'prefix_{s}.npz'), D.ids) for s in H.schemes}
    inner = {k: dyn_frame(inner_fwd7(H, k), D.ids) for k in (5, 7)}
    print('inner fwd7 prefix OOF ready', round(time.time() - t0), 's', flush=True)

    def fit_fn(Xtr, ytr, Xva, seed, tri, vai, sch, k):
        if sch == 'fwd7':
            kk = (5, 7)[k]; Ztr = inner[kk]; Zva = Zs['fwd7']
        else:
            Ztr = Zva = Zs[sch]
        A = Xtr.join(Ztr); B = Xva.join(Zva)
        m = make_model('cat', None, seed).fit(A, ytr)
        return m.predict_proba(B[A.columns])[:, 1]

    H.run('E6_dynamics', fit_fn=fit_fn, hypothesis='C6.9 B0 + prefix score dynamics (strict OOF / nested)', leak_risk='stacked OOF')
    sys.argv = ['x']
    from r6_integrate import report
    report('E6_dynamics')

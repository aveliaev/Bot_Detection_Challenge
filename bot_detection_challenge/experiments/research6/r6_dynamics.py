"""C6.8 prefix score dynamics: trajectory features from strict-OOF P25/P50/P75/P100 and their diagnostics."""
import sys, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
R6 = Path(__file__).resolve().parent; ROOT = R6.parent
sys.path.insert(0, str(ROOT / 'research4'))
import numpy as np, pandas as pd
from common4 import SCHEMES, IDS, Y, PFP, pr
pd.set_option('display.width', 250)
FR = np.array([.25, .5, .75, 1.0])


def dyn(P):
    """P: (n, 4) probabilities at 25/50/75/100 % -> trajectory features."""
    d = np.diff(P, axis=1)
    x = FR - FR.mean()
    Fz = pd.DataFrame({'dyn_d25_50': d[:, 0], 'dyn_d50_75': d[:, 1], 'dyn_d75_100': d[:, 2], 'dyn_total': P[:, 3] - P[:, 0],
                       'dyn_max_jump': np.abs(d).max(1), 'dyn_mean_jump': d.mean(1),
                       'dyn_slope': (P - P.mean(1, keepdims=True)) @ x / (x @ x), 'dyn_var': P.var(1),
                       'dyn_monotone': (d >= -1e-9).all(1).astype(float),
                       'dyn_sign_changes': (np.diff(np.sign(d), axis=1) != 0).sum(1).astype(float),
                       'dyn_P25': P[:, 0], 'dyn_P50': P[:, 1], 'dyn_P75': P[:, 2]})
    for t in (.5, .7, .9):
        hit = P >= t
        Fz[f'dyn_first_gt{t}'] = np.where(hit.any(1), FR[hit.argmax(1)], 1.25)
    return Fz


def scheme_dyn(sch):
    z = np.load(R6 / 'oof' / f'prefix_{sch}.npz')
    P = np.c_[z['P25'], z['P50'], z['P75'], z['P100']]
    m = ~np.isnan(P).any(1)
    Fz = dyn(P[m]); Fz.index = IDS[m]
    return Fz, m, P


if __name__ == '__main__':
    print('== P@R of the prefix target models (strict OOF, same rows as B0)')
    rows = []
    for s in SCHEMES:
        z = np.load(R6 / 'oof' / f'prefix_{s}.npz'); m = ~np.isnan(z['P100'])
        rows.append({'scheme': s, **{f: pr(Y[m], z[f][m]) for f in ['P25', 'P50', 'P75', 'P100']}})
    Q = pd.DataFrame(rows).set_index('scheme'); Q.loc['rep mean'] = Q.loc[SCHEMES[:5]].mean()
    print(Q.round(4))
    # averaged over the 5 partitions (each row OOF in every partition)
    Z = [scheme_dyn(s)[0] for s in SCHEMES[:5]]
    A = sum(Z) / 5
    A.to_parquet(R6 / 'cache' / 'dyn_repavg.parquet')
    o = pd.read_csv(ROOT / 'research2' / 'oof_final_B0.csv').set_index('cookie_id'); o = o[o.oof_repeated_dayblock > 0]
    rk = o.oof_repeated_dayblock.rank(ascending=False)
    g = pd.Series('easy neg', index=o.index)
    g[(o.target == 0) & rk.between(605, 1500)] = 'hard neg'
    g[(o.target == 0) & (rk <= 604)] = 'other FP'
    g[o.index.isin(PFP)] = 'persistent FP'
    g[(o.target == 1) & (rk <= 604)] = 'TP'
    g[(o.target == 1) & (rk > 604)] = 'FN'
    Pm = A[['dyn_P25', 'dyn_P50', 'dyn_P75']].assign(P100=A.dyn_P25 + A.dyn_total)
    print('\n== mean trajectory P(bot) by group (averaged repeated OOF)')
    print(Pm.groupby(g.reindex(Pm.index)).agg(['mean', 'median']).round(3))
    print('\n== mean dynamics features by group')
    print(A.groupby(g.reindex(A.index)).mean().round(3).T)

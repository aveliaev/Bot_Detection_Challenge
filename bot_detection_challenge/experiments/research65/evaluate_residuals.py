"""Cycle 6.5 evaluation: (1) mapping quality vs same-row baselines + negative controls; (2) diagnostics of the OOF residuals
(persistent FP vs matched TP with bootstrap CI, conditional band AUC, length stratification, orthogonality) via the
cycle-6 gate evaluator; (3) temporal stability by week."""
import sys, subprocess, shutil, json, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
R65 = Path(__file__).resolve().parent; ROOT = R65.parent
sys.path.insert(0, str(R65)); sys.path.insert(0, str(ROOT / 'research4'))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from representations import BEH
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)

# ---------------- (1) mapping quality
Q = pd.read_csv(R65 / 'mapping_quality.csv')
Q['group'] = Q.dim.map(lambda d: BEH.get(d, 'item'))
agg = Q.groupby(['scheme', 'mapping', 'control'])[['mae', 'mae_const', 'r2', 'spearman']].mean().round(4)
print('== (1) mapping quality (mean over target dims and folds; standardised targets)')
print(agg.to_string())
pairs = [('MB', 'CB_m', 'mouse over context (mouse rows)'), ('IB', 'CB', 'item over context'), ('MIB', 'IB_m', 'mouse over item+context (mouse rows)'),
         ('MIB', 'CB_m', 'mouse+item over context (mouse rows)'), ('BI', 'CI', 'behaviour over context')]
rows = []
for sch in ('P0', 'fwd7'):
    for a, b, lab in pairs:
        A = agg.loc[(sch, a, 'none')]; B = agg.loc[(sch, b, 'none')]
        S = agg.loc[(sch, a, 'shuffled_source')] if (sch, a, 'shuffled_source') in agg.index else None
        rows.append(dict(scheme=sch, mapping=a, baseline=b, what=lab, dMAE=A.mae - B.mae, dR2=A.r2 - B.r2, R2=A.r2, R2_base=B.r2,
                         shuffled_MAE_increase=(S.mae - A.mae) if S is not None else np.nan))
print('\n== incremental predictability over the same-row baseline, and negative control (MAE increase when source shuffled)')
print(pd.DataFrame(rows).round(4).to_string(index=False))
g = Q[(Q.control == 'none') & Q.mapping.isin(['CB_m', 'MB', 'CB', 'IB', 'IB_m', 'MIB'])].groupby(['scheme', 'mapping', 'group']).r2.mean().unstack().round(3)
print('\n== R2 by behaviour group'); print(g)

# ---------------- (2) averaged OOF validation residuals over P0..P4 -> gate diagnostics
V = []
for s in ['P0', 'P1', 'P2', 'P3', 'P4']:
    parts = [pd.read_parquet(R65 / 'oof' / f'resid_{s}_k{k}.parquet') for k in range(6)]
    V.append(pd.concat([p[p.part == 'val'].drop(columns='part') for p in parts]))
A = sum(v.reindex(V[0].index) for v in V) / 5
A.index.name = 'cookie_id'
A.to_parquet(R65 / 'cache' / 'resid_repavg.parquet')
subprocess.run([sys.executable, str(ROOT / 'research6' / 'evaluate_auxiliary.py'), str(R65 / 'cache' / 'resid_repavg.parquet'), 'cm65'],
               stdout=open(R65 / 'gate_cm65.log', 'w'), stderr=subprocess.DEVNULL, check=True)
for f in ('matched', 'conditional'):
    shutil.move(str(ROOT / 'research6' / f'gate_cm65_{f}.csv'), str(R65 / f'gate_cm65_{f}.csv'))
M = pd.read_csv(R65 / 'gate_cm65_matched.csv'); C = pd.read_csv(R65 / 'gate_cm65_conditional.csv')
M['AUC_best_dir'] = np.maximum(M.AUC_TP_vs_pFP, 1 - M.AUC_TP_vs_pFP)
print('\n== (2) persistent FP vs matched TP (AUC > 0.5 = TP higher), sorted by strength')
print(M.sort_values('AUC_best_dir', ascending=False).round(3).to_string(index=False))
cols = ['feature', 'AUC_target', 'cond_AUC_top_decile', 'band300_900_AUC', 'band_AUC_len_strat', 'band_AUC_web+mouse', 'band_AUC_web no mouse',
        'band_AUC_android', 'auc_vs_n_events', 'rho_B0', 'rho_mouse', 'rho_behavior', 'rho_item', 'rho_n', 'rho_residual_band']
C['band_best_dir'] = np.maximum(C.band_AUC_len_strat, 1 - C.band_AUC_len_strat)
print('\n== conditional analysis around the operating point (ranks 300-900), orthogonality, confounders')
print(C.sort_values('band_best_dir', ascending=False)[cols].round(3).to_string(index=False))

# ---------------- (3) temporal stability of the strongest features
from common4 import PFP
o = pd.read_csv(ROOT / 'research2' / 'oof_final_B0.csv').set_index('cookie_id'); o = o[o.oof_repeated_dayblock > 0]
o['rank'] = o.oof_repeated_dayblock.rank(ascending=False); D = o.join(A)
F = pd.read_parquet(ROOT / 'work' / 'features.parquet'); T = pd.read_parquet(ROOT / 'research3' / 'cache' / 'traits.parquet')
D['grp'] = np.where(T.pl.reindex(D.index) == 'web', np.where(F.ptr_n.reindex(D.index).fillna(0) > 0, 'wm', 'wn'), T.pl.reindex(D.index))
band = D['rank'].between(300, 900)
top = list(M[M.match == 'B0+platform+mouse'].sort_values('AUC_best_dir', ascending=False).feature[:4]) + \
      list(C.sort_values('band_best_dir', ascending=False).feature[:4])
print('\n== (3) temporal stability (week 1 = days 2-7, week 2 = days 8-13)')
for f in dict.fromkeys(top):
    r = {}
    for name, w in [('w1', D.day < 8), ('w2', D.day >= 8)]:
        k = band & w & D[f].notna(); r[f'band_{name}'] = round(roc_auc_score(D.target[k], D[f][k]), 3)
        fp = D.index[D.index.isin(PFP) & w & D[f].notna()]; tp = D[(D.target == 1) & w & D[f].notna()]
        m = []
        for c in fp:
            cand = tp[tp.grp == D.at[c, 'grp']]; m += list((cand['rank'] - D.at[c, 'rank']).abs().nsmallest(3).index)
        if len(fp) >= 5:
            r[f'pFP_vs_TP_{name}'] = round(roc_auc_score(np.r_[np.zeros(len(fp)), np.ones(len(m))], pd.concat([D.loc[fp, f], D.loc[m, f]])), 3)
            r[f'n_pFP_{name}'] = len(fp)
    print(f, r)

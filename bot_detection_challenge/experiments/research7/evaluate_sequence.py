"""C7.6-C7.8 cheap-stage evaluation: standalone local-model performance, window AUC by length, chunk-distribution
diagnostics (persistent FP vs matched TP, conditional band, consistency at same mean), order control."""
import sys, subprocess, shutil, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
R7 = Path(__file__).resolve().parent; ROOT = R7.parent
sys.path.insert(0, str(R7)); sys.path.insert(0, str(ROOT / 'research4'))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from common4 import IDS, Y, PFP, pr
from build_seq_features import aggregate, scheme_features, geom, LOCKED
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
SCH = ['P0', 'P1', 'P2', 'P3', 'P4', 'fwd7']
K = {s: range(6) for s in SCH[:5]}; K['fwd7'] = range(2)
lab = pd.Series(Y, index=IDS)
b0 = {s: pd.Series(np.load(ROOT / 'research2' / 'oof' / f'B0_cat_top120__{s}.npy'), index=IDS) for s in SCH}

# ---- 1. standalone local models (cookie level) + correlation with B0
rows = []
for col in ('rf', 'cat'):
    for s in SCH:
        F = scheme_features(s, K[s], col)
        if s == 'fwd7':   # later fold overwrites days 7..11, as in the harness
            F = F[~F.index.duplicated(keep='last')]
        y = lab[F.index]
        r = dict(model=col, scheme=s, n=len(F))
        for f in ('seq_full', 'chunk_mean', 'chunk_median', 'len5_mean'):
            x = F[f].fillna(F.seq_full)
            r[f'PR_{f}'] = pr(y.values, x.values); r[f'AUC_{f}'] = roc_auc_score(y, x)
        r['spearman_chunkmean_B0'] = F.chunk_mean.corr(b0[s][F.index], method='spearman')
        r['PR_B0_same_rows'] = pr(y.values, b0[s][F.index].values)
        rows.append(r)
S = pd.DataFrame(rows)
print('== 1. standalone cheap local models (cookie-level P@R>=0.70, ROC-AUC) and Spearman with B0')
print(S.round(4).to_string(index=False))
print('\nrep mean (P0-P4):'); print(S[S.scheme != 'fwd7'].groupby('model').mean(numeric_only=True).round(4).T)

# ---- 2. window-level discrimination by length (P0 validation windows)
G = geom()
wa = []
for col in ('rf', 'cat'):
    t = pd.concat([pd.read_parquet(R7 / 'oof' / f'winscores_P0_k{k}.parquet') for k in range(6)]); t = t[t.part == 'val']
    g = G.loc[t.index]; yy = lab[g.cookie_id].values
    for L in (3, 5, 10, 20, 40, -1):
        m = (g.L == L).values
        wa.append(dict(model=col, L='full' if L == -1 else L, n_windows=int(m.sum()), AUC=roc_auc_score(yy[m], t[col].values[m]),
                       PR_AUC=average_precision_score(yy[m], t[col].values[m])))
print('\n== 2. window-level discrimination by window length (P0 validation windows, label = cookie target)')
print(pd.DataFrame(wa).round(4).to_string(index=False))

# ---- 3. diagnostics on the averaged repeated OOF (CatBoost window model)
Fs = [scheme_features(s, K[s], 'cat') for s in SCH[:5]]
A = sum(f.reindex(Fs[0].index) for f in Fs) / 5
A.index.name = 'cookie_id'; A.to_parquet(R7 / 'cache' / 'seq_repavg_cat.parquet')
subprocess.run([sys.executable, str(ROOT / 'research6' / 'evaluate_auxiliary.py'), str(R7 / 'cache' / 'seq_repavg_cat.parquet'), 'seq7'],
               stdout=open(R7 / 'gate_seq7.log', 'w'), stderr=subprocess.DEVNULL, check=True)
for f in ('matched', 'conditional'):
    shutil.move(str(ROOT / 'research6' / f'gate_seq7_{f}.csv'), str(R7 / f'gate_seq7_{f}.csv'))
M = pd.read_csv(R7 / 'gate_seq7_matched.csv'); C = pd.read_csv(R7 / 'gate_seq7_conditional.csv')
M['best'] = np.maximum(M.AUC_TP_vs_pFP, 1 - M.AUC_TP_vs_pFP)
print('\n== 3a. persistent FP vs matched TP (AUC>0.5 = TP higher), length-matched variant, sorted')
print(M[M.match.str.contains('n_events')].sort_values('best', ascending=False).round(3).to_string(index=False))
print('\n== 3b. B0-only matching (top 8)'); print(M[~M.match.str.contains('n_events')].sort_values('best', ascending=False).head(8).round(3).to_string(index=False))
C['band_best'] = np.maximum(C.band_AUC_len_strat, 1 - C.band_AUC_len_strat)
print('\n== 3c. conditional band 300-900 (length-stratified), orthogonality')
print(C.sort_values('band_best', ascending=False)[['feature', 'AUC_target', 'band300_900_AUC', 'band_AUC_len_strat', 'band_AUC_web+mouse',
      'band_AUC_web no mouse', 'band_AUC_android', 'auc_vs_n_events', 'rho_B0', 'rho_n']].round(3).to_string(index=False))

# ---- 4. group table + consistency at same mean
o = pd.read_csv(ROOT / 'research2' / 'oof_final_B0.csv').set_index('cookie_id'); o = o[o.oof_repeated_dayblock > 0]
rk = o.oof_repeated_dayblock.rank(ascending=False)
grp = pd.Series('easy neg', index=o.index)
grp[(o.target == 0) & rk.between(605, 1500)] = 'hard neg'; grp[(o.target == 0) & (rk <= 604)] = 'other FP'
grp[o.index.isin(PFP)] = 'persistent FP'; grp[(o.target == 1) & (rk <= 604)] = 'TP'; grp[(o.target == 1) & (rk > 604)] = 'FN'
T = A.join(o.oof_repeated_dayblock.rename('B0')).join(grp.rename('grp'))
print('\n== 4a. medians by group')
print(T.groupby('grp')[['B0', 'seq_full', 'chunk_mean', 'chunk_std', 'chunk_q10', 'chunk_q90', 'chunk_min', 'chunk_max']].median().round(3))
from common4 import IDS as _I
F7 = pd.read_parquet(ROOT / 'work' / 'features.parquet'); Tr = pd.read_parquet(ROOT / 'research3' / 'cache' / 'traits.parquet')
T['grp2'] = np.where(Tr.pl.reindex(T.index) == 'web', np.where(F7.ptr_n.reindex(T.index).fillna(0) > 0, 'wm', 'wn'), Tr.pl.reindex(T.index))
T['rank'] = rk; T['n'] = F7.n_events.reindex(T.index)
fp = T[T.grp == 'persistent FP']; tp = T[(o.target.reindex(T.index) == 1)]
pairs = []
for c, r in fp.iterrows():
    cand = tp[tp.grp2 == r.grp2]
    d = (cand['rank'] - r['rank']).abs() / 100 + (cand.chunk_mean - r.chunk_mean).abs() * 20
    for j in d.nsmallest(3).index: pairs.append((c, j))
print('\n== 4b. consistency at the same mean (matched on B0 rank + chunk_mean + platform x mouse): AUC TP vs pFP')
for f in ('chunk_mean', 'chunk_std', 'chunk_q10', 'chunk_min', 'chunk_range', 'len5_std', 'position_std_L5', 'frac_gt07'):
    x = pd.concat([T.loc[[a for a, _ in pairs], f], T.loc[[b for _, b in pairs], f]]).values
    y = np.r_[np.zeros(len(pairs)), np.ones(len(pairs))]; m = ~np.isnan(x)
    print(f'  {f:16s} AUC={roc_auc_score(y[m], x[m]):.3f}  median pFP={np.nanmedian(x[:len(pairs)]):.3f}  median TP={np.nanmedian(x[len(pairs):]):.3f}')

# ---- 5. order control (shuffled event order inside cookie, same window positions, same outer models)
print('\n== 5. order control: original vs order-shuffled validation windows')
for s in ('P0', 'fwd7'):
    t = pd.concat([pd.read_parquet(R7 / 'oof' / f'winscores_{s}_k{k}.parquet') for k in K[s]]); t = t[t.part == 'val']
    t = t[~t.index.duplicated(keep='last')]
    for col in ('rf', 'cat'):
        Fo, Fs_ = aggregate(t[col]), aggregate(t[f'{col}_shuf'])
        Fo = Fo[~Fo.index.duplicated(keep='last')]; Fs_ = Fs_[~Fs_.index.duplicated(keep='last')]
        y = lab[Fo.index]
        gw = G.loc[t.index]; yy = lab[gw.cookie_id].values; m5 = (gw.L == 5).values
        print(f'  {s} {col}: chunk_mean P@R {pr(y.values, Fo.chunk_mean.fillna(Fo.seq_full).values):.4f} -> shuffled {pr(y.values, Fs_.chunk_mean.fillna(Fs_.seq_full).values):.4f} | '
              f'len-5 window AUC {roc_auc_score(yy[m5], t[col].values[m5]):.3f} -> {roc_auc_score(yy[m5], t[col + "_shuf"].values[m5]):.3f} | '
              f'corr(window score, shuffled) {np.corrcoef(t[col], t[col + "_shuf"])[0, 1]:.3f}')

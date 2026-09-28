"""C6.6 cheap-gate diagnostics for a cookie-level auxiliary feature table (label-free features, OOF by construction).
Usage: python research6/evaluate_auxiliary.py <parquet under research6/cache> [name]"""
import sys, json, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
R6 = Path(__file__).resolve().parent; ROOT = R6.parent
sys.path.insert(0, str(ROOT / 'research4'))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from common4 import load, SCHEMES, IDS, Y, DAY, PFP, rankpct
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)

path = sys.argv[1] if len(sys.argv) > 1 else 'aux_cookie.parquet'
A = pd.read_parquet(R6 / 'cache' / path)
FE = list(A.columns)
b0 = pd.read_csv(ROOT / 'research2' / 'oof_final_B0.csv').set_index('cookie_id')
b0 = b0[b0.oof_repeated_dayblock > 0]
Lx = [load(s) for s in SCHEMES[:5]]
ex = pd.DataFrame({e: np.nanmean([l[e] for l in Lx], axis=0) for e in ['mouse', 'behavior', 'item']}, index=IDS)
F = pd.read_parquet(ROOT / 'work' / 'features.parquet')
T = pd.read_parquet(ROOT / 'research3' / 'cache' / 'traits.parquet')
D = pd.DataFrame(index=b0.index)
D['y'] = b0.target; D['B0'] = b0.oof_repeated_dayblock; D['rank'] = D.B0.rank(ascending=False)
D['pl'] = T.pl.reindex(D.index); D['has_mouse'] = F.ptr_n.reindex(D.index).fillna(0) > 0
D['grp'] = np.where(D.pl == 'web', np.where(D.has_mouse, 'web+mouse', 'web no mouse'), D.pl)
D['n'] = F.n_events.reindex(D.index)
D = D.join(A).join(ex).join(F[['bigram_ent', 'dtbin_ent', 'trigram_nent', 'dtexact_uniq_ratio']])
pfp = D.index[D.index.isin(PFP)]


def auc(y, x):
    m = x.notna() & y.notna()
    return roc_auc_score(y[m], x[m]) if m.sum() > 10 and y[m].nunique() == 2 else np.nan


def matched(k=3, use_n=False):
    tp = D[D.y == 1]
    rows = []
    for c in pfp:
        r = D.loc[c]; cand = tp[(tp.grp == r.grp)]
        dist = (cand['rank'] - r['rank']).abs() / 100 + (np.abs(np.log1p(cand.n) - np.log1p(r.n)) * 2 if use_n else 0)
        for j in dist.nsmallest(k).index: rows.append((c, j))
    return rows


rng = np.random.RandomState(0)
out = []
for use_n in (False, True):
    pairs = matched(use_n=use_n)
    fp_ids = [a for a, _ in pairs]; tp_ids = [b for _, b in pairs]
    for f in FE:
        x = pd.concat([D.loc[fp_ids, f], D.loc[tp_ids, f]]).reset_index(drop=True)
        y = pd.Series(np.r_[np.zeros(len(fp_ids)), np.ones(len(tp_ids))])
        a = auc(y, x)
        # bootstrap over persistent FP (each FP resampled with its matches)
        grp = np.repeat(np.arange(len(pfp)), 3)
        bs = []
        for _ in range(500):
            s = rng.choice(len(pfp), len(pfp))
            idx = np.r_[np.concatenate([np.flatnonzero(grp == i) for i in s]), len(fp_ids) + np.concatenate([np.flatnonzero(grp == i) for i in s])]
            bs.append(auc(y.iloc[idx].reset_index(drop=True), x.iloc[idx].reset_index(drop=True)))
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        out.append(dict(feature=f, match='B0+platform+mouse' + ('+n_events' if use_n else ''), AUC_TP_vs_pFP=a, ci_lo=lo, ci_hi=hi,
                        med_pFP=D.loc[fp_ids, f].median(), med_TP=D.loc[tp_ids, f].median()))
M = pd.DataFrame(out)
print('== persistent FP (45) vs matched TP (3 per FP); AUC>0.5 means TP higher')
print(M.round(3).to_string(index=False))

rows = []
dec = pd.qcut(D.B0.rank(pct=True), 10, labels=False)
band = D['rank'].between(300, 900)
nb = pd.qcut(np.log1p(D.n), 5, labels=False, duplicates='drop')
for f in FE:
    r = dict(feature=f, AUC_target=auc(D.y, D[f]), coverage=D[f].notna().mean())
    ca = [auc(D.y[dec == k], D[f][dec == k]) for k in range(10)]
    r['cond_AUC_top3_deciles'] = np.nanmean(ca[7:]); r['cond_AUC_top_decile'] = ca[9]
    r['band300_900_AUC'] = auc(D.y[band], D[f][band])
    # length-stratified AUC inside the band (weighted by stratum size)
    st = [(auc(D.y[band & (nb == k)], D[f][band & (nb == k)]), (band & (nb == k)).sum()) for k in range(5)]
    st = [(a, w) for a, w in st if not np.isnan(a)]
    r['band_AUC_len_strat'] = np.average([a for a, _ in st], weights=[w for _, w in st]) if st else np.nan
    for g in ['web+mouse', 'web no mouse', 'android']:
        k = band & (D.grp == g); r[f'band_AUC_{g}'] = auc(D.y[k], D[f][k])
    r['auc_vs_n_events'] = auc((D.n > D.n.median()).astype(float), D[f])    # length dependence (0.5 = none)
    for c in ['B0', 'mouse', 'behavior', 'item', 'n', 'bigram_ent', 'dtbin_ent']:
        r[f'rho_{c}'] = D[[f, c]].corr(method='spearman').iloc[0, 1]
    # residual difficulty in the hard region (ranks 300-900): positives 1-B0, negatives B0
    hd = np.where(D.y == 1, 1 - D.B0.rank(pct=True), D.B0.rank(pct=True))
    r['rho_residual_band'] = pd.Series(hd, index=D.index)[band].corr(D[f][band], method='spearman')
    rows.append(r)
C = pd.DataFrame(rows)
print('\n== bot information, conditional on B0, length dependence, orthogonality')
print(C.round(3).T.to_string())
name = sys.argv[2] if len(sys.argv) > 2 else Path(path).stem
M.to_csv(R6 / f'gate_{name}_matched.csv', index=False); C.to_csv(R6 / f'gate_{name}_conditional.csv', index=False)

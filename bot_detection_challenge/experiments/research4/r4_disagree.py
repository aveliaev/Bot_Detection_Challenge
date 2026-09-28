"""C4 diagnostics: expert performance, correlations, scores of the 45 persistent FP vs TP near threshold,
disagreement features AUC (TPnear vs pFP), and the 'mouse HIGH, item HIGH, behavior LOW' regime.
All expert scores are OOF; per row they are averaged over the 5 repeated partitions (days >= 2)."""
import sys, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common4 import *
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 80)
EX = ['B0', 'mouse', 'behavior', 'item', 'context']
L = [load(s) for s in SCHEMES[:5]]
mask = ~np.isnan(L[0]['B0'])
P = pd.DataFrame({e: np.nanmean([l[e] for l in L], axis=0) for e in EX}, index=IDS)       # averaged OOF probability
Rk = pd.DataFrame({e: np.nanmean([rankpct(l[e], mask & ~np.isnan(l[e])) for l in L], axis=0) for e in EX}, index=IDS)  # averaged OOF rank
P, Rk = P[mask], Rk[mask]; y = pd.Series(Y, index=IDS)[mask]
T = pd.read_parquet(ROOT / 'research3' / 'cache' / 'traits.parquet'); pl = T.pl.reindex(P.index)
avail = P.mouse.notna()

# 1. expert performance
print('== 1-3. expert performance (averaged repeated OOF, days>=2)')
rows = []
for e in EX:
    s = Rk[e].fillna(-1)
    r = dict(expert=e, PR_all=pr(y, s), ROC_all=roc_auc_score(y, s), coverage=float(Rk[e].notna().mean()))
    a = Rk[e].notna(); r['ROC_available'] = roc_auc_score(y[a], Rk[e][a])
    for p in ['web', 'android', 'ios']:
        k = (pl == p) & a
        r[f'PR_{p}'] = pr(y[k], Rk[e][k]) if k.sum() and y[k].sum() else np.nan
    rows.append(r)
print(pd.DataFrame(rows).round(4).to_string(index=False))

# 4. correlations
print('\n== 4. Spearman correlation of expert OOF scores (all rows / mouse-available rows / positives only)')
for name, k in [('all', slice(None)), ('mouse available', avail), ('bots only, mouse avail', avail & (y == 1)), ('humans only, mouse avail', avail & (y == 0))]:
    print(f'-- {name}'); print(Rk[k][EX].corr(method='spearman').round(3))

# groups
b0rank = P.B0.rank(ascending=False)
grp = pd.Series('TN', index=P.index)
grp[(y == 1) & (b0rank <= 604)] = 'TP_top604'
grp[(y == 1) & b0rank.between(300, 604)] = 'TPnear'
grp[(y == 0) & (b0rank <= 604)] = 'FP_other'
grp[P.index.isin(PFP)] = 'pFP'
grp[(y == 1) & (b0rank > 604)] = 'FN'
print('\ngroups:', grp.value_counts().to_dict())

# 5-6. scores of persistent FP and matched TP
print('\n== 5. persistent FP: averaged OOF probability of each expert (sorted by B0)')
tab = P[grp == 'pFP'].assign(target=0, mouse_avail=avail[grp == 'pFP'].astype(int), platform=pl[grp == 'pFP']).sort_values('B0', ascending=False)
tab.index = [f'pFP_{i:02d}' for i in range(len(tab))]
print(tab.round(3).to_string())
print('\n== 6. medians by group (probability | rank percentile)')
print(pd.concat([P.groupby(grp).median().add_suffix('_p'), Rk.groupby(grp).median().add_suffix('_r')], axis=1).round(3).T)
# matched TP: nearest B0 score among TPs for each pFP
tps = P[grp.isin(['TP_top604'])]
pairs = []
for c in P.index[grp == 'pFP']:
    j = (tps.B0 - P.B0[c]).abs().idxmin()
    pairs.append({**{f'FP_{e}': P.loc[c, e] for e in EX}, **{f'TP_{e}': tps.loc[j, e] for e in EX}})
pairs = pd.DataFrame(pairs)
print('\nmatched pairs (each pFP vs TP with the closest B0 score): medians')
print(pd.DataFrame({e: [pairs[f'FP_{e}'].median(), pairs[f'TP_{e}'].median(), (pairs[f'TP_{e}'] > pairs[f'FP_{e}']).mean()] for e in EX},
                   index=['FP median', 'matched TP median', 'P(TP>FP)']).round(3))

# 7. disagreement features
print('\n== 7. disagreement features: AUC for TPnear (=1) vs pFP (=0) and TP_top604 vs pFP')
def feats(S):
    A = S[['mouse', 'behavior', 'item']].values
    m, b, i = A[:, 0], A[:, 1], A[:, 2]
    F = pd.DataFrame(index=S.index)
    F['min'] = np.nanmin(A, 1); F['max'] = np.nanmax(A, 1); F['mean'] = np.nanmean(A, 1); F['median'] = np.nanmedian(A, 1); F['std'] = np.nanstd(A, 1)
    F['mouse-behavior'] = m - b; F['mouse-item'] = m - i; F['behavior-item'] = b - i
    F['mouse*behavior'] = m * b; F['mouse*item'] = m * i; F['behavior*item'] = b * i; F['mouse*behavior*item'] = m * b * i
    for t in (.5, .7, .8, .9): F[f'count>{t}'] = (A > t).sum(1)
    F['behavior'] = b; F['mouse'] = m; F['item'] = i; F['context'] = S['context'].values; F['B0'] = S['B0'].values
    return F
out = []
for scale, S in (('prob', P), ('rank', Rk)):
    F = feats(S)
    for c in F.columns:
        for ref in ('TPnear', 'TP_top604'):
            k = grp.isin([ref, 'pFP']) & F[c].notna()
            t = (grp[k] == ref).astype(int)
            if t.nunique() < 2: continue
            a = roc_auc_score(t, F[c][k])
            out.append(dict(scale=scale, feature=c, ref=ref, n_pFP=int((t == 0).sum()), n_TP=int(t.sum()), AUC=a, AUC_best_dir=max(a, 1 - a)))
D = pd.DataFrame(out)
W = D.pivot_table(index=['feature'], columns=['scale', 'ref'], values='AUC').round(3)
print(W.to_string())
print('\nmax |AUC-0.5|+0.5 over all disagreement features, TPnear vs pFP:', D[(D.ref == 'TPnear') & ~D.feature.isin(['B0', 'context'])].AUC_best_dir.max().round(3))
D.to_csv(R4 / 'disagreement_auc.csv', index=False)

# 9. regime analysis: levels by expert rank percentile (label-free cut-points)
print('\n== 9. regime analysis (mouse-available rows). Levels: H = top 10% of the expert, M = 75-90th pct, L = below 75th')
lv = lambda s: pd.cut(s, [-1, .75, .90, 1.01], labels=['L', 'M', 'H'])
Q = pd.DataFrame({'mouse': lv(Rk.mouse), 'item': lv(Rk.item), 'behavior': lv(Rk.behavior)})[avail]
Q['y'] = y[avail]; Q['pFP'] = grp[avail] == 'pFP'; Q['TPnear'] = grp[avail] == 'TPnear'; Q['B0_flag'] = (b0rank <= 604)[avail]
cell = Q.groupby(['mouse', 'item', 'behavior'], observed=True).agg(n=('y', 'size'), bots=('y', 'sum'), P_bot=('y', 'mean'),
                                                                    pFP=('pFP', 'sum'), TPnear=('TPnear', 'sum'), B0_flagged=('B0_flag', 'sum'))
print(cell[cell.n >= 5].round(3).to_string())
for c in [('H', 'H', 'L'), ('H', 'H', 'M'), ('H', 'H', 'H')]:
    if c in cell.index:
        r = cell.loc[c]; print(f'mouse={c[0]} item={c[1]} behavior={c[2]}: n={int(r.n)} P(bot)={r.P_bot:.3f} pFP={int(r.pFP)} TPnear={int(r.TPnear)}')
# where do the pFP sit?
print('\npFP by (mouse,item,behavior) level (mouse-available pFP only):'); print(Q[Q.pFP].groupby(['mouse', 'item', 'behavior'], observed=True).size())
print('pFP without mouse:', int(((grp == 'pFP') & ~avail).sum()))
# no-mouse regime: item HIGH, behavior LOW
Qn = pd.DataFrame({'item': lv(Rk.item), 'behavior': lv(Rk.behavior)})[~avail]; Qn['y'] = y[~avail]; Qn['pFP'] = grp[~avail] == 'pFP'
print('\nno-mouse rows:'); print(Qn.groupby(['item', 'behavior'], observed=True).agg(n=('y', 'size'), P_bot=('y', 'mean'), pFP=('pFP', 'sum')).round(3))
tab.to_csv(R4 / 'persistent_fp_scores.csv')

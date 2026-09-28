import sys, warnings, pickle; sys.path.insert(0, 'research3'); warnings.filterwarnings('ignore')
from common import *
from rep import build, block_cols
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
R = build(); m = meta(); o = pd.read_parquet(CACHE / 'groups_opt.parquet')
pos = m[m.target == 1].index
cols = block_cols(R, ['ev', 'tr', 'dt', 'px', 'py', 'pm', 'cat', 'pg', 'ua', 'pl', 'm'])
sc = StandardScaler().fit(R.loc[pos, cols]); pca = PCA(15, random_state=0).fit(sc.transform(R.loc[pos, cols]))
Z = lambda idx: pca.transform(sc.transform(R.loc[idx, cols]))
km = KMeans(6, n_init=20, random_state=0).fit(Z(pos))
lab = pd.Series(km.labels_, index=pos)
T = pd.read_parquet(CACHE / 'traits.parquet')
prof = pd.DataFrame({'n': lab.value_counts(), 'web': R.loc[pos, 'pl_web'].groupby(lab).mean(), 'android': R.loc[pos, 'pl_android'].groupby(lab).mean(),
                     'ptr_frac': R.loc[pos, 'm_ptrfrac'].groupby(lab).mean(), 'ptr_comp': T.loc[pos, 'ptr_comp'].groupby(lab).mean(),
                     'edge': R.loc[pos, 'm_edge'].groupby(lab).mean(), 'fast': T.loc[pos, 'fast'].groupby(lab).mean(),
                     'deep': T.loc[pos, 'deep'].groupby(lab).mean(), 'scr_ua': T.loc[pos, 'scr_ua'].groupby(lab).mean(),
                     'pool': T.loc[pos, 'pool'].groupby(lab).mean(), 'young': T.loc[pos, 'young'].groupby(lab).mean()})
caught = o.reindex(pos).grp.isin(['TP', 'B', 'C'])
prof['recall_B0'] = caught.groupby(lab).mean()
prof['week2_share'] = (m.loc[pos, 'day'] >= 7).groupby(lab).mean()
print(prof.round(2))
# assign FP and ordinary negatives to prototypes; distance relative to within-cluster distances of members
dpos = km.transform(Z(pos)).min(1)
for g in ['A', 'D']:
    ids = o[o.grp == g].index
    d = km.transform(Z(ids)); a = d.argmin(1)
    print(g, 'nearest-prototype distribution', np.bincount(a, minlength=6), ' median min-dist', round(np.median(d.min(1)), 2), ' (bots own median', round(np.median(dpos), 2), ')')
# percentile of FP min-dist within bot distribution (in-sample for bots -> optimistic for bots)
ids = o[o.grp == 'A'].index; dA = km.transform(Z(ids)).min(1)
print('FP min-dist percentile among bot min-dists: median', np.median([(dpos < x).mean() for x in dA]).round(2))

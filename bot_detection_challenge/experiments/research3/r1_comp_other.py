"""Within compressed-pointer cookies: which non-pointer raw aspects differ t0 vs t1 (descriptive, all labels)."""
import sys; sys.path.insert(0, 'research3')
from common import *
from sklearn.metrics import roc_auc_score
from cv import Data
from history import pool_features
pd.set_option('display.width', 250)
K = pd.read_parquet(CACHE / 'ptr_compressed.parquet')
D = Data(); F = D.F
P = pool_features(D.ci, D.labels, D.ids, 7)
X = F.loc[K.index].join(P)
t = K.target.values
rows = []
for c in X.columns:
    v = X[c].astype(float)
    if v.notna().sum() < 50 or v.nunique() < 2: continue
    a = roc_auc_score(t, v.fillna(v.median()))
    rows.append((c, a, v[t == 1].median(), v[t == 0].median()))
R = pd.DataFrame(rows, columns=['f', 'auc', 'med_t1', 'med_t0']); R['s'] = (R.auc - .5).abs()
print(R.sort_values('s', ascending=False).head(40).round(3).to_string(index=False))
ev = events(); e = ev[ev.cookie_id.isin(K.index)].join(K[['target']], on='cookie_id')
d = e[e['dt'].notna() & (e['dt'] <= 1800)]
print('\ndt quantiles by target'); print(d.groupby('target')['dt'].quantile([.1, .25, .5, .75, .9]).unstack())
print('\ndt band shares'); print(pd.crosstab(d.target, pd.cut(d['dt'], [-1, 2, 5, 10, 20, 30, 60, 90, 180, 1800]), normalize='index').round(3))
print('\nage at first event (d)'); print(e.groupby(['target', 'cookie_id']).age_h.min().div(24).groupby(level=0).quantile([.1, .25, .5, .75, .9]).unstack().round(1))
print('\nday distribution'); print(pd.crosstab(K.target, K.day))

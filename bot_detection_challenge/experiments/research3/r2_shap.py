"""C3.2/24 SHAP reasons for persistent FP (P0 fold models), grouped by family; cluster FP by reason."""
import sys, json, warnings; sys.path.insert(0, 'research3'); sys.path.insert(0, 'research2'); warnings.filterwarnings('ignore')
from common import *
from rcv import Harness, BASE_COLS
from cv import make_model
from catboost import Pool
from sklearn.cluster import KMeans
pd.set_option('display.width', 250)
H = Harness(n_rep=1); D = H.D
pfp = set(json.load(open(CACHE / 'persistent.json'))); o = pd.read_parquet(CACHE / 'groups_opt.parquet')
def fam(c):
    if c.startswith('pool_'): return 'pool'
    if c.startswith('ptr') or c == 'is_web': return 'pointer'
    if c.startswith(('dt_', 'sdt_', 'dtbin', 'dtexact', 'ddt_', 'dwell_')): return 'timing'
    if c.startswith(('cookie_age', 'created', 'log_cookie')): return 'cookie_age'
    if c.startswith(('loc_', 'cat_', 'item', 'n_item', 'sel_')): return 'items/diversity'
    if c.startswith(('page', 'query', 'q_', 'n_quer', 'same_query', 'n_pages')): return 'search'
    if c.startswith(('ua_', 'pl_')): return 'ua/platform'
    if c.startswith(('hour', 'frac_night', 'frac_day', 'frac_eve', 'sec')): return 'time_of_day'
    return 'activity/events'
rows = []
for k, (tri, vai) in enumerate(H.schemes['P0']):
    Xtr = H.X(BASE_COLS, ('pool',), 'P0', k, tri); Xva = H.X(BASE_COLS, ('pool',), 'P0', k, vai)
    mdl = make_model('cat', dict(thread_count=2), 0).fit(Xtr, D.y[tri])
    sv = mdl.get_feature_importance(Pool(Xva), type='ShapValues')[:, :-1]
    S = pd.DataFrame(sv, index=Xva.index, columns=Xva.columns)
    rows.append(S)
S = pd.concat(rows)
famS = S.T.groupby([fam(c) for c in S.columns]).sum().T
g = pd.Series('other', index=S.index)
g[S.index.isin(pfp)] = 'persistentFP'
near = o.index[(o.target == 1) & (o['rank'].between(300, 604))]
g[S.index.isin(near)] = 'TP_rank300-604'
g[S.index.isin(o.index[(o.target == 1) & (o['rank'] < 300)])] = 'TP_top300'
print('mean family SHAP (log-odds contribution) by group'); print(famS.groupby(g).mean().round(2).T)
F = famS[g == 'persistentFP']
Zc = F.div(F.abs().sum(axis=1), axis=0)
for kk in (2, 3, 4):
    km = KMeans(kk, n_init=20, random_state=0).fit(Zc)
    print(f'\nFP reason clusters k={kk}: sizes', np.bincount(km.labels_)); print(F.groupby(km.labels_).mean().round(2).T)
km = KMeans(3, n_init=20, random_state=0).fit(Zc)
pd.Series(km.labels_, index=F.index).to_frame('reason').to_parquet(CACHE / 'fp_reason.parquet')
top = S[g == 'persistentFP'].mean().sort_values(ascending=False)
print('\ntop individual SHAP features pushing persistent FP up'); print(top.head(15).round(3).to_string())
famS.to_parquet(CACHE / 'shap_fam_P0.parquet')

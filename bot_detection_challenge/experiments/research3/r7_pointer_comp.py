import sys; sys.path.insert(0, 'research3')
from common import *
from sklearn.metrics import roc_auc_score
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60)
ev = events(); m = meta()
C = pd.read_parquet(CACHE / 'ptr_cookie.parquet')
C = C.drop(columns=['target', 'day']).join(m[['target', 'day']])
C8 = C[(C.n >= 6)].copy()
C8['comp'] = (C8.sx < 300) & (C8.sy < 250)
K = C8[C8.comp & C8.target.notna()].copy()
print('compressed labelled cookies:', K.target.value_counts().to_dict(), ' test compressed:', int((C8.comp & C8.target.isna()).sum()))
# missing pointer pattern on web: which events lack pointer
w = ev[ev.pl == 'web'].copy(); w['hasp'] = w.pointer_x.notna()
w['edge'] = w.pointer_x.isin([0, 1920]) | w.pointer_y.isin([0, 1080])
mp = w.groupby('cookie_id').agg(ptr_frac=('hasp', 'mean'), edge_n=('edge', 'sum'), nweb=('hasp', 'size'))
mp['miss_srv'] = w[w.event_name == 'search_results_view'].groupby('cookie_id').hasp.apply(lambda s: 1 - s.mean())
mp['miss_iv'] = w[w.event_name == 'item_view'].groupby('cookie_id').hasp.apply(lambda s: 1 - s.mean())
K = K.join(mp)
K['has_edge'] = (K.edge_n > 0).astype(int)
cols = ['n', 'sx', 'sy', 'madx', 'mady', 'rngx', 'rngy', 'aniso', 'corr', 'cx', 'cy', 'kurt_x', 'kurt_y', 'edge_frac', 'has_edge', 'ptr_frac', 'miss_srv', 'miss_iv']
print(K.groupby('target')[cols].median().round(3).T)
for c in cols:
    v = K[c].fillna(-9)
    print(f'{c:10s} AUC(t1 vs t0 | compressed) = {roc_auc_score(K.target, v):.3f}')
print('\nptr_frac distribution by target'); print(K.groupby('target').ptr_frac.describe().round(3))
# per-event: gaussian vs uniform within box? standardized residual kurtosis pooled
q = ev[ev.cookie_id.isin(K.index) & ev.pointer_x.notna() & ~ev.pointer_x.isin([0, 1920]) & ~ev.pointer_y.isin([0, 1080])].copy()
q = q.join(K[['target', 'cx', 'cy', 'sx', 'sy']], on='cookie_id')
q['zx'] = (q.pointer_x - q.cx) / q.sx; q['zy'] = (q.pointer_y - q.cy) / q.sy
for t in (0, 1):
    z = q[q.target == t]
    print('target', t, 'pooled kurt zx', round(z.zx.kurt(), 3), 'zy', round(z.zy.kurt(), 3), 'max|z| q99', round(z.zx.abs().quantile(.99), 2))
K.to_parquet(CACHE / 'ptr_compressed.parquet')

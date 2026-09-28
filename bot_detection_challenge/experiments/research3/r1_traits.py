"""Latent 'bot trait' structure: binary traits per cookie, rates by class, co-occurrence, trait counts."""
import sys; sys.path.insert(0, 'research3')
from common import *
from cv import Data
from history import pool_features
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
D = Data(); F = D.F.loc[D.ids]
P = pool_features(D.ci, D.labels, D.ids, 7)   # descriptive only (uses all train labels, earlier days)
X = F.join(P)
o = pd.read_parquet(CACHE / 'groups_opt.parquet')
C = pd.read_parquet(CACHE / 'ptr_cookie.parquet')
y = pd.Series(D.y, index=D.ids)
T = pd.DataFrame(index=D.ids)
comp = (C.n >= 6) & (C.sx < 300) & (C.sy < 250)
T['ptr_comp'] = comp.reindex(D.ids).fillna(False).astype(int)
T['web_noptr'] = X.ptr_none_web.fillna(0).astype(int)
T['fast'] = (X.sdt_median < 12).astype(int)
T['pool'] = (X.pool_nb_ge1_frac > 0.3).astype(int)
T['scr_ua'] = ((X.ua_is_lib > 0) | (X.ua_headless > 0)).astype(int)
T['young'] = (X.cookie_age_d < 3).astype(int)
T['deep'] = (X.page_max > 10).astype(int)
T['edge'] = (X.ptr_exact_edge_frac > 0).astype(int)
T['app'] = X.ua_app.astype(int)
T['pl'] = np.select([X.pl_web == 1, X.pl_android == 1], ['web', 'android'], 'ios')
tr = [c for c in T.columns if c != 'pl']
print('trait rates: P(trait|bot), P(trait|human), lift')
R = pd.DataFrame({'bot': T[y == 1][tr].mean(), 'human': T[y == 0][tr].mean()})
R['lift'] = R.bot / R.human
print(R.round(3))
T['grp'] = o.grp.reindex(D.ids).fillna('d01')
print('\ntrait rates by forensic group'); print(T.groupby('grp')[tr].mean().round(2))
# co-occurrence among bots: correlation of traits (web only)
wb = T[(y == 1) & (T.pl == 'web')][tr]
print('\ntrait correlation among WEB bots'); print(wb.corr().round(2))
wh = T[(y == 0) & (T.pl == 'web')][tr]
print('\ntrait correlation among WEB humans'); print(wh.corr().round(2))
# bot 'mode' traits count (web): ptr_comp|web_noptr, fast, pool, young, deep
T['k'] = (T.ptr_comp | T.web_noptr) + T.fast + T.pool + T.young + T.deep + T.scr_ua
print('\ntrait count distribution (web)'); print(pd.crosstab(T[T.pl == 'web'].k, y[T.pl == 'web']))
print('\ntrait count by group (web)'); print(pd.crosstab(T[T.pl == 'web'].grp, T[T.pl == 'web'].k))
T.to_parquet(CACHE / 'traits.parquet')

"""C3.7 pointer generator forensics (per cookie): what mechanism produces the coordinates?"""
import sys; sys.path.insert(0, 'research3')
from common import *
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60)
ev = events(); m = meta(); o = pd.read_parquet(CACHE / 'groups_opt.parquet')
p = ev[ev.pointer_x.notna()].copy()
p = p.join(o[['grp']], on='cookie_id'); p['grp'] = p.grp.fillna('T')  # T = test / day<2
x, y = p.pointer_x, p.pointer_y
p['edge'] = (x == 0) | (y == 0) | (x == 1920) | (y == 1080)
p['xi'] = (x % 1 == 0); p['x5'] = (x % 5 == 0) & (y % 5 == 0); p['x10'] = (x % 10 == 0) & (y % 10 == 0)
print('fraction integer coords', p.xi.mean(), ' mod5', p.x5.mean(), ' mod10', p.x10.mean())
print('edge value combos (top):'); print(p[p.edge].groupby('grp').apply(lambda d: ((d.pointer_x.isin([0, 1920])).map({True: 'Xedge', False: 'x'}) + '_' + (d.pointer_y.isin([0, 1080])).map({True: 'Yedge', False: 'y'})).value_counts(normalize=True).round(2).to_dict()))
q = p[~p.edge]
g = q.groupby('cookie_id')
C = pd.DataFrame({'n': g.size(), 'cx': g.pointer_x.median(), 'cy': g.pointer_y.median(),
                  'sx': g.pointer_x.std(), 'sy': g.pointer_y.std(),
                  'madx': g.pointer_x.agg(lambda v: np.median(np.abs(v - np.median(v)))),
                  'mady': g.pointer_y.agg(lambda v: np.median(np.abs(v - np.median(v)))),
                  'rngx': g.pointer_x.agg(lambda v: v.max() - v.min()), 'rngy': g.pointer_y.agg(lambda v: v.max() - v.min()),
                  'corr': g.apply(lambda d: d.pointer_x.corr(d.pointer_y) if len(d) > 3 else np.nan)})
C['edge_frac'] = p.groupby('cookie_id').edge.mean()
C['n_all'] = p.groupby('cookie_id').size()
C['aniso'] = C.sx / C.sy
C['kurt_x'] = g.pointer_x.apply(lambda v: v.kurt() if len(v) > 4 else np.nan)
C['kurt_y'] = g.pointer_y.apply(lambda v: v.kurt() if len(v) > 4 else np.nan)
C = C.join(o[['grp', 'target']]).join(m[['day']])
C['grp'] = C.grp.fillna('T')
C.to_parquet(CACHE / 'ptr_cookie.parquet')
C8 = C[C.n >= 8]
print(C8.groupby('grp')[['n', 'cx', 'cy', 'sx', 'sy', 'madx', 'mady', 'aniso', 'corr', 'edge_frac', 'kurt_x', 'kurt_y']].median().round(2))
print('\nquantiles of sx by group'); print(C8.groupby('grp').sx.quantile([.05, .25, .5, .75, .95]).unstack().round(0))
print('\nquantiles of sy by group'); print(C8.groupby('grp').sy.quantile([.05, .25, .5, .75, .95]).unstack().round(0))
print('\ncenter cx quantiles'); print(C8.groupby('grp').cx.quantile([.05, .25, .5, .75, .95]).unstack().round(0))
print('\ncenter cy quantiles'); print(C8.groupby('grp').cy.quantile([.05, .25, .5, .75, .95]).unstack().round(0))
# within-cookie distribution shape: uniform box vs gaussian? compare kurtosis (uniform=-1.2, gaussian=0)
print('\nkurtosis x quantiles'); print(C8.groupby('grp').kurt_x.quantile([.1, .5, .9]).unstack().round(2))
# compressed = sx<300
C8 = C8.assign(comp=(C8.sx < 300) & (C8.sy < 250))
print('\ncompressed share by group'); print(C8.groupby('grp').comp.mean().round(3))
print('\namong compressed: stats by group'); print(C8[C8.comp].groupby('grp')[['sx', 'sy', 'aniso', 'corr', 'cx', 'cy', 'edge_frac', 'kurt_x', 'kurt_y']].median().round(2))
print('counts compressed', C8[C8.comp].groupby('grp').size().to_dict())

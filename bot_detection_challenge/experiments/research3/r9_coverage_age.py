"""C3.9 coverage/crawling strategy + C3.13 cookie-creation behaviour + C3.10 exact fingerprints: pFP vs TP groups."""
import sys, json; sys.path.insert(0, 'research3')
from common import *
from sklearn.metrics import roc_auc_score
pd.set_option('display.width', 250)
ev = events(); m = meta(); o = pd.read_parquet(CACHE / 'groups_opt.parquet')
pfp = json.load(open(CACHE / 'persistent.json'))
grp = pd.Series(np.nan, index=m.index, dtype=object); grp[o.index] = o.grp
grp[pfp] = 'pFP'; grp[(grp == 'A')] = 'FPother'
grp[(grp.isin(['B', 'C'])) & o['rank'].reindex(m.index).between(300, 604)] = 'TPnear'
grp[(grp.isin(['B', 'C']))] = 'TPtop'
e = ev.join(grp.rename('g'), on='cookie_id'); e = e[e.g.notna()]
g = e.groupby('cookie_id')
# --- coverage: cells (category, location), (category, page), (location, page)
s = e[e.event_name == 'search_results_view']
def cov(d, a, b):
    c = d[a].astype(str) + '|' + d[b].astype(str)
    return c.groupby(d.cookie_id).nunique() / d.groupby('cookie_id').size()
C = pd.DataFrame({'cov_cat_loc': cov(e, 'item_category', 'item_location'),
                  'cov_q_page': cov(s, 'search_query', 'search_page'),
                  'cov_loc_page': cov(s, 'item_location', 'search_page')})
# page monotonicity within same query: fraction of consecutive same-query searches with page +1
s2 = s.sort_values(['cookie_id', 't'])
same = s2.cookie_id.eq(s2.cookie_id.shift()) & s2.search_query.eq(s2.search_query.shift())
dp = (s2.search_page - s2.search_page.shift()).where(same)
C['page_inc1'] = (dp == 1).groupby(s2.cookie_id).sum() / same.groupby(s2.cookie_id).sum().replace(0, np.nan)
C['page_backtrack'] = (dp < 0).groupby(s2.cookie_id).sum() / same.groupby(s2.cookie_id).sum().replace(0, np.nan)
# revisit rate of (query,page)
qp = s2.search_query.astype(str) + '|' + s2.search_page.astype(str)
C['qp_revisit'] = 1 - qp.groupby(s2.cookie_id).nunique() / s2.groupby('cookie_id').size()
# --- cookie creation
first = g.t.min(); cr = (m.cookie_created_at - DAY0).dt.total_seconds()
C['create_to_first_h'] = (first - cr.reindex(first.index)) / 3600
C['created_same_day'] = (C.create_to_first_h < 24).astype(int)
C['created_sec'] = m.cookie_created_at.dt.second.reindex(C.index)
C['created_hour'] = m.cookie_created_at.dt.hour.reindex(C.index)
C['create_to_first_mod_h'] = (C.create_to_first_h % 1)
# --- exact fingerprints: repeated exact pointer coords/dt sequences shared with other train positives
p = e[e.pointer_x.notna()]
xy = p.pointer_x.astype(int).astype(str) + ',' + p.pointer_y.astype(int).astype(str)
C['ptr_xy_uniq'] = xy.groupby(p.cookie_id).nunique() / p.groupby('cookie_id').size()
C['g'] = grp.reindex(C.index)
cols = [c for c in C.columns if c != 'g']
print(C.groupby('g')[cols].median().round(3).T)
k = C[C.g.isin(['pFP', 'TPnear'])]
print('\nAUC TPnear vs pFP (descriptive):')
for c in cols:
    v = k[c].fillna(k[c].median()); print(f'  {c:22s} {roc_auc_score(k.g == "TPnear", v):.3f}')
print('\ncreated hour distribution share (pFP vs TPnear vs TPtop vs D)'); print(pd.crosstab(C.g, pd.cut(C.created_hour, [-1, 5, 11, 17, 23]), normalize='index').round(2))

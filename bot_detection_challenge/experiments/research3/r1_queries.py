import sys; sys.path.insert(0, 'research3')
from common import *
pd.set_option('display.width', 250)
ev = events(); m = meta(); o = pd.read_parquet(CACHE / 'groups_opt.parquet')
e = ev.join(m[['target']], on='cookie_id')
s = e[e.event_name == 'search_results_view'].copy()
print('n unique queries', s.search_query.nunique())
# query -> category mapping: is it deterministic?
qc = s.groupby('search_query').item_category.agg(lambda x: x.value_counts(normalize=True).iloc[0] if x.notna().any() else np.nan)
print('query->top category share: median', qc.median(), 'q10', qc.quantile(.1))
# per event: category consistent with query's majority category?
maj = s.groupby('search_query').item_category.agg(lambda x: x.mode().iloc[0] if x.notna().any() else None)
s['cons'] = s.item_category == s.search_query.map(maj)
s = s.join(o[['grp']], on='cookie_id')
print('query-category consistency by group'); print(s.groupby('grp').cons.mean().round(3)); print('by target', s.groupby('target').cons.mean().round(3))
# query bot-rate stability across weeks (train only), cookie-level
tq = s[s.target.notna()].drop_duplicates(['cookie_id', 'search_query'])
tq['wk'] = (tq.day >= 7).astype(int)
r = tq.groupby(['search_query', 'wk']).target.agg(['mean', 'size']).unstack()
r = r[(r['size'][0] >= 15) & (r['size'][1] >= 15)]
print('query bot-rate week1 vs week2 corr (n=%d):' % len(r), np.corrcoef(r['mean'][0], r['mean'][1])[0, 1])
# category-by-day for bots: do bots focus on different categories on different days?
b = e[(e.target == 1) & e.item_category.notna()]
ct = pd.crosstab(b.day, b.item_category, normalize='index')
print('bot category share by day (std across days / mean):'); print((ct.std() / ct.mean()).sort_values().round(2).to_dict())
h = e[(e.target == 0) & e.item_category.notna()]
ct0 = pd.crosstab(h.day, h.item_category, normalize='index')
print('human category share by day (cv):'); print((ct0.std() / ct0.mean()).sort_values().round(2).to_dict())

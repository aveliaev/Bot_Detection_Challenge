"""C3.14 item-pool usage forensics: item-id structure, item 'age' at view, lag to earlier bot views, revisit."""
import sys, json; sys.path.insert(0, 'research3')
from common import *
pd.set_option('display.width', 250)
ev = events(); m = meta(); o = pd.read_parquet(CACHE / 'groups_opt.parquet')
pfp = json.load(open(CACHE / 'persistent.json'))
grp = pd.Series('d01', index=m.index); grp[m.target.isna()] = 'T'
grp[o.index] = o.grp.replace({'A': 'FPop'}); grp[pfp] = 'pFP'
it = ev[ev.item_id.notna()][['cookie_id', 'item_id', 't', 'day', 'event_name']].copy()
it['grp'] = it.cookie_id.map(grp); it['y'] = it.cookie_id.map(m.target)
print('item_id quantiles by group'); print(it.groupby('grp').item_id.quantile([.01, .25, .5, .75, .99]).unstack().round(0))
# item first appearance in whole log (any cookie, unlabelled) -> 'item age' at this view (days); strictly uses past events only
first = it.groupby('item_id').t.transform('min')
it['item_age_d'] = (it.t - first) / 86400
print('\nitem age at view (days since first seen anywhere)'); print(it.groupby('grp').item_age_d.quantile([.1, .25, .5, .75, .9]).unstack().round(2))
# lag to the most recent earlier view of same item by a labelled bot of an EARLIER day (label-available at test time)
b = it[(it.y == 1)][['item_id', 't', 'day']].rename(columns={'t': 'tb', 'day': 'db'})
x = it[['cookie_id', 'item_id', 't', 'day', 'grp']].reset_index().merge(b, on='item_id')
x = x[x.db < x.day]
lag = x.groupby('index').apply(lambda d: (d.t - d.tb).min() / 86400)
it['lag_bot_d'] = lag
print('\nlag since last earlier-day bot view (days), items with such a view'); print(it.groupby('grp').lag_bot_d.quantile([.1, .25, .5, .75, .9]).unstack().round(2))
print('share of item views with an earlier-day bot view'); print(it.groupby('grp').lag_bot_d.apply(lambda s: s.notna().mean()).round(3))
# number of distinct earlier bots per item
nb = x.groupby('index').apply(lambda d: d.db.nunique())
it['n_bot_days'] = nb
print('\ndistinct earlier bot-days per pool item'); print(it[it.lag_bot_d.notna()].groupby('grp').n_bot_days.describe()[['mean', '50%', '75%']].round(2))
it.to_parquet(CACHE / 'item_views.parquet')

"""C3.1 raw-event forensics: group-level comparison of raw fields (not model features)."""
import sys; sys.path.insert(0, 'research3')
from common import *
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 50); pd.set_option('display.max_rows', 300)
ev = events(); m = meta()
o = groups()
# B = TP in the same rank region as FP (ranks spanned by FP); C = easy TP rank<=200
fpr = o[o.grp == 'FP']['rank']
lo, hi = fpr.quantile(.1), 604
o.loc[(o.grp == 'TP') & (o['rank'] >= lo), 'grp'] = 'B'
o.loc[(o.grp == 'TP') & (o['rank'] < lo), 'grp'] = 'C'
o.loc[o.grp == 'FP', 'grp'] = 'A'
print('rank lo', lo, o.grp.value_counts().to_dict())
e = ev.merge(o[['grp', 'rank']], left_on='cookie_id', right_index=True)
ck = e.groupby('cookie_id').agg(grp=('grp', 'first'), pl=('pl', 'first'), n=('t', 'size'),
                                age_d=('age_h', lambda x: x.min() / 24), ua=('user_agent', 'first'),
                                nua=('user_agent', 'nunique'))
print('\n== platform share by group'); print(pd.crosstab(ck.grp, ck.pl, normalize='index').round(3))
print('\n== n events / cookie age (median) by group'); print(ck.groupby('grp')[['n', 'age_d', 'nua']].median())
print('\n== age quantiles'); print(ck.groupby('grp').age_d.quantile([.1, .25, .5, .75, .9]).unstack().round(1))
print('\n== event share by group'); print(pd.crosstab(e.grp, e.event_name, normalize='index').round(3))
def ua_family(u):
    u = u or ''
    for k in ['python-requests', 'curl', 'Scrapy', 'Go-http', 'node-fetch', 'axios', 'Java/', 'wget', 'HeadlessChrome', 'Avito/', 'YaBrowser', 'Firefox', 'Edg/', 'OPR/', 'Chrome', 'Safari']:
        if k in u: return k
    return 'other'
ck['uaf'] = ck.ua.map(ua_family)
print('\n== UA family by group'); print(pd.crosstab(ck.uaf, ck.grp, normalize='columns').round(3))
print('\n== top UA strings in A (FP) and their share in B/C/D')
uas = ck[ck.grp == 'A'].ua.value_counts().head(15)
for u, c in uas.items():
    print(c, {g: int(((ck.grp == g) & (ck.ua == u)).sum()) for g in 'BCDE'}, u[:110])
for col in ['item_category', 'item_location', 'seller_type']:
    t = pd.crosstab(e[col].fillna('NA'), e.grp, normalize='columns')
    t['A/B'] = (t['A'] + 1e-3) / (t['B'] + 1e-3)
    print(f'\n== {col}'); print(t.sort_values('A', ascending=False).head(25).round(3))
s = e[e.event_name == 'search_results_view']
print('\n== search_page distribution'); print(pd.crosstab(s.search_page.clip(upper=15), s.grp, normalize='columns').round(3))
print('\n== top queries in A vs B')
for g in 'AB':
    print(g, s[s.grp == g].search_query.value_counts().head(15).to_dict())
p = e[e.pointer_x.notna()]
print('\n== pointer frac with pointer (web only)'); w = e[e.pl == 'web']; print(w.groupby('grp').pointer_x.apply(lambda x: x.notna().mean()).round(3))
print('\n== pointer x/y quantiles'); print(p.groupby('grp')[['pointer_x', 'pointer_y']].quantile([.05, .25, .5, .75, .95]).unstack().round(0))
print('\n== dt (in-session <1800) quantiles'); d = e[(e.dt.notna()) & (e.dt <= 1800)]
print(d.groupby('grp').dt.quantile([.1, .25, .5, .75, .9]).unstack())
print('\n== hour of day share'); e['hr'] = ((e.t % 86400) // 3600).astype(int)
print(pd.crosstab(e.grp, pd.cut(e.hr, [-1, 5, 9, 13, 17, 21, 24]), normalize='index').round(3))
ck.to_parquet(CACHE / 'ck_groups.parquet')
o.to_parquet(CACHE / 'groups_opt.parquet')

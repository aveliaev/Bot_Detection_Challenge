"""C3.12 Android service fingerprints: UA device / OS / app version; week-1 vs week-2 stability of bot rates."""
import sys, json; sys.path.insert(0, 'research3')
from common import *
from sklearn.metrics import roc_auc_score
pd.set_option('display.width', 250)
ev = events(); m = meta(); o = pd.read_parquet(CACHE / 'groups_opt.parquet')
a = ev[ev.pl == 'android'].groupby('cookie_id').agg(ua=('user_agent', 'first'), n=('t', 'size'), nua=('user_agent', 'nunique'))
a = a.join(m[['target', 'day']])
u = a.ua.fillna('')
a['kind'] = np.select([u.str.startswith('Avito/'), u.str.contains('Mozilla')], ['app', 'browser'], 'other')
a['device'] = u.str.extract(r'Android \d+; ([^;)]+)')[0].fillna(u.str.extract(r'\(([^;)]*);')[0])
a['osv'] = u.str.extract(r'Android (\d+)')[0]
a['appv'] = u.str.extract(r'^Avito/([\d.]+)')[0]
print('android UA kinds by target'); print(pd.crosstab(a.kind, a.target, margins=True))
print('\nexample UAs:'); print(u.drop_duplicates().sample(8, random_state=0).tolist())
L = a[a.target.notna()]
for col in ['kind', 'device', 'osv', 'appv']:
    t = L.groupby([col, L.day >= 7]).target.agg(['mean', 'size']).unstack()
    t = t[(t['size'] >= 10).all(axis=1)]
    if len(t) > 2:
        r = np.corrcoef(t['mean'][False], t['mean'][True])[0, 1]
        print(f'\n{col}: {len(t)} values with >=10 cookies in both weeks; bot-rate corr week1 vs week2 = {r:.3f}')
        print(t['mean'].round(3).sort_values(False, ascending=False).head(8))
# android: one UA per cookie? UA switching?
print('\nUA switching within cookie (android) by target:', L.groupby('target').nua.mean().round(3).to_dict())
# android: session/timing signature: bots in-session dt histogram vs humans
e = ev[ev.cookie_id.isin(L.index)].join(m[['target']], on='cookie_id')
d = e[e['dt'].notna() & (e['dt'] <= 1800)]
print('\nandroid dt bands by target'); print(pd.crosstab(d.target, pd.cut(d['dt'], [-1, 2, 5, 10, 20, 30, 60, 120, 1800]), normalize='index').round(3))

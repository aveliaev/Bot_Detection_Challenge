"""C3.8 timing-generator forensics: session relaunch gaps, session start phase, delay quantization."""
import sys; sys.path.insert(0, 'research3')
from common import *
pd.set_option('display.width', 250)
ev = events(); m = meta(); o = pd.read_parquet(CACHE / 'groups_opt.parquet')
e = ev.join(m[['target']], on='cookie_id').join(o[['grp']], on='cookie_id')
e['grp'] = np.where(e.grp.isna(), np.where(e.target.isna(), 'T', 'd01'), e.grp)
first = ~e.cookie_id.duplicated()
for gap in (600, 1800):
    ns = first | (e['dt'] > gap)
    sg = e[ns & ~first]
    print(f'\n== session gaps (> {gap}s) quantiles (s) by grp');
    print(sg.groupby('grp')['dt'].quantile([.1, .25, .5, .75, .9]).unstack().round(0))
    print('share of gaps within 2h +- 15min / 1h +-10min')
    print(sg.groupby('grp')['dt'].agg(two_h=lambda x: ((x - 7200).abs() < 900).mean(), one_h=lambda x: ((x - 3600).abs() < 600).mean(), n='size').round(3))
# gap histogram fine for bots vs humans
ns = first | (e['dt'] > 1800); sg = e[ns & ~first & e.target.notna()]
print(pd.crosstab(pd.cut(sg['dt'], [1800, 3000, 4000, 5000, 6000, 6800, 7600, 8500, 10000, 15000, 20000, 90000]), sg.target, normalize='columns').round(3))
# number of sessions
e['sess'] = (first | (e['dt'] > 1800)).cumsum()
S = e.groupby('cookie_id').agg(grp=('grp', 'first'), target=('target', 'first'), nsess=('sess', 'nunique'), n=('t', 'size'))
print('\nsessions per cookie'); print(S.groupby('grp').nsess.describe().round(2))
# delay quantization: fraction of dt that are integers in exact seconds -> all ints; look at dt mod 5, 10
d = e[e['dt'].notna() & (e['dt'] <= 120) & (e['dt'] > 0)]
print('\n dt mod 5 ==0 share by target', d.groupby('target')['dt'].apply(lambda x: (x % 5 == 0).mean()).round(3).to_dict(), 'uniform expect 0.2')
print(' dt mod 10 ==0 share by target', d.groupby('target')['dt'].apply(lambda x: (x % 10 == 0).mean()).round(3).to_dict(), 'uniform 0.1')
# fine histogram of dt 1..40 for bots vs humans
h = pd.crosstab(d['dt'].clip(upper=40), d.target, normalize='columns')
print((h * 100).round(2).T.to_string())
# event second-of-minute & session start second-of-hour
st = e[first | (e['dt'] > 1800)].copy(); st['soh'] = st.t % 3600; st['som'] = st.t % 60
print('\nsession start second-of-hour: share within first 2 min of hour', st.groupby('grp').soh.apply(lambda x: (x < 120).mean()).round(3).to_dict(), 'uniform', round(120/3600, 3))
print('event second-of-minute==0 share', e.groupby('grp').t.apply(lambda x: (x % 60 == 0).mean()).round(4).to_dict())

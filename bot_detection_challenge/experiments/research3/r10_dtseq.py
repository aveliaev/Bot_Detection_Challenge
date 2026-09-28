"""Are shared delay 4-grams a replay template or low-entropy coincidence? Null = shuffle delays within class."""
import sys, json; sys.path.insert(0, 'research3')
from common import *
pd.set_option('display.width', 250)
ev = events(); m = meta(); pfp = set(json.load(open(CACHE / 'persistent.json')))
ev['y'] = ev.cookie_id.map(m.target)
def share_rate(e, dtcol):
    s = e[dtcol].fillna(-1).astype(int).astype(str); c = e.cookie_id
    ok = c.eq(c.shift(1)) & c.eq(c.shift(2)) & c.eq(c.shift(3)) & (e[dtcol].notna()) & e[dtcol].shift(3).notna()
    g = (s.shift(3) + ',' + s.shift(2) + ',' + s.shift(1) + ',' + s)[ok]
    G = pd.DataFrame({'c': c[ok].values, 'g': g.values}).drop_duplicates()
    n = G.groupby('g').c.transform('size')
    return G.assign(shared=n > 1)
e = ev[ev.y == 1].copy()
obs = share_rate(e, 'dt')
print('bots observed share rate', round(obs.shared.mean(), 4))
# null: permute dt values across all bot events (keeps marginal distribution, breaks sequence templates)
rng = np.random.RandomState(0)
for rep in range(3):
    e2 = e.copy(); msk = e2['dt'].notna(); e2.loc[msk, 'dt'] = rng.permutation(e2.loc[msk, 'dt'].values)
    print(' null (global shuffle of bot delays)', round(share_rate(e2, 'dt').shared.mean(), 4))
    # within-cookie shuffle keeps each cookie's own delay distribution
    e3 = e.copy(); e3['dt'] = e3.groupby('cookie_id')['dt'].transform(lambda x: pd.Series(rng.permutation(x.values), index=x.index))
    print(' null (within-cookie shuffle)', round(share_rate(e3, 'dt').shared.mean(), 4))
# examples of shared grams: which values
top = obs[obs.shared].g.value_counts().head(15); print(top)

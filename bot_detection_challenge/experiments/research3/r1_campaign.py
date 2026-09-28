"""Is bot category focus day-specific beyond sampling noise? cookie-level permutation null."""
import sys; sys.path.insert(0, 'research3')
from common import *
ev = events(); m = meta()
e = ev.join(m[['target']], on='cookie_id')
e = e[e.target.notna() & e.item_category.notna()]
cp = pd.crosstab(e.cookie_id, e.item_category, normalize='index')  # per-cookie category profile
info = m.loc[cp.index, ['day', 'target']]
def stat(P, day):
    # between-day variance of mean profile, weighted
    M = P.groupby(day.values).mean()
    return float(((M - P.mean()) ** 2).sum(axis=1).mean())
rng = np.random.RandomState(0)
for t in (1, 0):
    P = cp[info.target == t]; d = info.day[info.target == t]
    if t == 0:  # subsample humans to bot size per day for comparable noise
        idx = d.groupby(d).apply(lambda s: s.sample(min(len(s), 64), random_state=0).index.to_series()).values
        P, d = P.loc[idx], d.loc[idx]
    s0 = stat(P, d)
    null = [stat(P, pd.Series(rng.permutation(d.values), index=d.index)) for _ in range(300)]
    print('target', t, 'n', len(P), 'stat', round(s0, 5), 'null mean', round(np.mean(null), 5), 'p', np.mean(np.array(null) >= s0))
# same for dominant query
s = ev[ev.event_name == 'search_results_view'].join(m[['target']], on='cookie_id')
s = s[s.target.notna()]
qp = pd.crosstab(s.cookie_id, s.search_query, normalize='index'); qi = m.loc[qp.index, ['day', 'target']]
for t in (1, 0):
    P = qp[qi.target == t]; d = qi.day[qi.target == t]
    if t == 0:
        idx = d.groupby(d).apply(lambda s: s.sample(min(len(s), 50), random_state=0).index.to_series()).values
        P, d = P.loc[idx], d.loc[idx]
    s0 = stat(P, d); null = [stat(P, pd.Series(rng.permutation(d.values), index=d.index)) for _ in range(300)]
    print('QUERY target', t, 'n', len(P), 'stat', round(s0, 5), 'null', round(np.mean(null), 5), 'p', np.mean(np.array(null) >= s0))

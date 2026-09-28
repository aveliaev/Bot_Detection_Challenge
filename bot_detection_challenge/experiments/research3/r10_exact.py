"""C3.10/16 exact cross-cookie fingerprints + week1->week2 stability of motif log-odds."""
import sys, json; sys.path.insert(0, 'research3')
from common import *
from llr import patterns, _table, FAMS
pd.set_option('display.width', 250)
ev = events(); m = meta(); pfp = set(json.load(open(CACHE / 'persistent.json')))
y = ev.cookie_id.map(m.target)
same = lambda k: ev.cookie_id.eq(ev.cookie_id.shift(k))
def grams(series, k, name):
    s = series.astype(str)
    g = s.copy()
    ok = pd.Series(True, index=s.index)
    for j in range(1, k):
        g = s.shift(j).fillna('') + ',' + g; ok &= same(j)
    return pd.DataFrame({'cookie_id': ev.cookie_id, 'g': g})[ok]
for name, ser, k in [('dt4', ev['dt'].fillna(-1).astype(int), 4), ('dt3', ev['dt'].fillna(-1).astype(int), 3), ('ev6', ev.event_name.str[:4], 6),
                     ('xy', ev.pointer_x.astype('Int64').astype(str) + ':' + ev.pointer_y.astype('Int64').astype(str), 1)]:
    G = grams(ser, k, name).drop_duplicates()
    if name == 'xy':
        G = G[~G.g.str.contains('NA')]
    G['y'] = G.cookie_id.map(m.target); G['pfp'] = G.cookie_id.isin(pfp)
    nck = G.groupby('g').cookie_id.transform('size')
    nbot = G.groupby('g').y.transform(lambda v: (v == 1).sum())
    G['shared_other_bot'] = (nbot - (G.y == 1)) > 0
    G['shared_any'] = nck > 1
    r = G.groupby(np.where(G.pfp, 'pFP', G.y.map({1.0: 'bot', 0.0: 'human'}).fillna('test'))).agg(
        frac_shared_any=('shared_any', 'mean'), frac_shared_with_other_bot=('shared_other_bot', 'mean'), n=('g', 'size'))
    print(f'\n== {name}: exact {k}-gram sharing across cookies'); print(r.round(4))
# stability of motif LLR between weeks (bot vs all humans), patterns with support >= 10 cookies each week
P = patterns(); lab = m.target.dropna(); d = m.day
w1, w2 = lab[d[lab.index] < 7], lab[d[lab.index] >= 7]
print('\n== week1 vs week2 pattern LLR correlation (support-weighted)')
for f in FAMS:
    l1, p1 = _table(P[P.cookie_id.isin(w1.index)], f, set(w1.index[w1 == 1]), set(w1.index[w1 == 0]))
    l2, p2 = _table(P[P.cookie_id.isin(w2.index)], f, set(w2.index[w2 == 1]), set(w2.index[w2 == 0]))
    j = l1.index.intersection(l2.index)
    sup = (p1[j] * len(w1)).clip(upper=500)
    k = j[(p1[j] * len(w1) >= 10) & (p2[j] * len(w2) >= 10)]
    print(f'  {f:5s} patterns={len(j):6d} with support>=10: {len(k):5d}  corr={np.corrcoef(l1[k], l2[k])[0, 1]:.3f}')

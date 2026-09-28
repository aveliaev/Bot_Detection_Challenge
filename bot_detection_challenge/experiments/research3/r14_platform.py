"""C3.14 per-platform P@R (inside segment) and fwd7 for B0 and every cycle-3 candidate (averaged repeated OOF)."""
import sys, json; sys.path.insert(0, 'research3')
from common import *
m = meta(); ids = pd.read_csv(ROOT / 'research2' / 'oof_final_B0.csv').cookie_id.values; y = m.target.reindex(ids).values
T = pd.read_parquet(CACHE / 'traits.parquet'); pl = T.pl.reindex(ids).values
def load(name, d, sch=None):
    if sch: return np.load(f'{d}/{name}__{sch}.npy')
    return np.mean([np.load(f'{d}/{name}__P{i}.npy') for i in range(5)], axis=0)
rows = []
cands = [('B0_cat_top120', 'research2/oof')] + [(n, 'research3/oof') for n in sys.argv[1:]]
for n, d in cands:
    s = load(n, d); mk = s > 0; f = load(n, d, 'fwd7'); mf = f > 0
    r = {'exp': n, 'all': precision_at_recall(y[mk], s[mk]), 'fwd7': precision_at_recall(y[mf], f[mf])}
    for p in ['web', 'android', 'ios']:
        k = mk & (pl == p); r[p] = precision_at_recall(y[k], s[k])
    rows.append(r)
print(pd.DataFrame(rows).round(4).to_string(index=False))

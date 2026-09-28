"""Ceiling simulation: value of an extra, independent signal with class-separation AUC a (bots vs humans among
the top-1500 B0 cookies), blended with B0 at the best weight (optimistic: weight chosen on the same data)."""
import sys; sys.path.insert(0, 'research3')
from common import *
from scipy.stats import norm
from scipy.special import logit
o = pd.read_csv(ROOT / 'research2' / 'oof_final_B0.csv'); o = o[o.oof_repeated_dayblock > 0]
y = o.target.values; s = logit(o.oof_repeated_dayblock.rank(pct=True).clip(1e-4, 1 - 1e-4).values)
base = precision_at_recall(y, s); print('B0', round(base, 4))
rng = np.random.RandomState(0)
rows = []
for a in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90]:
    dprime = np.sqrt(2) * norm.ppf(a)
    res = []
    for r in range(30):
        z = rng.randn(len(y)) + dprime * y
        best = max(precision_at_recall(y, s + w * z) for w in [0, .1, .2, .3, .5, .7, 1.0, 1.5, 2.0])
        res.append(best)
    rows.append((a, round(np.mean(res), 4), round(np.mean(res) - base, 4)))
print(pd.DataFrame(rows, columns=['AUC of extra signal (bot vs human)', 'best blended P@R', 'gain']).to_string(index=False))

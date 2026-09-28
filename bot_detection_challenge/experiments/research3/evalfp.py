"""For an experiment: averaged repeated OOF -> P@R, FP count at the operating point, how many persistent FP remain flagged,
AUC TPnear vs pFP, paired bootstrap vs B0."""
import sys, json; sys.path.insert(0, 'research3'); sys.path.insert(0, 'research2')
from common import *
from rcv import paired_bootstrap
from sklearn.metrics import roc_auc_score
m = meta(); ids = pd.read_csv(ROOT / 'research2' / 'oof_final_B0.csv').cookie_id.values; y = m.target.reindex(ids).values
pfp = set(json.load(open(CACHE / 'persistent.json')))
def rep(name, d):
    return np.mean([np.load(f'{d}/{name}__P{i}.npy') for i in range(5)], axis=0)
b0 = rep('B0_cat_top120', 'research2/oof'); mk = b0 > 0
def stats(s):
    ss = s[mk]; yy = y[mk]; ii = ids[mk]
    order = np.argsort(-ss, kind='mergesort'); tp = np.cumsum(yy[order]); k = np.arange(1, len(yy) + 1)
    ok = tp / yy.sum() >= .7; i = np.flatnonzero(ok)[np.argmax((tp / k)[ok])]
    flagged = set(ii[order[:i + 1]])
    rk = pd.Series(ss, index=ii).rank(ascending=False)
    near = [c for c, t in zip(ii, yy) if t == 1 and 300 <= rk[c] <= 604]
    return dict(PR=round(precision_at_recall(yy, ss), 4), FP=int(i + 1 - tp[i]), pFP_flagged=len(flagged & pfp))
print('B0', stats(b0))
for name in sys.argv[1:]:
    s = rep(name, 'research3/oof'); r = stats(s)
    r.update({k: (round(v, 4) if isinstance(v, float) else v) for k, v in paired_bootstrap(y, s, b0, n=1000).items()})
    print(name, r)

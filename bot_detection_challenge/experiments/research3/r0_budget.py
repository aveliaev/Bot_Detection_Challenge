"""C3.0 exact error budget at the operating point (repeated OOF of B0, days>=2)."""
import sys; sys.path.insert(0, 'research3')
from common import *
o = pd.read_csv(ROOT / 'research2' / 'oof_final_B0.csv')
o = o[o.oof_repeated_dayblock > 0]
y, s = o.target.values, o.oof_repeated_dayblock.values
P = int(y.sum()); print('rows', len(y), 'positives', P, 'P@R', precision_at_recall(y, s))
order = np.argsort(-s, kind='mergesort'); ys = y[order]
tp = np.cumsum(ys); k = np.arange(1, len(ys) + 1); prec = tp / k; rec = tp / P
ok = rec >= .7; i = np.flatnonzero(ok)[np.argmax(prec[ok])]
TP, FP = int(tp[i]), int(k[i] - tp[i]); FN = P - TP
print(f'operating point: flagged={k[i]} TP={TP} FP={FP} FN={FN} precision={prec[i]:.4f} recall={rec[i]:.4f}')
need_tp = int(np.ceil(0.7 * P)); print('min TP for recall .70 =', need_tp)
print('\nFP allowed at recall=.70 (TP fixed =', need_tp, ') for target precision:')
for p in [.90, .91, .92, .93, .95, .97, .98, .99]:
    fp_max = int(np.floor(need_tp * (1 - p) / p))
    print(f'  P@R={p:.2f}: FP<={fp_max:3d}  -> remove {max(0, FP - fp_max)} FP (from current {FP}) ')
# where do the FPs sit? precision by rank band
for a, b in [(0, 100), (100, 200), (200, 300), (300, 400), (400, 500), (500, k[i])]:
    print(f'rank {a:3d}-{b:3d}: FP={int((1 - ys[a:b]).sum()):3d} TP={int(ys[a:b].sum())}')
# fwd7 view
f = o[o.oof_fwd7 > 0]
print('\nfwd7 rows', len(f), 'P@R', round(precision_at_recall(f.target, f.oof_fwd7), 4))

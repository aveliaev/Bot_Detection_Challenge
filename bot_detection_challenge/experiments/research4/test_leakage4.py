"""Leakage tests for the cycle-4 fusion layer.
 1) repeated CV: the cross-fitted fusion score of fold k does not change when the labels of fold k are flipped;
 2) fwd7: fusion scores of the fwd7 validation rows do not change when ALL fwd7 validation labels are flipped
    (parameters come from the nested inner OOF of the training days only);
 3) channels are disjoint (no feature in two experts) and the mouse expert sees no item / behaviour column.
Run: python research4/test_leakage4.py"""
import sys, json, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common4
from fusion import *
import r4_eval as ev

ok = {}
def check(a, b, name):
    r = np.allclose(np.nan_to_num(a, nan=-9), np.nan_to_num(b, nan=-9)); ok[name] = r
    print(('ok:   ' if r else 'FAIL: ') + name, flush=True)

E = load('P0'); mask = valid_mask(E); Sp, Sr = scaled(E, mask, 'prob'), scaled(E, mask, 'rank')
tri, vai = folds_of('P0')[2]
trm = mask.copy(); trm[vai] = False; tr = np.flatnonzero(trm)
for kind in ['wmean_rank', 'iliou_fit_beh', 'meta_lr', 'meta_cat2']:
    y1 = Y.copy(); y2 = Y.copy(); y2[vai] = 1 - y2[vai]
    a, _ = ev.fitted_scores(kind, ev.sub(Sr, tr), ev.sub(Sp, tr), E['B0'][tr], y1[tr], ev.sub(Sr, vai), ev.sub(Sp, vai), E['B0'][vai])
    b, _ = ev.fitted_scores(kind, ev.sub(Sr, tr), ev.sub(Sp, tr), E['B0'][tr], y2[tr], ev.sub(Sr, vai), ev.sub(Sp, vai), E['B0'][vai])
    check(a, b, f'P0 fold 2, {kind}: own-fold labels')

E7 = load('fwd7'); m7 = valid_mask(E7); Sp7, Sr7 = scaled(E7, m7, 'prob'), scaled(E7, m7, 'rank')
I = dict(np.load(R4 / 'oof' / 'experts_fwd7inner_k7.npz')); im = ~np.isnan(I['B0']); itr = np.flatnonzero(im)
print('inner rows use only days', sorted(set(DAY[itr])), '(fwd7 k=7 validation = days 7..13)')
assert DAY[itr].max() < 7
va = np.flatnonzero(m7 & (DAY >= 7))
for kind in ['wmean_rank', 'meta_lr']:
    y2 = Y.copy(); y2[DAY >= 7] = 1 - y2[DAY >= 7]
    Ir, Ip = scaled(I, im, 'rank'), scaled(I, im, 'prob')
    a, _ = ev.fitted_scores(kind, ev.sub(Ir, itr), ev.sub(Ip, itr), I['B0'][itr], Y[itr], ev.sub(Sr7, va), ev.sub(Sp7, va), E7['B0'][va])
    b, _ = ev.fitted_scores(kind, ev.sub(Ir, itr), ev.sub(Ip, itr), I['B0'][itr], y2[itr], ev.sub(Sr7, va), ev.sub(Sp7, va), E7['B0'][va])
    check(a, b, f'fwd7 k=7, {kind}: validation-day labels')

CH = json.load(open(R4 / 'channels.json'))
allc = sum(CH.values(), []); r = len(allc) == len(set(allc)); ok['channels disjoint'] = r
print(('ok:   ' if r else 'FAIL: ') + 'channels disjoint')
bad = [c for c in CH['mouse'] if not c.startswith('ptr')]
ok['mouse expert sees only pointer columns'] = not bad
print(('ok:   ' if not bad else 'FAIL: ') + 'mouse expert sees only pointer columns', bad)
bad = [c for c in CH['behavior'] if c.startswith(('ptr', 'pool_', 'pop_', 'item', 'cat_', 'loc_', 'sel_', 'ua_', 'pl_'))]
ok['behavior expert sees no mouse/item/UA column'] = not bad
print(('ok:   ' if not bad else 'FAIL: ') + 'behavior expert sees no mouse/item/UA column', bad)
bad = [c for c in CH['item'] if c.startswith(('ptr', 'ua_', 'pl_', 'dt_', 'sdt_'))]
ok['item expert sees no mouse/UA/timing column'] = not bad
print(('ok:   ' if not bad else 'FAIL: ') + 'item expert sees no mouse/UA/timing column', bad)
print('\nSUMMARY', {k: 'ok' if v else 'FAIL' for k, v in ok.items()})

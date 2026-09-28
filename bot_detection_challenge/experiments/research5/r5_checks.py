"""Cycle 5 checks: (1) boundary crossings of the nested procedure by subtype; (2) leakage: flipping the labels of an
outer fold (or of all fwd7 validation days) changes neither its scores nor the chosen config."""
import sys, pickle, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import local, common4
from local import *
import r5_summary as S
print('\n== (1) nested crossings by subtype (summed over P0..P4 and fwd7)')
rows = []
for sch in SCHEMES:
    mask, b0, s = S.assemble(sch)
    ids, y = IDS[mask], Y[mask]; sub = pd.Series(np.where(PL == 'web', np.where(~np.isnan(load(sch)['mouse']), 'web+mouse', 'web no mouse'), np.where(PL == 'android', 'android', 'ios')), index=IDS)
    A, B = op_point(y, b0[mask], ids)['flagged'], op_point(y, s[mask], ids)['flagged']
    for c in (A - B) | (B - A):
        rows.append(dict(scheme=sch, sub=sub[c], kind=('TP' if common4.Y[np.flatnonzero(IDS == c)[0]] else 'FP') + ('_removed' if c in A else '_added')))
X = pd.DataFrame(rows); print(pd.crosstab(X['sub'], X['kind']))

print('\n== (2) leakage tests')
import r5_eval as ev
ok = {}
for sch, k in [('P0', 3), ('fwd7', 7)]:
    a = ev.outer_job(sch, k)
    Y0 = common4.Y.copy(); Yf = Y0.copy()
    if sch == 'P0':
        vai = local.folds_of('P0')[k][1]; Yf[vai] = 1 - Yf[vai]
    else:
        Yf[DAY >= 7] = 1 - Yf[DAY >= 7]
    for mod in (common4, local, ev): mod.Y = Yf
    b = ev.outer_job(sch, k)
    for mod in (common4, local, ev): mod.Y = Y0
    same = a['chosen'] == b['chosen'] and all(np.allclose(a['fixed'][key], b['fixed'][key]) for key in a['fixed'])
    ok[f'{sch} fold {k}: outer labels flipped -> identical scores and choice'] = same
for k, v in ok.items(): print(('ok:   ' if v else 'FAIL: ') + k)

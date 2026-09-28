"""Leakage unit tests for the cycle-3 label-dependent families (llr, manifold, uate).
Run:  python research3/test_leakage3.py
"""
import sys, warnings; sys.path.insert(0, 'research3'); sys.path.insert(0, 'research2'); warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from rcv import Data
from llr import llr_features
from manifold import manifold_features
from uate import ua_te_features

D = Data()
lab = D.labels.copy()
day = pd.Series(D.day, index=D.ids)
Dt = 8
rows_d = day.index[day == Dt]
res = {}


def check(a, b, name):
    a, b = a.sort_index(), b.sort_index()
    ok = a.index.equals(b.index) and np.allclose(a.fillna(-9).values.astype(float), b.fillna(-9).values.astype(float))
    res[name] = ok
    print(('ok:   ' if ok else 'FAIL: ') + name, flush=True)


# 1) training rows: own-day labels must not influence own features (inner leave-one-day-out)
lab_f = lab.copy(); lab_f.loc[rows_d] = 1 - lab_f.loc[rows_d]
check(llr_features(D, lab, rows_d, neg='all'), llr_features(D, lab_f, rows_d, neg='all'), 'llr(all): own-day labels of training rows')
check(manifold_features(D, lab, rows_d, which=('knn', 'gmm', 'maha', 'proto')), manifold_features(D, lab_f, rows_d, which=('knn', 'gmm', 'maha', 'proto')),
      'manifold: own-day labels of training rows')
check(llr_features(D, lab, rows_d, neg='hard'), llr_features(D, lab_f, rows_d, neg='hard'), 'llr(hard): own-day labels of training rows')

# 2) uate: labels of the target day and later must not matter (temporal rule like the item pool)
fut = day.index[day >= Dt]
lab_t = lab.copy(); lab_t.loc[fut] = 1 - lab_t.loc[fut]
check(ua_te_features(D, lab, rows_d), ua_te_features(D, lab_t, rows_d), 'uate: same-day and future labels')

# 3) validation rows (not in lab): their own labels are never passed -> features identical whatever they are
lab_wo = lab.drop(rows_d)
check(llr_features(D, lab_wo, rows_d, neg='all'), llr_features(D, lab_wo.copy(), rows_d, neg='all'), 'llr: validation rows see only training labels (determinism)')

print('\nSUMMARY', {k: 'ok' if v else 'FAIL' for k, v in res.items()})
check(llr_features(D, lab, rows_d, neg='trait'), llr_features(D, lab_f, rows_d, neg='trait'), 'llr(trait): own-day labels of training rows')
print('SUMMARY2', {k: 'ok' if v else 'FAIL' for k, v in res.items()})

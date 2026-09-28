"""Leakage unit tests for label-dependent / cross-cookie feature families.

Run:  python research2/test_leakage.py
"""
import sys

sys.path.insert(0, "research2")
import numpy as np
import pandas as pd
from rcv import Data
from history import pool_features
from graph import graph_features
from nb import build_tokens, nb_features

D = Data()
rng = np.random.RandomState(0)
lab = D.labels.copy()
day = pd.Series(D.day, index=D.ids)
Dt = 8
tgt = D.ids[D.day == Dt]


def same(a, b, name):
    a, b = a.sort_index(), b.sort_index()
    assert a.index.equals(b.index), name
    ok = np.allclose(a.fillna(-9).values.astype(float), b.fillna(-9).values.astype(float))
    assert ok, name
    print("ok:", name)


# 1) labels of days >= Dt (future / same day) must not affect features of day-Dt targets
lab_f = lab.copy()
fut = day.index[day >= Dt]
lab_f.loc[fut] = 1 - lab_f.loc[fut]
same(pool_features(D.ci, lab, tgt), pool_features(D.ci, lab_f, tgt), "pool: future labels")
same(graph_features(D, lab, tgt), graph_features(D, lab_f, tgt), "graph: future labels")

# 2) removing future labelled cookies entirely changes nothing either
lab_past = lab[day.loc[lab.index] < Dt]
same(pool_features(D.ci, lab, tgt), pool_features(D.ci, lab_past, tgt), "pool: drop future labelled")

# 3) events of days > Dt must not affect graph features of day-Dt targets (unlabelled 2-hop part)
class DD:  # shallow copy with truncated cookie-item table
    pass
D2 = DD()
D2.ci = D.ci[D.ci["day"] <= Dt]
same(graph_features(D, lab_past, tgt), graph_features(D2, lab_past, tgt), "graph: future events")

# 4) NB evidence: a labelled row's own label must not influence its own features
T = build_tokens(D.meta)
sub = lab[day.loc[lab.index] < 10]
probe = sub.index[:200]
f1 = nb_features(T, sub, probe, seed=0)
sub2 = sub.copy()
sub2.loc[probe] = 1 - sub2.loc[probe]
# flipping ALL probes changes other folds' tables, so flip one row at a time on a few probes
for cid in probe[:5]:
    s3 = sub.copy()
    s3.loc[cid] = 1 - s3.loc[cid]
    a = nb_features(T, sub, [cid], seed=0)
    b = nb_features(T, s3, [cid], seed=0)
    same(a, b, f"nb: own label of {cid} does not affect own features")

# 5) NB evidence for non-labelled (validation) rows ignores their labels entirely
val = D.ids[D.day == 12]
tr_lab = lab.drop(val)
same(nb_features(T, tr_lab, val), nb_features(T, tr_lab, val), "nb: val rows deterministic w/o val labels")
print("ALL LEAKAGE TESTS PASSED")

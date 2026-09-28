"""Leakage tests + adversarial validation for cycle 6.5. Run: python research65/test_leakage65.py
 1. flipping the labels of the validation fold does not change its residuals (pool uses training labels only)
 2. validation rows do not take part in the mapping fit or its preprocessing (garbage validation data -> same model)
 3. inner residual of a training day is produced without that day (garbage day-d data -> same residuals)
 4. representations contain no target / B0 / expert-probability / UA / cookie_id column
 5. adversarial validation of the residual family: train vs test cookies"""
import sys, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
R65 = Path(__file__).resolve().parent; ROOT = R65.parent
sys.path.insert(0, str(R65)); sys.path.insert(0, str(ROOT / 'research2')); sys.path.insert(0, str(ROOT / 'cycle1'))
import numpy as np, pandas as pd
from rcv import Harness
from history import pool_features
from representations import build
from cross_modal import Mapping
from residuals import resid_block, RESID_MAPS
ok = {}


def check(c, name):
    ok[name] = bool(c); print(('ok:   ' if c else 'FAIL: ') + name, flush=True)


def same(a, b):
    return a.index.equals(b.index) and np.allclose(a.fillna(-9).values, b.fillna(-9).values)


H = Harness(n_rep=1); D = H.D
tri, vai = H.schemes['P0'][2]
ti, vi = pd.Index(D.ids[tri]), pd.Index(D.ids[vai])
lab = D.labels.iloc[tri]
# 1
lab_all = D.labels.copy(); lab_flip = lab_all.copy(); lab_flip.iloc[vai] = 1 - lab_flip.iloc[vai]
P1 = pool_features(D.ci, lab_all.iloc[tri], D.ids, 7); P2 = pool_features(D.ci, lab_flip.iloc[tri], D.ids, 7)
R1, R2 = build(P1, D.ids), build(P2, D.ids)
check(same(resid_block(R1, ti, vi), resid_block(R2, ti, vi)), 'validation-fold labels flipped -> identical validation residuals')
# 2
Rg = {k: v.copy() for k, v in R1.items()}
for k in ('mouse', 'behavior', 'item'):
    Rg[k].loc[vi] = 1e6
for name in RESID_MAPS:
    a = Mapping(name).fit(R1, ti).predict(R1, Mapping(name).rows(R1, vi))
    b = Mapping(name).fit(Rg, ti).predict(R1, Mapping(name).rows(R1, vi))
    check(same(a, b), f'{name}: validation rows not used in fit / preprocessing')
# 3
days = D.day[tri]; d = np.unique(days)[3]
fit_ids, pred_ids = pd.Index(D.ids[tri[days != d]]), pd.Index(D.ids[tri[days == d]])
Rg2 = {k: v.copy() for k, v in R1.items()}
for k in ('mouse', 'behavior', 'item'):
    Rg2[k].loc[pred_ids] = -1e6
a = resid_block(R1, fit_ids, pred_ids)
b = pd.concat([__import__('cross_modal').summarize(n, Mapping(n).fit(Rg2, fit_ids).residual(R1, pred_ids)) for n in RESID_MAPS], axis=1).reindex(pred_ids)
check(same(a, b), f'inner residuals of training day {d} computed without day {d}')
# 4
cols = sum([list(v.columns) for v in R1.values()], [])
bad = [c for c in cols if any(s in c.lower() for s in ('target', 'b0', 'expert', 'prob', 'ua_', 'cookie'))]
check(not bad, f'representation columns free of target/B0/expert/UA/cookie_id {bad}')
# 5 adversarial: residuals for test cookies from mappings fitted on all train cookies (pool from all train labels)
import lightgbm as lgb
from sklearn.model_selection import cross_val_score, StratifiedKFold
te = D.te.cookie_id.values; allids = np.r_[D.ids, te]
Pall = pool_features(D.ci, D.labels, allids, 7)
Rall = build(Pall, allids)
tr_ids = pd.Index(D.ids)
Rtr = pd.concat([resid_block(Rall, pd.Index(D.ids[D.day != dd]), pd.Index(D.ids[D.day == dd])) for dd in np.unique(D.day)])   # train rows: LODO
Rte = resid_block(Rall, tr_ids, pd.Index(te))
X = pd.concat([Rtr, Rte]); y = np.r_[np.zeros(len(Rtr)), np.ones(len(Rte))]
cv = StratifiedKFold(5, shuffle=True, random_state=0)
for fam in ['IB', 'BI', 'MIB', 'MB']:
    c = [x for x in X.columns if x.split('_')[0] == fam]
    auc = cross_val_score(lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, verbose=-1, n_jobs=4), X[c], y, cv=cv, scoring='roc_auc').mean()
    print(f'adversarial AUC train-vs-test, {fam} residuals: {auc:.3f}')
print('\nSUMMARY', {k: 'ok' if v else 'FAIL' for k, v in ok.items()})

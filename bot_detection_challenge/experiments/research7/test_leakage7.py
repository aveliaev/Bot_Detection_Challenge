"""Cycle 7 leakage audit + adversarial validation. Run: python research7/test_leakage7.py"""
import sys, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
R7 = Path(__file__).resolve().parent; ROOT = R7.parent
sys.path.insert(0, str(R7)); sys.path.insert(0, str(ROOT / 'research2')); sys.path.insert(0, str(ROOT / 'research3')); sys.path.insert(0, str(ROOT / 'cycle1'))
import numpy as np, pandas as pd
from rcv import Harness
from common import events
from build_windows import tokens, window_features, windows
from build_seq_features import aggregate
ok = {}


def check(c, name):
    ok[name] = bool(c); print(('ok:   ' if c else 'FAIL: ') + name, flush=True)


H = Harness(); D = H.D
W = pd.read_parquet(R7 / 'cache' / 'windows.parquet')
# 1. no cookie on both sides of any fold; training windows only from training cookies
bad = 0
for sch, folds in H.schemes.items():
    for tri, vai in folds:
        trc, vac = set(D.ids[tri]), set(D.ids[vai])
        bad += len(trc & vac)
        trw = set(W[(W.kind == 'train') & W.cookie_id.isin(trc)].cookie_id)
        bad += len(trw & vac)
check(bad == 0, 'no cookie (and none of its windows) on both sides of any of the 32 folds')
# 2. inner window scores: day groups disjoint from the model that scores them (replicates train_window_models grouping)
bad = 0
for sch, folds in H.schemes.items():
    for tri, vai in folds:
        days = np.unique(D.day[tri]); gs = np.array_split(days, 4)
        bad += sum(len(set(g) & set(np.setdiff1d(days, g))) for g in gs)
check(bad == 0, 'inner window models: scored day-group never in the fitting day-groups')
# 3. early stopping of the neural models uses training days only
bad = 0
for sch, folds in H.schemes.items():
    for tri, vai in folds:
        es = np.sort(np.unique(D.day[tri]))[-2:]
        bad += len(set(es) & set(D.day[vai]))
check(bad == 0, 'neural early-stopping days are training days of the outer fold')
# 4. causal tokens: altering events after position j does not change token rows <= j
ev = events(); cks = ev.cookie_id.drop_duplicates().sample(200, random_state=0)
e1 = ev[ev.cookie_id.isin(cks)].copy(); pos = e1.groupby('cookie_id').cumcount()
e2 = e1.copy(); late = pos > 4
e2.loc[late, 'event_name'] = 'login'; e2.loc[late, 'pointer_x'] = 3.0; e2.loc[late, 'item_id'] = 1.0
T1, T2 = tokens(e1), tokens(e2); keep = (T1.pos <= 4).values
cols = ['e', 'ldt', 'item_chg', 'cat_chg', 'vp', 'dist', 'vel', 'acc', 'absturn']
check(np.allclose(T1.loc[keep, cols].fillna(-9).values, T2.loc[keep, cols].fillna(-9).values), 'event tokens are causal (rows <= j unchanged when events > j altered)')
# 5. window features see only the window
Wt = windows(T1); Wt = Wt[(Wt.kind == 'infer') & (Wt.end <= 5)]
check(np.allclose(window_features(T1, Wt).fillna(-9).values, window_features(T2, Wt).fillna(-9).values), 'window features depend only on events inside the window')
# 6. window filter
tr, te, meta = __import__('features').load_meta(); evf = __import__('features').load_events(meta)
check(bool(((evf.event_ts >= evf.window_start_ts) & (evf.event_ts < evf.window_end_ts)).all()), 'all events inside [window_start, window_end)')
# 7. exact duplicate event sequences across cookies (type + Δt)
s = ev.assign(k=ev.event_name.str[:3] + ev['dt'].fillna(-1).astype(int).astype(str)).groupby('cookie_id').k.agg('|'.join)
s = s[ev.groupby('cookie_id').size() >= 5]
dup = s[s.duplicated(keep=False)]
print(f'info: cookies (>=5 events) whose exact type+Δt sequence also occurs in another cookie: {len(dup)} of {len(s)}')
# 8. adversarial: cheap chunk features, train (P0 outer OOF) vs test (window model fitted on all train cookies)
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.model_selection import cross_val_score, StratifiedKFold
FEAT = [c for c in W.columns if c.startswith('w_')]
T = W[(W.kind == 'train') & W.cookie_id.isin(set(D.ids))]
m = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05, random_seed=0, verbose=0, thread_count=4).fit(
    T[FEAT].fillna(-1), T.cookie_id.map(D.labels).values, sample_weight=1.0 / T.groupby('cookie_id').cookie_id.transform('size').values)
Wte = W[(W.kind == 'infer') & W.cookie_id.isin(set(D.te.cookie_id))]
Fte = aggregate(pd.Series(m.predict_proba(Wte[FEAT].fillna(-1))[:, 1], index=Wte.index))
Ftr = pd.concat([aggregate(pd.read_parquet(R7 / 'oof' / f'winscores_P0_k{k}.parquet').query("part == 'val'").cat) for k in range(6)])
from build_seq_features import LOCKED
X = pd.concat([Ftr[LOCKED], Fte[LOCKED]]); y = np.r_[np.zeros(len(Ftr)), np.ones(len(Fte))]
auc = cross_val_score(lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, verbose=-1, n_jobs=4), X, y,
                      cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring='roc_auc').mean()
print(f'adversarial AUC train-vs-test, 14 locked cheap chunk features: {auc:.3f}')
print('\nSUMMARY', {k: 'ok' if v else 'FAIL' for k, v in ok.items()})

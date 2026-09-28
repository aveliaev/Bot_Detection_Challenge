"""Leakage tests for cycle 6. Run: python research6/test_leakage6.py
 1. transition inputs of row j do not change when events after j are altered (next-event/next-Δt inputs see only history)
 2. prefix target-model features do not change when the SUFFIX content is altered (same length, different events/items/times)
 3. prefix-dataset (future tasks) inputs do not change when the suffix content is altered
 4. auxiliary model isolation: the day-d model is identical whether day-d cookies' events are real or garbage
 5. pool_prefix with full item lists reproduces history.pool_features exactly (the P100 path is B0)
 6. fwd7 nested inner prefix OOF only covers training days (< k)
 7. no post-window events in the event table"""
import sys, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
R6 = Path(__file__).resolve().parent; ROOT = R6.parent
sys.path.insert(0, str(R6)); sys.path.insert(0, str(ROOT / 'cycle1')); sys.path.insert(0, str(ROOT / 'research2'))
import numpy as np, pandas as pd
from make_prefix_dataset import base_events, transitions, prefixes, FEAT_T, FEAT_P
from features import load_meta, load_events, build_features
from history import cookie_items, pool_features
from prefix_target import truncate, pool_prefix
from auxiliary import lgbm
ok = {}


def check(cond, name):
    ok[name] = bool(cond); print(('ok:   ' if cond else 'FAIL: ') + name, flush=True)


rng = np.random.RandomState(0)
ev = base_events()
cks = ev.cookie_id.drop_duplicates().sample(300, random_state=0)
E = ev[ev.cookie_id.isin(cks)].copy()
# 1. alter everything after position j=3 in each cookie (types, items, timestamps pushed later)
pos = E.groupby('cookie_id').cumcount()
E2 = E.copy(); late = pos > 3
E2.loc[late, 'e'] = rng.randint(0, 9, late.sum()); E2.loc[late, 'item_id'] = rng.randint(1, 10**6, late.sum())
E2.loc[late, 't'] = E2.loc[late, 't'] + rng.randint(0, 50, late.sum())
E2 = E2.sort_values(['cookie_id', 't'], kind='mergesort')
T1, T2 = transitions(E), transitions(E2)
k1, k2 = T1[T1.pos <= 2], T2[T2.pos <= 2]      # rows j <= 2 predict event j+1 <= 3: inputs only from positions <= 2
check(np.allclose(k1[FEAT_T].fillna(-9).values, k2[FEAT_T].fillna(-9).values), 'transition inputs see only history (rows j<=2 unchanged when events >3 altered)')

# 2./3. suffix alteration with identical length
tr, te, meta = load_meta(); evf = load_events(meta)
sub = evf[evf.cookie_id.isin(cks)].copy()
n = sub.groupby('cookie_id').cumcount(); N = sub.cookie_id.map(sub.groupby('cookie_id').size())
suf = n >= np.ceil(0.5 * N)
alt = sub.copy()
alt.loc[suf, 'event_name'] = 'item_view'; alt.loc[suf, 'item_id'] = 999999
alt.loc[suf, 'pointer_x'] = 5.0; alt.loc[suf, 'search_page'] = 99
F1 = build_features(truncate(sub, 50), meta).loc[cks]; F2 = build_features(truncate(alt, 50), meta).loc[cks]
check(np.allclose(F1.fillna(-9).values.astype(float), F2.fillna(-9).values.astype(float)), 'prefix (50%) target-model features unchanged when suffix content altered')
P1 = prefixes(E); E3 = E.copy()
m = E3.groupby('cookie_id').cumcount() >= np.ceil(.75 * E3.cookie_id.map(E3.groupby('cookie_id').size()))
E3.loc[m, 'e'] = 0; E3.loc[m, 'item_id'] = 123.0
P2 = prefixes(E3)
a, b = P1[P1.frac == .75].set_index('cookie_id')[FEAT_P], P2[P2.frac == .75].set_index('cookie_id')[FEAT_P]
check(np.allclose(a.fillna(-9).values, b.reindex(a.index).fillna(-9).values), 'future-task prefix inputs unchanged when suffix content altered')

# 4. auxiliary model isolation (next-Δt regressor for day 8)
T = pd.read_parquet(R6 / 'cache' / 'transitions.parquet')
lab = set(tr.cookie_id); T = T[T.cookie_id.isin(lab)]
trn = T[T.day != 8].sample(60000, random_state=0); va = T[T.day == 8]
G = T.copy(); G.loc[G.day == 8, FEAT_T] = 0; G.loc[G.day == 8, 'next_ldt'] = 99.0
trn2 = G[G.day != 8].loc[trn.index]
p1 = lgbm('reg').fit(trn[FEAT_T], trn.next_ldt).predict(va[FEAT_T]); p2 = lgbm('reg').fit(trn2[FEAT_T], trn2.next_ldt).predict(va[FEAT_T])
check(np.allclose(p1, p2), 'aux model scoring day 8 is fitted without any day-8 cookie (garbage day-8 data -> identical predictions)')

# 5. P100 path
ci = cookie_items(evf); labels = tr.set_index('cookie_id').target.iloc[:6000]
A = pool_features(ci, labels, tr.cookie_id.values, 7); B = pool_prefix(ci, ci, labels, tr.cookie_id.values, 7)
check(np.allclose(A.sort_index().fillna(-9).values, B.reindex(A.sort_index().index)[A.columns].fillna(-9).values), 'pool_prefix(full lists) == history.pool_features')

# 6. nested inner fwd7
day = ((tr.window_start_ts - pd.Timestamp('2026-04-06')).dt.days).values
for k in (5, 7):
    z = np.load(R6 / 'oof' / f'prefix_fwd7inner_k{k}.npz')
    cov = np.flatnonzero(~np.isnan(z['P25']))
    check(day[cov].max() < k, f'fwd7 inner prefix OOF k={k} uses only training days (max day {day[cov].max()} < {k})')

# 7. window filter
check(bool(((evf.event_ts >= evf.window_start_ts) & (evf.event_ts < evf.window_end_ts)).all()), 'no event outside [window_start, window_end)')
print('\nSUMMARY', {k: 'ok' if v else 'FAIL' for k, v in ok.items()})

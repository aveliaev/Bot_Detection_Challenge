"""Persistent FP / FN across 4 model families x 5 partitions = 20 runs (flag = top-604 of each run)."""
import sys, json; sys.path.insert(0, 'research3'); sys.path.insert(0, 'research2')
from common import *
from rcv import OOF2
m = meta(); o = pd.read_parquet(CACHE / 'groups_opt.parquet')
ids = pd.read_csv(ROOT / 'research2' / 'oof_final_B0.csv').cookie_id.values
y = m.target.reindex(ids).values
flag = []
for mdl in ['cat', 'lgb', 'xgb', 'et']:
    for i in range(5):
        a = np.load(OOF2 / f'B0_{mdl}_top120__P{i}.npy'); mk = a > 0
        thr = np.sort(a[mk])[::-1][603]; flag.append((a >= thr) & mk)
Fl = np.array(flag); share = Fl.mean(0); mk = np.load(OOF2 / 'B0_cat_top120__P0.npy') > 0
S = pd.Series(share, index=ids)[mk]; yy = pd.Series(y, index=ids)[mk]
print('negatives by share of 20 runs:', pd.cut(S[yy == 0], [-.01, 0, .25, .5, .75, .99, 1]).value_counts().sort_index().to_dict())
print('positives by share:', pd.cut(S[yy == 1], [-.01, 0, .25, .5, .75, .99, 1]).value_counts().sort_index().to_dict())
pfp = S[(yy == 0) & (S >= 0.999)].index; pfn = S[(yy == 1) & (S == 0)].index; atp = S[(yy == 1) & (S >= .999)].index
print('persistent FP', len(pfp), 'persistent FN', len(pfn), 'always TP', len(atp))
json.dump(list(pfp), open(CACHE / 'persistent.json', 'w'))
S.to_frame('share').to_parquet(CACHE / 'persist_share.parquet')
print('overlap persistent FP with op-point FP:', len(set(pfp) & set(o[o.grp == 'A'].index)))
oo = o.oof_repeated_dayblock.reindex(S.index); y2 = yy.copy(); y2[pfp] = 1
print('hypothetical P@R if persistent FP were positives (analysis only):', round(precision_at_recall(y2, oo), 4), 'vs', round(precision_at_recall(yy, oo), 4))
T = pd.read_parquet(CACHE / 'traits.parquet')
print('persistent FP platform', T.loc[pfp, 'pl'].value_counts().to_dict(), ' ptr_comp share', T.loc[pfp, 'ptr_comp'].mean().round(2), ' pool share', T.loc[pfp, 'pool'].mean().round(2))

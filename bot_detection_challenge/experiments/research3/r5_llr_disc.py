"""Discovery: AUC of OOF LLR aggregates (validation rows only) on P0 + fwd7 folds."""
import sys, warnings; sys.path.insert(0, 'research3'); sys.path.insert(0, 'research2'); warnings.filterwarnings('ignore')
from common import *
from rcv import Harness
from llr import llr_features
from sklearn.metrics import roc_auc_score
pd.set_option('display.width', 250)
H = Harness(n_rep=1); D = H.D
o = pd.read_parquet(CACHE / 'groups_opt.parquet')
for neg in ['all', 'hard']:
    for sch in ['P0', 'fwd7']:
        parts = []
        for k, (tri, vai) in enumerate(H.schemes[sch]):
            parts.append(llr_features(D, D.labels.iloc[tri], D.ids[vai], neg=neg, prefix=neg))
        F = pd.concat(parts); F = F[~F.index.duplicated(keep='last')]
        F.to_parquet(CACHE / f'llr_{sch}_{neg}.parquet')
        g = o.grp.reindex(F.index); yy = D.labels.reindex(F.index)
        tp = g.isin(['B', 'C', 'TP']); fp = g.isin(['A', 'FP']); near = tp & (o['rank'].reindex(F.index) > 300)
        rows = []
        for c in F.columns:
            v = F[c].fillna(F[c].median())
            rows.append((c, roc_auc_score(yy, v), roc_auc_score(tp[tp | fp], v[tp | fp]), roc_auc_score(near[near | fp], v[near | fp])))
        T = pd.DataFrame(rows, columns=['f', 'AUC_bot', 'AUC_TPvsFP', 'AUC_TPnear_vsFP'])
        T['dev_near'] = (T.AUC_TPnear_vsFP - .5).abs()
        print(f'\n== neg={neg} scheme={sch} n_val={len(F)}  (top by bot AUC, then top by near-FP AUC)')
        print(T.sort_values('AUC_bot', ascending=False).head(8).round(3).to_string(index=False))
        print(T.sort_values('dev_near', ascending=False).head(10).round(3).to_string(index=False), flush=True)

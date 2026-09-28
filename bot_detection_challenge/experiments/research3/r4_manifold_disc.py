"""C3.4/C3.5 discovery test: OOF manifold scores on P0 + fwd7 folds -> AUC(TP vs FP) and 2D generic-vs-service table.
No model is trained here: features only, strict fold/inner-leave-day-out references."""
import sys, warnings; sys.path.insert(0, 'research3'); sys.path.insert(0, 'research2'); warnings.filterwarnings('ignore')
from common import *
from rcv import Harness
from manifold import manifold_features
from sklearn.metrics import roc_auc_score
pd.set_option('display.width', 250)
H = Harness(n_rep=1)
D = H.D
o = pd.read_parquet(CACHE / 'groups_opt.parquet')
REPS = {'all': ('ev', 'tr', 'dt', 'px', 'py', 'pm', 'cat', 'pg', 'ua', 'pl', 'm'),
        'behav': ('ev', 'tr', 'dt', 'pg', 'm'), 'ptr_time': ('dt', 'px', 'py', 'pm')}
out = {}
for rn, bl in REPS.items():
    parts = []
    for k, (tri, vai) in enumerate(H.schemes['P0']):
        lab = D.labels.iloc[tri]
        F = manifold_features(D, lab, D.ids[vai], blocks=bl, prefix=rn)
        parts.append(F)
    F = pd.concat(parts)
    F.to_parquet(CACHE / f'mf_P0_{rn}.parquet')
    g = o.grp.reindex(F.index)
    rows = []
    for c in F.columns:
        v = F[c]
        tpfp = g.isin(['TP', 'A']) if False else g.isin(['TP', 'FP', 'B', 'C', 'A'])
        tp = g.isin(['TP', 'B', 'C']); fp = g.isin(['FP', 'A'])
        mk = tp | fp
        a_tpfp = roc_auc_score(tp[mk], v[mk].fillna(v.median()))
        yy = D.labels.reindex(F.index)
        a_all = roc_auc_score(yy, v.fillna(v.median()))
        # near-threshold: TP ranks 300-604 vs FP
        nb = (tp & (o['rank'].reindex(F.index) > 300)) | fp
        a_near = roc_auc_score(tp[nb], v[nb].fillna(v.median()))
        rows.append((c, round(a_all, 3), round(a_tpfp, 3), round(a_near, 3)))
    print(pd.DataFrame(rows, columns=['feature', 'AUC_bot_vs_human', 'AUC_TP_vs_FP', 'AUC_TPnear_vs_FP']).to_string(index=False), flush=True)

"""Is the TP-vs-FP signal incremental to the generic B0 score? (a) AUC of B0 OOF itself; (b) rank-matched B vs A;
(c) aux model with B0 OOF score as an extra input, LODO."""
import sys, json, warnings; sys.path.insert(0, 'research3'); warnings.filterwarnings('ignore')
from common import *
from cv import Data, POOL_COLS
from history import pool_features
from rep import build
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from catboost import CatBoostClassifier
FPSET = sys.argv[1] if len(sys.argv) > 1 else 'persistent'
D = Data(); o = pd.read_parquet(CACHE / 'groups_opt.parquet'); m = meta()
cols = [c for c in json.load(open(ROOT / 'cycle1' / 'feature_list.json')) if c not in POOL_COLS]
XB = D.F.loc[D.ids, cols].join(pool_features(D.ci, D.labels, D.ids, 7))
R = build(); R.columns = ['r_' + c for c in R.columns]
A = json.load(open(CACHE / 'persistent.json')) if FPSET == 'persistent' else list(o[o.grp == 'A'].index)
lo = o.loc[A, 'rank'].min()
for name, P in [('all flagged TP', list(o[o.grp.isin(['B', 'C'])].index)),
                ('rank-matched TP (rank>=min FP rank)', list(o[o.grp.isin(['B', 'C']) & (o['rank'] >= lo)].index)),
                ('TP rank 300-604', list(o[o.grp.isin(['B', 'C']) & (o['rank'] >= 300)].index))]:
    ids = np.array(A + P); t = np.r_[np.zeros(len(A)), np.ones(len(P))].astype(int); day = m.loc[ids, 'day'].values
    s = o.loc[ids, 'oof_repeated_dayblock'].values
    X = XB.loc[ids].join(R.loc[ids]).assign(b0=s).values.astype(float)
    res = {'B0_score_AUC': roc_auc_score(t, s)}
    for mk in ['ET', 'cat', 'ET+b0', 'cat+b0']:
        Xm = X if '+b0' in mk else X[:, :-1]
        p = np.zeros(len(t))
        for d in np.unique(day):
            tr, va = day != d, day == d
            mdl = (make_pipeline(SimpleImputer(strategy='median'), ExtraTreesClassifier(500, min_samples_leaf=3, max_features=0.3, class_weight='balanced', n_jobs=4, random_state=0))
                   if mk.startswith('ET') else CatBoostClassifier(iterations=300, depth=3, learning_rate=0.05, verbose=0, auto_class_weights='Balanced', random_seed=0, thread_count=4))
            mdl.fit(Xm[tr], t[tr]); p[va] = mdl.predict_proba(Xm[va])[:, 1]
        res[mk] = roc_auc_score(t, p)
        # blend test: does aux prob add to B0 rank within this set?
        res[mk + '_blend'] = roc_auc_score(t, pd.Series(s).rank().values + pd.Series(p).rank().values)
    print(f'{name}: nFP={len(A)} nTP={len(P)}', {k: round(v, 3) for k, v in res.items()}, flush=True)

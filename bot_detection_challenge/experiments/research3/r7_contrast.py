"""C3.7 contrastive auxiliary task: flagged TP (known services) vs FP (bot-like negatives).
Leave-one-day-out CV; permutation null; feature-set comparison. Diagnostic only (never fed to the main model)."""
import sys, json, warnings; sys.path.insert(0, 'research3'); warnings.filterwarnings('ignore')
from common import *
from cv import Data, POOL_COLS
from history import pool_features
from rep import build
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.impute import SimpleImputer
from catboost import CatBoostClassifier
FPSET = sys.argv[1] if len(sys.argv) > 1 else 'persistent'
D = Data(); o = pd.read_parquet(CACHE / 'groups_opt.parquet'); m = meta()
cols = [c for c in json.load(open(ROOT / 'cycle1' / 'feature_list.json')) if c not in POOL_COLS]
XB = D.F.loc[D.ids, cols].join(pool_features(D.ci, D.labels, D.ids, 7))  # pool: only labels of strictly earlier days
R = build(); R.columns = ['r_' + c for c in R.columns]
if FPSET == 'persistent':
    A = json.load(open(CACHE / 'persistent.json'))
else:
    A = list(o[o.grp == 'A'].index)
P = list(o[o.grp.isin(['B', 'C'])].index)
ids = np.array(A + P); t = np.r_[np.zeros(len(A)), np.ones(len(P))].astype(int)
day = m.loc[ids, 'day'].values
print(f'FP set={FPSET}: {len(A)} FP vs {len(P)} TP; days {np.unique(day)}')
SETS = {'B0feat': XB.loc[ids], 'rep': R.loc[ids], 'both': XB.loc[ids].join(R.loc[ids])}
MODELS = {
    'LR': lambda s: make_pipeline(SimpleImputer(strategy='median'), QuantileTransformer(n_quantiles=100, output_distribution='normal'), LogisticRegression(C=0.05, max_iter=2000, class_weight='balanced')),
    'tree3': lambda s: make_pipeline(SimpleImputer(strategy='median'), DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, class_weight='balanced', random_state=s)),
    'ET': lambda s: make_pipeline(SimpleImputer(strategy='median'), ExtraTreesClassifier(500, min_samples_leaf=3, max_features=0.3, class_weight='balanced', n_jobs=4, random_state=s)),
    'cat': lambda s: CatBoostClassifier(iterations=300, depth=3, learning_rate=0.05, verbose=0, auto_class_weights='Balanced', random_seed=s, thread_count=4),
}
def lodo(X, t, mk, seed=0):
    p = np.zeros(len(t))
    for d in np.unique(day):
        tr, va = day != d, day == d
        if t[tr].min() == t[tr].max(): continue
        mdl = MODELS[mk](seed); mdl.fit(X[tr], t[tr]); p[va] = mdl.predict_proba(X[va])[:, 1]
    return p
rng = np.random.RandomState(0)
res = []
for sn, X in SETS.items():
    X = X.values.astype(float)
    for mk in MODELS:
        p = lodo(X, t, mk)
        auc = roc_auc_score(t, p)
        # week split stability
        w1, w2 = day < 8, day >= 8
        null = []
        if mk in ('LR', 'cat'):
            for r in range(10):
                tp = t.copy()
                for d in np.unique(day):  # permute within day
                    ix = np.flatnonzero(day == d); tp[ix] = rng.permutation(tp[ix])
                null.append(roc_auc_score(tp, lodo(X, tp, mk, r)))
        res.append((sn, mk, round(auc, 3), round(roc_auc_score(t[w1], p[w1]), 3), round(roc_auc_score(t[w2], p[w2]), 3),
                    round(np.mean(null), 3) if null else np.nan, round(np.std(null), 3) if null else np.nan))
        print(res[-1], flush=True)
T = pd.DataFrame(res, columns=['features', 'model', 'AUC_LODO', 'AUC_days<8', 'AUC_days>=8', 'null_mean', 'null_std'])
print(T.to_string(index=False)); T.to_csv(R3 / f'r7_contrast_{FPSET}.csv', index=False)
# interpretability: top LR coefficients / cat importances on 'both'
X = SETS['both']
mdl = MODELS['cat'](0).fit(X.values.astype(float), t)
imp = pd.Series(mdl.get_feature_importance(), index=X.columns).sort_values(ascending=False)
print('\ncat importances (TP=1 vs FP=0) top 20'); print(imp.head(20).round(2).to_string())
top = imp.index[:12]
print(X.groupby(t)[list(top)].median().T.rename(columns={0: 'FP_med', 1: 'TP_med'}).round(3))

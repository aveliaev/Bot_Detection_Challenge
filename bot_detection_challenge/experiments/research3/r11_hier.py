"""C3.11 hierarchical and multiclass service models (custom fit_fn inside the repeated-CV harness).

E11_multi : pseudo-service labels = KMeans(k) on the fold's TRAINING positives (service representation);
            CatBoost MultiClass (0 = human, 1..k = subtypes); score = 1 - P(human).
E11_hier  : Stage A = B0 CatBoost on the fold's training rows. Stage B = small CatBoost trained only on
            'automation-like' training rows (top 15% of an inner leave-day-out LGB score, fit on training labels)
            to separate known services from bot-like humans; final = logit(rankA) + 0.5*logit(P_B)."""
import sys, warnings; sys.path.insert(0, 'research3'); sys.path.insert(0, 'research2'); warnings.filterwarnings('ignore')
from rcv import *
import rcv
rcv.OOF2 = Path(__file__).resolve().parent / 'oof'; rcv.LOG = Path(__file__).resolve().parent / 'experiments3.jsonl'
from rep import build, block_cols
from common import meta
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from catboost import CatBoostClassifier
import lightgbm as lgb
from scipy.special import logit
R = build(); M = meta()
RC = block_cols(R, ['ev', 'tr', 'dt', 'px', 'py', 'pm', 'cat', 'pg', 'ua', 'pl', 'm'])
H = Harness()
D = H.D

def multi_fit(K):
    def fit(Xtr, ytr, Xva, seed, tri, vai, sch, k):
        pos = Xtr.index[ytr == 1]
        sc = StandardScaler().fit(R.loc[pos, RC]); pca = PCA(15, random_state=0).fit(sc.transform(R.loc[pos, RC]))
        km = KMeans(K, n_init=10, random_state=seed).fit(pca.transform(sc.transform(R.loc[pos, RC])))
        yy = np.zeros(len(ytr), int); yy[ytr == 1] = km.labels_ + 1
        m = CatBoostClassifier(iterations=1500, learning_rate=0.03, depth=6, l2_leaf_reg=5, rsm=0.5, random_seed=seed,
                               verbose=0, thread_count=4, loss_function='MultiClass').fit(Xtr, yy)
        return 1 - m.predict_proba(Xva)[:, 0]
    return fit

def hier_fit(Xtr, ytr, Xva, seed, tri, vai, sch, k):
    from cv import make_model
    A = make_model('cat', None, seed).fit(Xtr, ytr)
    pa = A.predict_proba(Xva)[:, 1]
    day = D.day[tri]; s = np.zeros(len(ytr))
    for d in np.unique(day):   # inner leave-day-out generic score for training rows
        a, b = day != d, day == d
        s[b] = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=30, subsample=0.8,
                                  subsample_freq=1, colsample_bytree=0.5, verbose=-1, n_jobs=4, random_state=seed).fit(Xtr[a], ytr[a]).predict_proba(Xtr[b])[:, 1]
    hard = s >= np.quantile(s, 0.85)
    Xb = Xtr[hard].join(R.loc[Xtr.index[hard], RC].add_prefix('r_'))
    Bm = CatBoostClassifier(iterations=400, depth=3, learning_rate=0.05, l2_leaf_reg=10, random_seed=seed, verbose=0, thread_count=4).fit(Xb, ytr[hard])
    pb = Bm.predict_proba(Xva.join(R.loc[Xva.index, RC].add_prefix('r_')))[:, 1]
    ra = pd.Series(pa).rank(pct=True).values.clip(1e-4, 1 - 1e-4)
    return logit(ra) + 0.5 * logit(pb.clip(1e-4, 1 - 1e-4))

for name in sys.argv[1:]:
    if name == 'E11_multi4':
        H.run(name, fit_fn=multi_fit(4), hypothesis='C3.11 multiclass pseudo-service labels k=4', leak_risk='low (clusters from training positives)')
    if name == 'E11_multi6':
        H.run(name, fit_fn=multi_fit(6), hypothesis='C3.11 multiclass pseudo-service labels k=6', leak_risk='low')
    if name == 'E11_hier':
        H.run(name, fit_fn=hier_fit, hypothesis='C3.11 hierarchical: generic x P(known service | automation)', leak_risk='medium (nested inner LODO)')

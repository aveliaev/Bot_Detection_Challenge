"""C3.14 adversarial validation of the new families in the FINAL setting (train rows: inner leave-day-out references,
test rows: all train labels). LightGBM train-vs-test, 5-fold CV AUC. Reference: B0 features without pool = 0.51 (cycle 1)."""
import sys, warnings; sys.path.insert(0, 'research3'); sys.path.insert(0, 'research2'); warnings.filterwarnings('ignore')
from common import *
from cv import Data
from llr import llr_features
from manifold import manifold_features
from uate import ua_te_features
from history import pool_features
import lightgbm as lgb
from sklearn.model_selection import cross_val_score, StratifiedKFold
D = Data(); te = D.te.cookie_id.values; tr = D.ids
allids = np.r_[tr, te]; yadv = np.r_[np.zeros(len(tr)), np.ones(len(te))]
AG = ['mean', 'q75', 'sum', 'fhi']
fams = {
    'pool (reference)': lambda: pool_features(D.ci, D.labels, allids, 7),
    'llrhard': lambda: (lambda F: F[[c for c in F.columns if c.split('_')[-1] in AG]])(llr_features(D, D.labels, allids, fams=['tdt', 'pg', 'm3dt'], neg='hard', prefix='lh')),
    'llrall': lambda: (lambda F: F[[c for c in F.columns if c.split('_')[-1] in AG]])(llr_features(D, D.labels, allids, fams=['tdt', 'pg', 'm3dt'], neg='all', prefix='la')),
    'manifold': lambda: manifold_features(D, D.labels, allids, which=('knn', 'gmm', 'maha', 'proto')),
    'uate': lambda: ua_te_features(D, D.labels, allids),
}
only = sys.argv[1:] or list(fams)
for name in only:
    F = fams[name]().reindex(allids)
    auc = cross_val_score(lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, verbose=-1, n_jobs=4),
                          F, yadv, cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring='roc_auc').mean()
    print(f'{name:18s} adversarial AUC train-vs-test = {auc:.3f}   n_features={F.shape[1]}', flush=True)

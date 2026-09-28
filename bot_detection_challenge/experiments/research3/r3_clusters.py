"""C3.3 service clustering of known bots (train target=1). Stability across weeks + separation from FP."""
import sys, warnings; sys.path.insert(0, 'research3'); warnings.filterwarnings('ignore')
from common import *
from rep import build, block_cols
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
pd.set_option('display.width', 250)
R = build(); m = meta(); o = pd.read_parquet(CACHE / 'groups_opt.parquet')
pos = m[(m.target == 1)].index
wk = (m.loc[pos, 'day'] >= 7).values
REPS = {'all': ['ev', 'tr', 'dt', 'px', 'py', 'pm', 'cat', 'pg', 'ua', 'pl', 'm'],
        'behav': ['ev', 'tr', 'dt', 'pg', 'm'],
        'seq_time': ['ev', 'tr', 'dt'],
        'tech': ['ua', 'pl', 'px', 'py', 'pm', 'm'],
        'no_cat': ['ev', 'tr', 'dt', 'px', 'py', 'pm', 'pg', 'ua', 'pl', 'm']}
res = []
best = {}
for rn, bl in REPS.items():
    cols = block_cols(R, bl)
    sc = StandardScaler().fit(R.loc[pos, cols])
    Z = PCA(n_components=min(15, len(cols)), random_state=0).fit_transform(sc.transform(R.loc[pos, cols]))
    for k in [2, 3, 4, 5, 6, 8]:
        for alg in ['km', 'gmm', 'agg']:
            if alg == 'km':
                f = lambda Zs: KMeans(k, n_init=10, random_state=0).fit(Zs)
                lab = f(Z).labels_
                m1 = f(Z[~wk]); m2 = f(Z[wk])
                ari = adjusted_rand_score(m1.predict(Z[wk]), m2.labels_)
            elif alg == 'gmm':
                f = lambda Zs: GaussianMixture(k, covariance_type='diag', random_state=0, n_init=3).fit(Zs)
                lab = f(Z).predict(Z); m1 = f(Z[~wk]); m2 = f(Z[wk])
                ari = adjusted_rand_score(m1.predict(Z[wk]), m2.predict(Z[wk]))
            else:
                lab = AgglomerativeClustering(k, linkage='ward').fit_predict(Z); ari = np.nan
            sil = silhouette_score(Z, lab)
            sh = pd.crosstab(lab, wk, normalize='columns')
            tvd = 0.5 * (sh[False] - sh[True]).abs().sum()
            res.append((rn, alg, k, round(sil, 3), round(ari, 3), round(tvd, 3), np.bincount(lab).min()))
            best[(rn, alg, k)] = lab
    h = HDBSCAN(min_cluster_size=20).fit(Z)
    lab = h.labels_
    res.append((rn, 'hdbscan', int(lab.max() + 1), round(silhouette_score(Z[lab >= 0], lab[lab >= 0]), 3) if lab.max() > 0 else np.nan, np.nan, np.nan, int((lab < 0).sum())))
T = pd.DataFrame(res, columns=['rep', 'alg', 'k', 'silhouette', 'ARI_week1_vs_week2', 'TVD_share_weeks', 'min_size_or_noise'])
print(T.to_string(index=False))
T.to_csv(R3 / 'r3_clusters.csv', index=False)
import pickle; pickle.dump(best, open(CACHE / 'clusters.pkl', 'wb'))

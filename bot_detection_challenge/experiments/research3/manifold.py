"""C3.4/C3.5 positive-manifold & prototype scores as a fold-aware feature family.

manifold_features(D, lab, ids): lab = labels of the fold's training part (cookie_id -> 0/1).
  * rows whose cookie is in `lab` (training rows) get scores from references built on the OTHER
    labelled days only (inner leave-one-day-out)  -> no row ever sees its own label;
  * all other rows (validation / test) get scores from references built on all of `lab`.
Scores (representation = standardised service representation, rep.py):
  knn_pos   mean dist to 10 nearest known bots        knn_neg  mean dist to 10 nearest labelled humans
  knn_lr    log(knn_pos/knn_neg)                      iso_pos  IsolationForest (fit on bots) score
  gmm_pos   GMM log-density under bots (diag, 6 comp) gmm_neg  same under humans;  gmm_lr = difference
  maha_pos  Mahalanobis (Ledoit-Wolf) to bot mean     proto_d{k}, proto_min  distance to KMeans(6) bot prototypes
"""
from __future__ import annotations
import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.covariance import LedoitWolf
from sklearn.cluster import KMeans
from rep import build, block_cols
from common import meta

_R = None; _M = None


def _scores(Zref_pos, Zref_neg, Zq, seed=0, which=('knn', 'iso', 'gmm', 'maha', 'proto')):
    out = {}
    if 'knn' in which:
        k = 10
        dp = NearestNeighbors(n_neighbors=k).fit(Zref_pos).kneighbors(Zq)[0].mean(1)
        dn = NearestNeighbors(n_neighbors=k).fit(Zref_neg).kneighbors(Zq)[0].mean(1)
        out.update(knn_pos=dp, knn_neg=dn, knn_lr=np.log(dp + 1e-6) - np.log(dn + 1e-6))
    if 'iso' in which:
        out['iso_pos'] = IsolationForest(n_estimators=200, random_state=seed).fit(Zref_pos).score_samples(Zq)
    if 'gmm' in which:
        gp = GaussianMixture(6, covariance_type='diag', random_state=seed, reg_covar=1e-3).fit(Zref_pos)
        gn = GaussianMixture(6, covariance_type='diag', random_state=seed, reg_covar=1e-3).fit(Zref_neg)
        a, b = gp.score_samples(Zq), gn.score_samples(Zq)
        out.update(gmm_pos=a, gmm_neg=b, gmm_lr=a - b)
    if 'maha' in which:
        lw = LedoitWolf().fit(Zref_pos)
        out['maha_pos'] = np.sqrt(lw.mahalanobis(Zq))
    if 'proto' in which:
        km = KMeans(6, n_init=10, random_state=seed).fit(Zref_pos)
        d = km.transform(Zq)
        for j in range(d.shape[1]):
            pass
        ds = np.sort(d, axis=1)  # sorted distances: prototype identity is not stable across refits
        out.update(proto_min=ds[:, 0], proto_2nd=ds[:, 1], proto_gap=ds[:, 1] - ds[:, 0])
    return out


def manifold_features(D, lab, ids, blocks=('ev', 'tr', 'dt', 'px', 'py', 'pm', 'cat', 'pg', 'ua', 'pl', 'm'),
                      which=('knn', 'iso', 'gmm', 'maha', 'proto'), n_pca=20, prefix='mf', neg_sample=3000):
    global _R, _M
    if _R is None:
        _R = build(); _M = meta()
    cols = block_cols(_R, list(blocks))
    ids = pd.Index(ids)
    lab = lab.astype(int)
    labday = _M.loc[lab.index, 'day']
    # scaler/PCA are unsupervised; fit on the labelled training part only
    sc = StandardScaler().fit(_R.loc[lab.index, cols])
    pca = PCA(n_components=min(n_pca, len(cols)), random_state=0).fit(sc.transform(_R.loc[lab.index, cols]))
    Z = lambda idx: pca.transform(sc.transform(_R.loc[idx, cols]))
    rng = np.random.RandomState(0)
    out = []
    in_lab = ids.isin(lab.index)
    # (1) rows outside the labelled part: references = all of lab
    q = ids[~in_lab]
    if len(q):
        P = lab.index[lab == 1]; N = lab.index[lab == 0]
        N = N[rng.permutation(len(N))[:neg_sample]]
        out.append(pd.DataFrame(_scores(Z(P), Z(N), Z(q)), index=q))
    # (2) labelled rows: leave-one-day-out references
    for d in np.sort(labday.unique()):
        q = ids[in_lab][labday.reindex(ids[in_lab]).values == d]
        if not len(q):
            continue
        ref = lab[labday != d]
        P = ref.index[ref == 1]; N = ref.index[ref == 0]
        N = N[rng.permutation(len(N))[:neg_sample]]
        out.append(pd.DataFrame(_scores(Z(P), Z(N), Z(q)), index=q))
    F = pd.concat(out).reindex(ids)
    F.columns = [f'{prefix}_{c}' for c in F.columns]
    F.index.name = 'cookie_id'
    return F

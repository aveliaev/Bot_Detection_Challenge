"""C7.5 cheap local-window models (RandomForest = Bombay transfer, CatBoost), strict OOF on the B0 folds.
Per outer fold (scheme, k):
  train : random windows of the outer TRAINING cookies (label = cookie target, weight 1/K per cookie)
  val   : deterministic windows of the validation cookies scored by the outer models (rf, cat)
  inner : deterministic windows of the training cookies scored by CatBoost fitted on the other 3 of 4 day-groups
          of the training part (for strict-OOF integration features only)
Controls: order-shuffled validation windows scored by the outer models (P0, fwd7); label permutation (P0 fold 0).
Output: research7/oof/winscores_<scheme>_k<k>.parquet (window id, part, rf, cat, rf_shuf, cat_shuf)."""
import sys, time, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
R7 = Path(__file__).resolve().parent; ROOT = R7.parent
sys.path.insert(0, str(ROOT / 'research2'))
import numpy as np, pandas as pd
from rcv import Harness
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

W = pd.read_parquet(R7 / 'cache' / 'windows.parquet')
WS = pd.read_parquet(R7 / 'cache' / 'windows_shuffled.parquet')
FEAT = [c for c in W.columns if c.startswith('w_')]
rf = lambda: RandomForestClassifier(300, min_samples_leaf=5, max_features='sqrt', n_jobs=4, random_state=0)
cb = lambda: CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05, random_seed=0, verbose=0, thread_count=4)


def fit(model, cookies, labels, perm=False):
    T = W[(W.kind == 'train') & W.cookie_id.isin(cookies)]
    y = T.cookie_id.map(labels).values
    if perm:
        y = T.cookie_id.map(pd.Series(np.random.RandomState(1).permutation(labels.values), index=labels.index)).values
    w = 1.0 / T.groupby('cookie_id').cookie_id.transform('size').values
    return model.fit(T[FEAT].fillna(-1), y, sample_weight=w)


if __name__ == '__main__':
    t0 = time.time()
    H = Harness(); D = H.D
    Winf = W[W.kind == 'infer']
    for sch, folds in H.schemes.items():
        for k, (tri, vai) in enumerate(folds):
            trc, vac = D.ids[tri], D.ids[vai]
            lab = D.labels
            Vw = Winf[Winf.cookie_id.isin(set(vac))]
            m_rf, m_cb = fit(rf(), set(trc), lab), fit(cb(), set(trc), lab)
            out = pd.DataFrame({'part': 'val', 'rf': m_rf.predict_proba(Vw[FEAT].fillna(-1))[:, 1],
                                'cat': m_cb.predict_proba(Vw[FEAT].fillna(-1))[:, 1]}, index=Vw.index)
            if sch in ('P0', 'fwd7'):
                Sw = WS.loc[Vw.index]
                out['rf_shuf'] = m_rf.predict_proba(Sw[FEAT].fillna(-1))[:, 1]; out['cat_shuf'] = m_cb.predict_proba(Sw[FEAT].fillna(-1))[:, 1]
            if sch == 'P0' and k == 0:   # label-permutation sanity check
                mp = fit(cb(), set(trc), lab, perm=True)
                ps = pd.Series(mp.predict_proba(Vw[FEAT].fillna(-1))[:, 1], index=Vw.index).groupby(Vw.cookie_id).mean()
                print('label-permutation control, cookie AUC of window-mean score:', round(roc_auc_score(lab[ps.index], ps), 3), flush=True)
            # inner OOF for the training cookies (4 contiguous day groups of the training part)
            days = np.unique(D.day[tri]); groups = np.array_split(days, 4)
            parts = []
            for g in groups:
                gin = set(D.ids[tri][np.isin(D.day[tri], g)]); gout = set(trc) - gin
                m = fit(cb(), gout, lab)
                Iw = Winf[Winf.cookie_id.isin(gin)]
                parts.append(pd.DataFrame({'part': 'train', 'cat': m.predict_proba(Iw[FEAT].fillna(-1))[:, 1]}, index=Iw.index))
            pd.concat([out] + parts).to_parquet(R7 / 'oof' / f'winscores_{sch}_k{k}.parquet')
        print(sch, 'done', round(time.time() - t0), 's', flush=True)

"""Cycle 6.5 STRICT OOF residuals on the B0 folds (P0..P4 + fwd7) and mapping-quality / negative-control tables.
For outer fold k:
  validation rows : mapping fitted on the fold's training rows            -> residual
  training rows   : inner leave-one-day-out over the training days          -> residual (never in-sample)
The item pool inside the representation is the fold's pool frame (training labels, strictly earlier days), as in B0.
Output: research65/oof/resid_<scheme>_k<k>.parquet (column 'part' = train/val), research65/mapping_quality.csv."""
import sys, time, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
R65 = Path(__file__).resolve().parent; ROOT = R65.parent
sys.path.insert(0, str(R65)); sys.path.insert(0, str(ROOT / 'research2')); sys.path.insert(0, str(ROOT / 'cycle1'))
import numpy as np, pandas as pd
from rcv import Harness
from representations import build
from cross_modal import Mapping, MAPS, summarize, quality, shuffled_source

RESID_MAPS = ['MB', 'IB', 'BI', 'MIB']


def resid_block(R, fit_idx, pred_idx):
    out = []
    for name in RESID_MAPS:
        mp = Mapping(name).fit(R, fit_idx)
        out.append(summarize(name, mp.residual(R, pred_idx)))
    return pd.concat(out, axis=1).reindex(pred_idx)


if __name__ == '__main__':
    t0 = time.time()
    H = Harness(); D = H.D
    rng = np.random.RandomState(0)
    Q = []
    for sch, folds in H.schemes.items():
        for k, (tri, vai) in enumerate(folds):
            pool = H.fam_frame('pool', sch, k)
            R = build(pool, D.ids)
            ti, vi = pd.Index(D.ids[tri]), pd.Index(D.ids[vai])
            # mapping quality + negative controls (validation rows of this fold)
            if sch in ('P0', 'fwd7'):
                for name in MAPS:
                    mp = Mapping(name).fit(R, ti)
                    q = quality(mp, R, vi).assign(mapping=name, scheme=sch, fold=k, control='none')
                    Q.append(q.rename_axis('dim').reset_index())
                    if name not in ('CB', 'CI', 'CB_m'):
                        Xs = shuffled_source(mp, R, vi, rng)
                        q = quality(mp, R, vi, X_override=Xs).assign(mapping=name, scheme=sch, fold=k, control='shuffled_source')
                        Q.append(q.rename_axis('dim').reset_index())
            V = resid_block(R, ti, vi).assign(part='val')
            days = D.day[tri]
            Tparts = [resid_block(R, pd.Index(D.ids[tri[days != d]]), pd.Index(D.ids[tri[days == d]])) for d in np.unique(days)]
            T = pd.concat(Tparts).assign(part='train')
            pd.concat([T, V]).to_parquet(R65 / 'oof' / f'resid_{sch}_k{k}.parquet')
        print(sch, 'done', round(time.time() - t0), 's', flush=True)
    pd.concat(Q).to_csv(R65 / 'mapping_quality.csv', index=False)

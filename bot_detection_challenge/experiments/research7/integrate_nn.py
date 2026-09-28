"""C7.15 B0 + neural sequence features (chosen arch), same CatBoost as B0.
Training-row features must be OOF:
  P0..P4 : the training cookies' own OUTER OOF window scores of the same partition (every training cookie is a validation
           cookie of another fold) -> cross-fitted 2-level stacking (those models saw fold-k labels: mild optimism, flagged)
  fwd7   : NESTED inner models on the training days only (leave-one-day-out over days 2..k-1, early stopping on the last
           2 remaining days) -> no validation label reaches a training feature.
Usage: python research7/integrate_nn.py   (runs the locked ablations E..I)"""
import sys, time, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
R7 = Path(__file__).resolve().parent; ROOT = R7.parent
sys.path.insert(0, str(R7)); sys.path.insert(0, str(ROOT / 'research2')); sys.path.insert(0, str(ROOT / 'research6'))
import numpy as np, pandas as pd, torch
from rcv import Harness
import rcv
from cv import make_model
from build_seq_features import aggregate, LOCKED
from build_windows import tokens
from sequence_model import token_matrix, train_model, predict, CONT_GROUPS
from train_sequence import spans_for
torch.set_num_threads(4)
ARCH = open(R7 / 'cache' / 'chosen_arch.txt').read().strip()
SETS = {'E7nn_seq_full': ['seq_full'], 'E7nn_chunk_mean': ['chunk_mean'],
        'E7nn_chunk_dist': ['chunk_mean', 'chunk_std', 'chunk_q10', 'chunk_median', 'chunk_q90', 'chunk_min', 'chunk_max', 'chunk_range'],
        'E7nn_len_specific': ['len3_mean', 'len5_mean', 'len10_mean', 'len20_mean', 'len40_mean', 'len5_std', 'len10_std', 'len20_std'],
        'E7nn_locked': LOCKED}


def outer_scores(s, ks):
    t = pd.concat([pd.read_parquet(R7 / 'oof' / f'nn_{ARCH}_{s}_k{k}.parquet') for k in ks])
    return t[~t.index.duplicated(keep='last')].score


def inner_fwd7(H, k):
    p = R7 / 'oof' / f'nn_{ARCH}_fwd7inner_k{k}.parquet'
    if p.exists(): return pd.read_parquet(p).score
    D = H.D; lab = D.labels
    E = tokens(); EV, XC, off, n, names = token_matrix(E); cols = list(range(len(names)))
    W = pd.read_parquet(R7 / 'cache' / 'windows.parquet', columns=['cookie_id', 'start', 'end', 'L', 'kind']); Winf = W[W.kind == 'infer']
    parts = []
    for d in range(2, k):
        tr = D.ids[(D.day < k) & (D.day != d)]; days = D.day[(D.day < k) & (D.day != d)]
        es_days = np.sort(np.unique(days))[-2:]
        net = train_model(tr[~np.isin(days, es_days)], tr[np.isin(days, es_days)], lab, EV, XC, off, n, cols, ARCH == 'bilstm', seed=0)
        vw = Winf[Winf.cookie_id.isin(set(D.ids[D.day == d]))]
        parts.append(pd.DataFrame({'score': predict(net, EV, XC, spans_for(vw, off, n), cols)}, index=vw.index))
    out = pd.concat(parts); out.to_parquet(p)
    return out.score


if __name__ == '__main__':
    t0 = time.time()
    rcv.OOF2 = ROOT / 'research6' / 'oof'; rcv.LOG = R7 / 'experiments7.jsonl'
    H = Harness()
    FE = {s: aggregate(outer_scores(s, range(6))) for s in ['P0', 'P1', 'P2', 'P3', 'P4']}
    FE['fwd7'] = aggregate(outer_scores('fwd7', range(2)))
    INN = {k: aggregate(inner_fwd7(H, k)) for k in (5, 7)}
    print('features ready', round(time.time() - t0), 's', flush=True)
    sys.argv = ['x']
    from r6_integrate import report
    for name, feats in SETS.items():
        def fit_fn(Xtr, ytr, Xva, seed, tri, vai, sch, k, feats=feats):
            Ftr = INN[(5, 7)[k]] if sch == 'fwd7' else FE[sch]
            A = Xtr.join(Ftr[feats].add_prefix('nn_')); B = Xva.join(FE[sch][feats].add_prefix('nn_'))
            return make_model('cat', None, seed).fit(A, ytr).predict_proba(B[A.columns])[:, 1]
        H.run(name, fit_fn=fit_fn, hypothesis=f'C7 B0 + {ARCH} {feats}', leak_risk='P: cross-fit stacking; fwd7: nested')
        report(name)

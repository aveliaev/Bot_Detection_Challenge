"""C7.10-C7.14 runner: LSTM / BiLSTM (all folds) + ablations and controls (P0 + fwd7). Window scores -> research7/oof/nn_*.parquet
Usage: python research7/train_sequence.py main|ablation [schemes...]"""
import sys, time, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
R7 = Path(__file__).resolve().parent; ROOT = R7.parent
sys.path.insert(0, str(R7)); sys.path.insert(0, str(ROOT / 'research2')); sys.path.insert(0, str(ROOT / 'research3'))
import numpy as np, pandas as pd, torch
from rcv import Harness
from common import events
from build_windows import tokens, shuffle_order
from sequence_model import token_matrix, train_model, predict, CONT_GROUPS, MAXLEN
torch.set_num_threads(4)


def pointer_shuffled_tokens(seed=2):
    rng = np.random.RandomState(seed); ev = events().copy()
    hp = ev.pointer_x.notna()
    for c, idx in ev[hp].groupby('cookie_id').groups.items():
        idx = np.asarray(idx); p = rng.permutation(len(idx))
        ev.loc[idx, ['pointer_x', 'pointer_y']] = ev.loc[idx[p], ['pointer_x', 'pointer_y']].values
    return tokens(ev)


def spans_for(Wg, off, n):
    s = Wg.cookie_id.map(off).values; st, en, L = Wg.start.values, Wg.end.values, Wg.L.values
    N = Wg.cookie_id.map(n).values
    mid = np.maximum(0, N - MAXLEN) // 2
    a = np.where(L == -1, s + mid, s + st); b = np.where(L == -1, s + mid + np.minimum(N, MAXLEN), s + en)
    return list(zip(a, b))


if __name__ == '__main__':
    mode = sys.argv[1]; only = sys.argv[2:]
    t0 = time.time()
    H = Harness(); D = H.D; lab = D.labels
    E = tokens(); EV, XC, off, n, names = token_matrix(E)
    col = {g: [names.index(c) for c in cs] for g, cs in CONT_GROUPS.items()}
    ALL = sorted(sum(col.values(), []))
    W = pd.read_parquet(R7 / 'cache' / 'windows.parquet', columns=['cookie_id', 'start', 'end', 'L', 'kind'])
    Winf = W[W.kind == 'infer']
    if mode == 'main':
        CFG = {'lstm': dict(bidir=False, cols=ALL, use_event=True), 'bilstm': dict(bidir=True, cols=ALL, use_event=True)}
        schemes = only or list(H.schemes)
        ctrl = {}
    else:
        CFG = {'abl_full': dict(bidir=False, cols=ALL, use_event=True),        # = main LSTM (same seed/data) -> controls for the full model
               'abl_event': dict(bidir=False, cols=[], use_event=True),
               'abl_event_timing': dict(bidir=False, cols=sorted(col['timing'] + col['platform']), use_event=True),
               'abl_event_timing_ptr': dict(bidir=False, cols=sorted(col['timing'] + col['platform'] + col['pointer']), use_event=True),
               'abl_pointer_only': dict(bidir=False, cols=sorted(col['pointer'] + [names.index('ldt'), names.index('miss_dt')]), use_event=False, mouse_only=True)}
        schemes = only or ['P0', 'fwd7']
        Es = shuffle_order(E); EVs, XCs, _, _, _ = token_matrix(Es)
        Ep = pointer_shuffled_tokens(); EVp, XCp, _, _, _ = token_matrix(Ep)
        ctrl = {'order_shuffled': (EVs, XCs), 'pointer_shuffled': (EVp, XCp)}
    F = pd.read_parquet(ROOT / 'work' / 'features.parquet')
    has_mouse = set(F.index[F.ptr_n.fillna(0) > 0])
    for sch in schemes:
        for k, (tri, vai) in enumerate(H.schemes[sch]):
            trc = D.ids[tri]; days = D.day[tri]; es_days = np.sort(np.unique(days))[-2:]
            fit_c = trc[~np.isin(days, es_days)]; es_c = trc[np.isin(days, es_days)]
            Vw = Winf[Winf.cookie_id.isin(set(D.ids[vai]))]
            for name, cfg in CFG.items():
                fc, ec, vw = fit_c, es_c, Vw
                if cfg.get('mouse_only'):
                    fc = np.array([c for c in fit_c if c in has_mouse]); ec = np.array([c for c in es_c if c in has_mouse])
                    vw = Vw[Vw.cookie_id.isin(has_mouse)]
                logs = []
                net = train_model(fc, ec, lab, EV, XC, off, n, cfg['cols'], cfg['bidir'], seed=0, use_event=cfg['use_event'], log=logs)
                sp = spans_for(vw, off, n)
                out = pd.DataFrame({'score': predict(net, EV, XC, sp, cfg['cols'])}, index=vw.index)
                if mode == 'main' and sch in ('P0', 'fwd7') or mode != 'main':
                    for cname, (EVc, XCc) in (ctrl.items() if mode != 'main' else []):
                        out[cname] = predict(net, EVc, XCc, sp, cfg['cols'])
                out.attrs['epochs'] = len(logs)
                out.to_parquet(R7 / 'oof' / f'nn_{name}_{sch}_k{k}.parquet')
                print(sch, k, name, 'epochs', len(logs), 'best inner AUC', round(max(a for _, a in logs), 4), round(time.time() - t0), 's', flush=True)

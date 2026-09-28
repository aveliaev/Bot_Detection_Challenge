"""Cycle 5 evaluation. For every outer fold of P0..P4 and fwd7:
  * every pre-specified config is trained on the training part only (OOF rows of the other folds; fwd7: nested inner OOF
    of the training days) and applied to the outer fold  -> 'fixed-config' results (reported, NOT used for selection);
  * NESTED: the config (method, band, delta/eps, lambda / rule thresholds and shift, or 'none') is chosen by inner
    cross-validation on the training part only (inner folds = the other folds / training days), then applied once.
Outer folds are used only for the final numbers."""
import sys, time, json, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from local import *
from joblib import Parallel, delayed

CFGS = configs()


def all_scores(TR, tr, AP, ap):
    """{(cfg, strength): new scores for ap}"""
    out = {}
    for cfg in CFGS:
        c = corrections(cfg, TR, tr, AP, ap)
        for st in strengths(cfg):
            out[(cfg, st)] = new_score(AP, ap, cfg, c, st)
    return out


def select(TR, tr_rows, inner_fold):
    """inner CV on the training population rows tr_rows grouped by inner_fold -> best (cfg, strength) by inner P@R."""
    acc = {}
    for f in np.unique(inner_fold):
        itr, iva = tr_rows[inner_fold != f], tr_rows[inner_fold == f]
        for key, s in all_scores(TR, itr, TR, iva).items():
            acc.setdefault(key, []).append((iva, s))
    best, bkey, base = -1, (('none',), 0), None
    for key, parts in acc.items():
        rows = np.concatenate([p[0] for p in parts]); s = np.concatenate([p[1] for p in parts])
        v = pr(TR.y[rows], s)
        if key[0] == ('none',): base = v
        if v > best + 1e-12: best, bkey = v, key
    if best <= base + 1e-12: bkey = (('none',), 0)
    return bkey, best, base


def outer_job(sch, k):
    t0 = time.time()
    if sch.startswith('P'):
        E = load(sch); mask = valid_mask(E)
        fold = np.full(len(Y), -1)
        for j, (_, vai) in enumerate(folds_of(sch)): fold[vai] = j
        R = Rows(E, mask, fold)
        TR, tr = R, np.flatnonzero(R.fold != k); AP, ap = R, np.flatnonzero(R.fold == k)
        inner_fold = R.fold[tr]
    else:
        E = load('fwd7'); mask = valid_mask(E)
        fold = np.where(DAY >= 7, 7, np.where(DAY >= 5, 5, -1))
        AP = Rows(E, mask, fold); ap = np.flatnonzero(AP.fold == k)
        I = dict(np.load(R4 / 'oof' / f'experts_fwd7inner_k{k}.npz')); im = ~np.isnan(I['B0'])
        TR = Rows(I, im, DAY); tr = np.arange(TR.n); inner_fold = TR.fold
    fixed = all_scores(TR, tr, AP, ap)
    key, ib, ibase = select(TR, tr, inner_fold)
    return dict(sch=sch, k=k, ap_idx=AP.idx[ap], fixed=fixed, chosen=key, inner_best=ib, inner_base=ibase, sec=time.time() - t0)


if __name__ == '__main__':
    jobs = [(s, k) for s in SCHEMES[:5] for k in range(6)] + [('fwd7', 5), ('fwd7', 7)]
    res = Parallel(n_jobs=4, verbose=5)(delayed(outer_job)(s, k) for s, k in jobs)
    import pickle
    pickle.dump(res, open(R5 / 'cache_r5.pkl', 'wb'))
    print('done', sum(r['sec'] for r in res))

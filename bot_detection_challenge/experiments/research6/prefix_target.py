"""C6.7 prefix target models: the SAME B0 model (CatBoost, top-120 + item pool, seed 0) trained on features computed from
the first 25 / 50 / 75 % of each cookie's in-window events (P100 = B0 itself), strict OOF on the B0 folds.

Suffix isolation: features of fraction f are built by solution/features.build_features on the truncated event table
(first ceil(f*n) events of every cookie; nothing after them). Item-list-dependent inputs use the PREFIX items of the
target cookie; their reference side stays as in B0 (pool: labelled cookies of strictly earlier days; popularity:
other cookies' in-window items of days [D-6, D], unlabelled). Labels come only from the fold's training part.
Output: research6/oof/prefix_<scheme>.npz with raw probabilities P25, P50, P75, P100 (NaN outside validation rows)."""
import sys, json, time, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
R6 = Path(__file__).resolve().parent; ROOT = R6.parent
sys.path.insert(0, str(ROOT / 'cycle1')); sys.path.insert(0, str(ROOT / 'research2'))
import numpy as np, pandas as pd
from features import load_meta, load_events, build_features
from history import cookie_items, DAY0
from cv import make_model, POOL_COLS
from rcv import Harness, BASE_COLS

FRACS = (25, 50, 75)


def truncate(ev, f):
    pos = ev.groupby('cookie_id').cumcount()
    n = ev.cookie_id.map(ev.groupby('cookie_id').size())
    return ev[pos < np.ceil(f / 100 * n)].copy()


def popularity_prefix(ci_full, ci_tgt, span=7):
    out = []
    for D in np.sort(ci_tgt.day.unique()):
        cur = ci_tgt[ci_tgt.day == D]
        hist = ci_full[(ci_full.day <= D) & (ci_full.day > D - span)]
        cnt = hist.groupby('item_id').size()
        c = cur.item_id.map(cnt).values - 1          # the cookie's own (full) view of the item is excluded
        out.append(pd.DataFrame({'cookie_id': cur.cookie_id.values, 'pop': c.astype(float)}))
    p = pd.concat(out)
    return pd.DataFrame({'pop_mean': p.groupby('cookie_id')['pop'].mean(), 'pop_logmean': np.log1p(p['pop']).groupby(p.cookie_id).mean()})


def pool_prefix(ci_full, ci_tgt, labels, target_cookies, hist_days=7, m=2.0, p0=0.08):
    """history.pool_features with history items = full lists of labelled earlier-day cookies, target items = prefix."""
    lab = ci_full[ci_full.cookie_id.isin(labels.index)].copy()
    lab['y'] = lab.cookie_id.map(labels).astype(int)
    lck = lab.drop_duplicates('cookie_id')[['cookie_id', 'day', 'y']]
    last = int(lck.day.max())
    tgt = ci_tgt[ci_tgt.cookie_id.isin(set(target_cookies))]
    out = []
    for D in np.sort(tgt.day.unique()):
        cur = tgt[tgt.day == D]; E = min(D, last + 1); lo = E - hist_days
        h = lab[(lab.day < E) & (lab.day >= lo)]; hc = lck[(lck.day < E) & (lck.day >= lo)]
        if int(hc.y.sum()) == 0 or int((1 - hc.y).sum()) == 0: continue
        agg = h.groupby('item_id').y.agg(['sum', 'size'])
        nb = cur.item_id.map(agg['sum']).fillna(0).values; nh = cur.item_id.map(agg['size']).fillna(0).values - nb
        out.append(pd.DataFrame({'cookie_id': cur.cookie_id.values, 'nb': nb, 'nh': nh, 'rate': (nb + m * p0) / (nb + nh + m),
                                 'llr': np.log((nb + 0.5) / (nh + 0.5))}))
    p = pd.concat(out); g = p.groupby('cookie_id')
    F = pd.DataFrame({'pool_rate_mean': g.rate.mean(), 'pool_rate_max': g.rate.max(), 'pool_llr_mean': g.llr.mean(),
                      'pool_llr_sum': g.llr.sum(), 'pool_llr_max': g.llr.max(),
                      'pool_nb_ge1_frac': (p.nb >= 1).groupby(p.cookie_id).mean(), 'pool_nb_ge2_frac': (p.nb >= 2).groupby(p.cookie_id).mean(),
                      'pool_nb_ge1_cnt': (p.nb >= 1).groupby(p.cookie_id).sum(), 'pool_known_frac': ((p.nb + p.nh) > 0).groupby(p.cookie_id).mean()})
    F.index.name = 'cookie_id'
    return F


def build_prefix_tables():
    tr, te, meta = load_meta()
    ev = load_events(meta)
    ci_full = cookie_items(ev)
    for f in FRACS:
        p = R6 / 'cache' / f'features_f{f}.parquet'
        if p.exists(): continue
        evf = truncate(ev, f)
        F = build_features(evf, meta)
        cif = cookie_items(evf)
        F = F.drop(columns=[c for c in F.columns if c.startswith('pop_')], errors='ignore').join(popularity_prefix(ci_full, cif))
        F.to_parquet(p); cif.to_parquet(R6 / 'cache' / f'ci_f{f}.parquet')
        print('built fraction', f, F.shape, flush=True)
    return ci_full


if __name__ == '__main__':
    t0 = time.time()
    ci_full = build_prefix_tables()
    H = Harness(); D = H.D
    B0C = [c for c in BASE_COLS if c not in POOL_COLS]; PC = [c for c in BASE_COLS if c in POOL_COLS]
    FF = {f: pd.read_parquet(R6 / 'cache' / f'features_f{f}.parquet') for f in FRACS}
    CI = {f: pd.read_parquet(R6 / 'cache' / f'ci_f{f}.parquet') for f in FRACS}
    B0raw = {s: np.load(ROOT / 'research4' / 'oof' / f'experts_{s}.npz')['B0'] for s in H.schemes}   # P100 = B0 (cycle-4 refit, bit-identical)
    for sch, folds in H.schemes.items():
        out = {f'P{f}': np.full(len(D.y), np.nan) for f in FRACS}
        out['P100'] = B0raw[sch]
        for k, (tri, vai) in enumerate(folds):
            lab = D.labels.iloc[tri]
            for f in FRACS:
                pool = pool_prefix(ci_full, CI[f], lab, D.ids, 7)
                X = lambda idx: FF[f].loc[D.ids[idx], B0C].join(pool[PC])[B0C + PC]
                m = make_model('cat', None, 0).fit(X(tri), D.y[tri])
                out[f'P{f}'][vai] = m.predict_proba(X(vai))[:, 1]
        np.savez(R6 / 'oof' / f'prefix_{sch}.npz', **out)
        print(sch, 'done', round(time.time() - t0), 's', flush=True)

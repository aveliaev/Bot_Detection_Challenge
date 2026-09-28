"""C3.5/C3.6/C3.10 event-level pattern log-likelihood ratios (fold-aware family).

Patterns per event (all computed from the cookie's own in-window events):
  tdt  : prev_event > event | delay bucket
  tri  : prev > event > next
  ptr  : platform | event | pointer zone (5x4 grid, edge, none)
  cat  : event | category-switch flag | next_event
  pg   : event | search-page bucket | delay bucket
  sig  : hash(platform, transition, pointer zone, delay bucket)   ('exact software fingerprint')
  m3dt : prev > event > next | delay bucket(prev->event) | delay bucket(event->next)  (motif with timing)
Counts are cookie-level presence (a pattern counts once per cookie) so heavy cookies cannot dominate.
LLR(pattern) = log((nb+a*pb)/(NB+a)) - log((nh+a*ph)/(NH+a)), a = smoothing towards the pooled rate.
Cookie aggregates: sum, mean, top3 mean, bottom3 mean, q25, q75, frac(LLR>1), frac(LLR<-1).
Negatives: 'all' = every labelled human, 'hard' = humans in the top 10% of an inner leave-day-out LGB score
(fit only on the fold's training labels)."""
from __future__ import annotations
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from common import events, meta, CACHE

FAMS = ['tdt', 'tri', 'ptr', 'cat', 'pg', 'sig', 'm3dt']
_P = None


def dtb(x):
    return pd.cut(x, [-1, 2, 5, 10, 20, 45, 90, 300, 1800, 1e9], labels=False).fillna(-1).astype(int).astype(str)


def patterns():
    global _P
    if _P is not None:
        return _P
    p = CACHE / 'patterns.parquet'
    if p.exists():
        _P = pd.read_parquet(p); return _P
    ev = events()
    sh = ev.event_name.str[:5]
    same_next = ev.cookie_id.eq(ev.cookie_id.shift(-1))
    nxt = sh.shift(-1).where(same_next, 'END')
    prv = ev.prev_ev.str[:5]
    d0 = dtb(ev['dt']); d1 = dtb(ev['dt'].shift(-1).where(same_next))
    x, y = ev.pointer_x, ev.pointer_y
    edge = x.isin([0, 1920]) | y.isin([0, 1080])
    zone = np.where(x.isna(), 'none', np.where(edge, 'edge', (x // 384).astype('Int64').astype(str) + '_' + (y // 270).astype('Int64').astype(str)))
    cs = ev.item_category.ne(ev.item_category.shift(1)) & ev.cookie_id.eq(ev.cookie_id.shift(1))
    pgb = pd.cut(ev.search_page, [0, 1, 2, 4, 8, 16, 1e9], labels=False).fillna(-1).astype(int).astype(str)
    P = pd.DataFrame({'cookie_id': ev.cookie_id,
                      'tdt': prv + '>' + sh + '|' + d0,
                      'tri': prv + '>' + sh + '>' + nxt,
                      'ptr': ev.pl + '|' + sh + '|' + zone,
                      'cat': sh + '|' + cs.astype(int).astype(str) + '|' + nxt,
                      'pg': sh + '|' + pgb + '|' + d0,
                      'sig': ev.pl + '|' + prv + '>' + sh + '|' + zone + '|' + d0,
                      'm3dt': prv + '>' + sh + '>' + nxt + '|' + d0 + '|' + d1})
    P.to_parquet(p); _P = P
    return P


def _table(Pc, fam, bots, hums, a=20.0):
    """cookie-presence counts -> smoothed LLR per pattern."""
    u = Pc[['cookie_id', fam]].drop_duplicates()
    nb = u[u.cookie_id.isin(bots)][fam].value_counts()
    nh = u[u.cookie_id.isin(hums)][fam].value_counts()
    T = pd.concat([nb.rename('nb'), nh.rename('nh')], axis=1).fillna(0)
    NB, NH = len(bots), len(hums)
    pp = (T.nb + T.nh + 1) / (NB + NH + 2)   # pooled rate prior
    return np.log((T.nb + a * pp) / (NB + a)) - np.log((T.nh + a * pp) / (NH + a)), pp


def _agg(Pq, fam, llr, prior_llr=0.0):
    v = Pq[fam].map(llr).fillna(prior_llr).values
    df = pd.DataFrame({'c': Pq.cookie_id.values, 'v': v}).sort_values(['c', 'v'], kind='mergesort')
    g = df.groupby('c', sort=True)['v']
    r_asc = g.cumcount().values
    r_desc = g.cumcount(ascending=False).values
    out = pd.DataFrame({'sum': g.sum(), 'mean': g.mean(), 'q25': g.quantile(.25), 'q75': g.quantile(.75),
                        'top3': df.v.where(r_desc < 3).groupby(df.c).mean(),
                        'bot3': df.v.where(r_asc < 3).groupby(df.c).mean(),
                        'fhi': (df.v > 1).groupby(df.c).mean(), 'flo': (df.v < -1).groupby(df.c).mean()})
    return out


def hard_negatives(D, lab, frac=0.10):
    """negatives in the top `frac` of an inner leave-one-day-out LightGBM score (training labels only)."""
    import lightgbm as lgb
    from common import meta as _meta
    M = _meta()
    cols = [c for c in D.F.columns if not c.startswith(('pop_', 'item_id_', 'dow', 'tech_'))]
    X = D.F.loc[lab.index, cols]; y = lab.values.astype(int); day = M.loc[lab.index, 'day'].values
    s = np.zeros(len(y))
    for d in np.unique(day):
        tr, va = day != d, day == d
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=30, subsample=0.8,
                               subsample_freq=1, colsample_bytree=0.5, verbose=-1, n_jobs=4, random_state=0).fit(X[tr], y[tr])
        s[va] = m.predict_proba(X[va])[:, 1]
    neg = pd.Series(s, index=lab.index)[y == 0]
    return set(neg[neg >= neg.quantile(1 - frac)].index)


def llr_features(D, lab, ids, fams=FAMS, neg='all', a=20.0, prefix='llr', _hard_cache={}):
    P = patterns(); M = meta()
    lab = lab.astype(int); ids = pd.Index(ids)
    labday = M.loc[lab.index, 'day']
    if neg == 'trait':  # label-free hardness: >= 2 per-cookie bot traits (no model, no labels, no pool)
        T = pd.read_parquet(CACHE / 'traits.parquet')
        k = T.ptr_comp + T.fast + T.deep + T.young + T.scr_ua + T.edge
        hard = set(T.index[k >= 2])
    if neg == 'hard':
        key = (len(lab), int(lab.sum()), int(labday.sum()))
        if key not in _hard_cache:
            _hard_cache[key] = hard_negatives(D, lab)
        hard = _hard_cache[key]
    out = []
    groups = [(ids[~ids.isin(lab.index)], None)]
    in_lab = ids[ids.isin(lab.index)]
    for d in np.sort(labday.unique()):
        groups.append((in_lab[labday.reindex(in_lab).values == d], d))
    for q, d in groups:
        if not len(q):
            continue
        ref = lab if d is None else lab[labday != d]
        bots = set(ref.index[ref == 1])
        hums = set(ref.index[ref == 0])
        if neg in ('hard', 'trait'):
            hums = hums & hard
        Pc = P[P.cookie_id.isin(bots | hums)]
        Pq = P[P.cookie_id.isin(set(q))]
        parts = []
        for f in fams:
            llr, _ = _table(Pc, f, bots, hums, a)
            A = _agg(Pq, f, llr); A.columns = [f'{prefix}_{f}_{c}' for c in A.columns]
            parts.append(A)
        out.append(pd.concat(parts, axis=1))
    F = pd.concat(out).reindex(ids); F.index.name = 'cookie_id'
    return F

"""C6.1 datasets for the self-supervised tasks. Events = cycle-3 cache: in-window only (window_start <= ts < window_end),
exact duplicates removed, sorted by (cookie, ts, eid).

transitions(ev): one row per (event j -> event j+1) inside a cookie.
    inputs  : ONLY events <= j (current/previous event types, Δt history, position, session state, running event mix,
              running item novelty, hour, seconds to window end, platform)
    targets : next_ev (event type of j+1), next_ldt = log1p(Δt to j+1)
prefixes(ev): cookies with >= 8 events, prefixes of k = ceil(f*n) events, f in {.25,.5,.75}.
    inputs  : ONLY the first k events (the fraction f and n are NOT inputs)
    targets : rem_cnt = log1p(n-k), rem_new_items = log1p(#items in suffix never seen in prefix)
Nothing here uses labels, pointer coordinates, item ids as features, or the B0 score."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
R6 = Path(__file__).resolve().parent; ROOT = R6.parent
sys.path.insert(0, str(ROOT / 'research3'))
from common import events, meta  # noqa: E402

EV = ['item_view', 'search_results_view', 'photo_swipe', 'favorite_add', 'seller_page_view', 'contact_phone_show',
      'login', 'contact_chat_open', 'contact_message_sent']
EVC = {e: i for i, e in enumerate(EV)}
PLC = {'web': 0, 'android': 1, 'ios': 2}


def base_events():
    ev = events()[['cookie_id', 'event_name', 'item_id', 'item_category', 'item_location', 't', 'day', 'pl']].copy()
    ev['e'] = ev.event_name.map(EVC).astype(int)
    return ev


def _hist(ev):
    """per-event history state computed from events <= j (cumulative within cookie)."""
    c = ev.cookie_id.values
    first = np.r_[True, c[1:] != c[:-1]]
    g = ev.groupby('cookie_id', sort=False)
    H = pd.DataFrame(index=ev.index)
    H['pos'] = g.cumcount()
    H['cur'] = ev.e
    H['prev1'] = g.e.shift(1).fillna(-1).astype(int)
    H['prev2'] = g.e.shift(2).fillna(-1).astype(int)
    H['prev3'] = g.e.shift(3).fillna(-1).astype(int)
    dt = ev.t.diff().where(~first)
    H['ldt0'] = np.log1p(dt)                                   # Δt into the current event
    H['ldt1'] = np.log1p(g.t.diff().groupby(ev.cookie_id).shift(1))
    H['ldt2'] = np.log1p(g.t.diff().groupby(ev.cookie_id).shift(2))
    news = first | (dt.fillna(0).values > 1800)
    sid = np.cumsum(news)
    H['sess_idx'] = pd.Series(news.astype(int)).groupby(c).cumsum().values - 1
    ss = pd.Series(ev.t.values).groupby(sid).transform('min').values
    H['pos_in_sess'] = pd.Series(np.ones(len(ev))).groupby(sid).cumsum().values - 1
    H['t_in_sess'] = ev.t.values - ss
    for e, i in EVC.items():
        H[f'share_{i}'] = (ev.e == i).groupby(c).cumsum().values / (H.pos.values + 1)
    it = ev.item_id
    seen = ev.assign(k=ev.cookie_id + '|' + it.astype(str)).k.where(it.notna())
    newitem = seen.notna() & ~seen.duplicated()
    H['new_item_rate'] = newitem.groupby(c).cumsum().values / (H.pos.values + 1)
    H['ldt_mean'] = np.log1p(dt).fillna(0).groupby(c).cumsum().values / np.maximum(H.pos.values, 1)
    H['hour'] = (ev.t.values % 86400) / 3600
    H['to_end'] = 86400 - (ev.t.values % 86400)                  # seconds left in the observation window (day)
    H['pl'] = ev.pl.map(PLC).values
    return H


FEAT_T = ['pos', 'cur', 'prev1', 'prev2', 'prev3', 'ldt0', 'ldt1', 'ldt2', 'sess_idx', 'pos_in_sess', 't_in_sess'] + \
         [f'share_{i}' for i in range(9)] + ['new_item_rate', 'ldt_mean', 'hour', 'to_end', 'pl']
CAT_T = ['cur', 'prev1', 'prev2', 'prev3', 'pl']


def transitions(ev):
    H = _hist(ev)
    c = ev.cookie_id.values
    has_next = np.r_[c[1:] == c[:-1], False]
    T = H[has_next].copy()
    nxt = np.flatnonzero(has_next) + 1
    T['next_ev'] = ev.e.values[nxt]
    T['next_ldt'] = np.log1p(ev.t.values[nxt] - ev.t.values[np.flatnonzero(has_next)])
    T['cookie_id'] = c[has_next]; T['day'] = ev.day.values[has_next]
    return T.reset_index(drop=True)


FEAT_P = ['k', 'dur', 'ldt_med', 'ldt_mean', 'ldt_std', 'n_sess', 'items', 'cats', 'locs', 'hour_end', 'to_end', 'pl'] + \
         [f'sh_{i}' for i in range(9)]


def prefixes(ev, fracs=(.25, .5, .75), min_n=8):
    n = ev.groupby('cookie_id').size()
    ok = n[n >= min_n].index
    E = ev[ev.cookie_id.isin(ok)].copy()
    E['pos'] = E.groupby('cookie_id').cumcount(); E['n'] = E.cookie_id.map(n).values
    E['dt'] = E.groupby('cookie_id').t.diff()
    rows = []
    for f in fracs:
        k = np.ceil(f * E.n).astype(int)
        P, S = E[E.pos < k], E[E.pos >= k]
        g = P.groupby('cookie_id')
        R = pd.DataFrame({'k': g.size(), 'dur': g.t.max() - g.t.min(), 'ldt_med': np.log1p(g.dt.median()),
                          'ldt_mean': np.log1p(P.dt).groupby(P.cookie_id).mean(), 'ldt_std': np.log1p(P.dt).groupby(P.cookie_id).std(),
                          'n_sess': (P.dt > 1800).groupby(P.cookie_id).sum() + 1, 'items': g.item_id.nunique(),
                          'cats': g.item_category.nunique(), 'locs': g.item_location.nunique(),
                          'hour_end': (g.t.max() % 86400) / 3600, 'to_end': 86400 - g.t.max() % 86400,
                          'pl': g.pl.first().map(PLC)})
        for i in range(9):
            R[f'sh_{i}'] = (P.e == i).groupby(P.cookie_id).mean()
        # targets from the suffix
        seen = set(zip(P.cookie_id, P.item_id.dropna().reindex(P.index)))
        Si = S.dropna(subset=['item_id'])
        newi = Si[[(a, b) not in seen for a, b in zip(Si.cookie_id, Si.item_id)]]
        R['rem_cnt'] = np.log1p(S.groupby('cookie_id').size().reindex(R.index).fillna(0))
        R['rem_new_items'] = np.log1p(newi.groupby('cookie_id').item_id.nunique().reindex(R.index).fillna(0))
        R['frac'] = f   # bookkeeping only, never an input
        rows.append(R)
    out = pd.concat(rows).reset_index()
    out['day'] = out.cookie_id.map(ev.drop_duplicates('cookie_id').set_index('cookie_id').day)
    return out


if __name__ == '__main__':
    ev = base_events()
    T = transitions(ev); P = prefixes(ev)
    T.to_parquet(R6 / 'cache' / 'transitions.parquet'); P.to_parquet(R6 / 'cache' / 'prefixes.parquet')
    print('transitions', T.shape, 'cookies', T.cookie_id.nunique(), '| prefixes', P.shape, 'cookies', P.cookie_id.nunique())
    print(T.next_ev.value_counts(normalize=True).round(3).to_dict())

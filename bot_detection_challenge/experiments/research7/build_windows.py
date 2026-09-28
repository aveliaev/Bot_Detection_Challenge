"""C7.4 local windows: per-event causal tokens, random training windows and deterministic inference windows, window features.
Windows are defined per cookie only (fold-independent geometry); which cookies' windows are used for training is decided
per outer fold AFTER the split, so a cookie's windows never cross folds."""
from __future__ import annotations
import sys; from pathlib import Path
R7 = Path(__file__).resolve().parent; ROOT = R7.parent
sys.path.insert(0, str(ROOT / 'research3'))
import numpy as np, pandas as pd
from common import events

EV = ['item_view', 'search_results_view', 'photo_swipe', 'favorite_add', 'seller_page_view', 'contact_phone_show',
      'login', 'contact_chat_open', 'contact_message_sent']
LENS = [3, 5, 10, 20, 40]
PAIR_GAP = 300.0


def tokens(ev: pd.DataFrame | None = None) -> pd.DataFrame:
    """per-event causal token table (uses only events j-1, j-2 of the same cookie)."""
    if ev is None:
        ev = events()
    E = pd.DataFrame({'cookie_id': ev.cookie_id.values, 'e': ev.event_name.map({k: i for i, k in enumerate(EV)}).values.astype(int),
                      't': ev.t.values, 'item': ev.item_id.values, 'cat': ev.item_category.values, 'loc': ev.item_location.values,
                      'page': ev.search_page.values, 'x': ev.pointer_x.values, 'y': ev.pointer_y.values, 'pl': ev.pl.values})
    c = E.cookie_id.values
    s1 = np.r_[False, c[1:] == c[:-1]]
    E['pos'] = E.groupby('cookie_id').cumcount().values
    dt = np.where(s1, np.r_[np.nan, np.diff(E.t.values)], np.nan)
    E['ldt'] = np.log1p(dt)
    E['gap'] = (dt > PAIR_GAP).astype(float)
    sh = lambda a: np.r_[[np.nan], a[:-1]] if a.dtype.kind == 'f' else np.r_[[None], a[:-1]]
    for col, name in [('item', 'item_chg'), ('cat', 'cat_chg'), ('loc', 'loc_chg')]:
        a = E[col].values; p = pd.Series(a).shift(1).values
        E[name] = (s1 & pd.notna(a) & pd.notna(p) & (a != p)).astype(float)
    E['has_item'] = pd.notna(E.item).astype(float)
    hp = E.x.notna().values
    vp = s1 & hp & np.r_[False, hp[:-1]] & (np.nan_to_num(dt, nan=1e9) <= PAIR_GAP)
    dx = np.where(vp, E.x.values - np.r_[np.nan, E.x.values[:-1]], np.nan)
    dy = np.where(vp, E.y.values - np.r_[np.nan, E.y.values[:-1]], np.nan)
    dist = np.sqrt(dx ** 2 + dy ** 2)
    vel = dist / np.maximum(dt, 1.0)
    th = np.arctan2(dy, dx)
    vp2 = vp & np.r_[False, vp[:-1]]
    acc = np.where(vp2, np.abs(vel - np.r_[np.nan, vel[:-1]]) / np.maximum(dt, 1.0), np.nan)
    dth = np.where(vp2, np.angle(np.exp(1j * (th - np.r_[np.nan, th[:-1]]))), np.nan)
    E['hp'] = hp.astype(float); E['vp'] = vp.astype(float); E['vp2'] = vp2.astype(float)
    E['dist'] = dist; E['vel'] = vel; E['acc'] = acc; E['absturn'] = np.abs(dth)
    return E


def windows(E: pd.DataFrame, seed=0, n_train=6) -> pd.DataFrame:
    """rows: cookie_id, start, end (exclusive), L, kind ('train' random / 'infer' deterministic), position index."""
    rng = np.random.RandomState(seed)
    n = E.groupby('cookie_id', sort=False).size()
    out = []
    for cid, N in n.items():
        allowed = [L for L in LENS if L <= N] + ['full']
        for _ in range(n_train):
            L = allowed[rng.randint(len(allowed))]
            L = N if L == 'full' else L
            s = rng.randint(0, N - L + 1)
            out.append((cid, s, s + L, L if L != N else -1 if L not in LENS else L, 'train', -1))
        for L in [L for L in LENS if L <= N]:
            starts = sorted(set(int(round(f * (N - L))) for f in (0, .25, .5, .75, 1.0)))
            for k, s in enumerate(starts):
                out.append((cid, s, s + L, L, 'infer', k))
        out.append((cid, 0, N, -1, 'infer', 0))                 # full window, L = -1
    W = pd.DataFrame(out, columns=['cookie_id', 'start', 'end', 'L', 'kind', 'posk'])
    return W


def window_features(E: pd.DataFrame, W: pd.DataFrame) -> pd.DataFrame:
    """features of every window (only events start..end-1 of its cookie; pairs/turns fully inside the window)."""
    off = E.groupby('cookie_id', sort=False).size().cumsum().shift(1, fill_value=0)
    base = W.cookie_id.map(off).values
    a0, a1 = base + W.start.values, base + W.end.values
    arr = {k: E[k].values.astype(float) for k in ['e', 't', 'ldt', 'gap', 'item_chg', 'cat_chg', 'loc_chg', 'has_item', 'page', 'hp',
                                                    'vp', 'vp2', 'dist', 'vel', 'acc', 'absturn', 'x', 'y']}
    item = E.item.values; pl = E.pl.values
    rows = np.zeros((len(W), 44)); names = None
    for i in range(len(W)):
        s, e = a0[i], a1[i]; L = e - s
        ev = arr['e'][s:e].astype(int)
        cnt = np.bincount(ev, minlength=9)
        ld = arr['ldt'][s + 1:e]                                   # Δt between events inside the window
        vp = arr['vp'][s + 1:e] > 0; vp2 = arr['vp2'][s + 2:e] > 0 if L > 2 else np.zeros(0, bool)
        dist = arr['dist'][s + 1:e][vp]; vel = arr['vel'][s + 1:e][vp]
        acc = arr['acc'][s + 2:e][vp2] if L > 2 else np.zeros(0); turn = arr['absturn'][s + 2:e][vp2] if L > 2 else np.zeros(0)
        it = item[s:e]; itn = it[pd.notna(it)]
        _, first = np.unique(itn, return_index=True) if len(itn) else (np.zeros(0), np.zeros(0))
        hp = arr['hp'][s:e] > 0
        xs, ys = arr['x'][s:e][hp], arr['y'][s:e][hp]
        path = dist.sum() if len(dist) else np.nan
        net = np.hypot(xs[-1] - xs[0], ys[-1] - ys[0]) if len(xs) >= 2 else np.nan
        pg = arr['page'][s:e]; pg = pg[~np.isnan(pg)]
        f = [L, np.log1p(arr['t'][e - 1] - arr['t'][s]), *(cnt / L), (cnt > 0).sum(),
             ld.mean() if len(ld) else np.nan, ld.std() if len(ld) > 1 else np.nan, ld.max() if len(ld) else np.nan,
             arr['gap'][s + 1:e].mean() if L > 1 else np.nan,
             arr['item_chg'][s + 1:e].mean() if L > 1 else np.nan, arr['cat_chg'][s + 1:e].mean() if L > 1 else np.nan,
             arr['loc_chg'][s + 1:e].mean() if L > 1 else np.nan,
             (len(itn) - len(first)) / L, len(first) / L, pg.mean() if len(pg) else np.nan, hp.mean(), vp.sum(),
             dist.mean() if len(dist) else np.nan, dist.std() if len(dist) > 1 else np.nan, dist.max() if len(dist) else np.nan,
             vel.mean() if len(vel) else np.nan, vel.std() if len(vel) > 1 else np.nan,
             acc.mean() if len(acc) else np.nan, turn.mean() if len(turn) else np.nan,
             path, net, path / max(net, 1.0) if len(dist) and not np.isnan(net) else np.nan,
             float(pl[s] == 'web'), float(pl[s] == 'android'), float(pl[s] == 'ios')]
        rows[i, :len(f)] = f
    names = ['w_len', 'w_ldur'] + [f'w_sh_{k}' for k in range(9)] + ['w_ntypes', 'w_ldt_mean', 'w_ldt_std', 'w_ldt_max', 'w_gap_frac',
             'w_item_chg', 'w_cat_chg', 'w_loc_chg', 'w_revisit', 'w_newitem', 'w_page', 'w_ptr_frac', 'w_npairs', 'w_dist_mean',
             'w_dist_std', 'w_dist_max', 'w_vel_mean', 'w_vel_std', 'w_acc_mean', 'w_turn_mean', 'w_path', 'w_net', 'w_straight',
             'w_web', 'w_android', 'w_ios']
    return pd.DataFrame(rows[:, :len(names)], columns=names, index=W.index)


def shuffle_order(E: pd.DataFrame, seed=1) -> pd.DataFrame:
    """negative control: permute event attributes within each cookie while keeping the timestamps in place,
    then recompute all causal tokens (transitions, Δt pairing, kinematics) on the shuffled order."""
    rng = np.random.RandomState(seed)
    ev = events().copy()
    idx = ev.groupby('cookie_id', sort=False).apply(lambda d: pd.Series(rng.permutation(d.index.values.copy()))).values
    for col in ['event_name', 'item_id', 'item_category', 'item_location', 'search_page', 'pointer_x', 'pointer_y']:
        ev[col] = ev.loc[idx, col].values
    return tokens(ev)


if __name__ == '__main__':
    import time
    t0 = time.time()
    E = tokens(); W = windows(E)
    F = window_features(E, W)
    W.join(F).to_parquet(R7 / 'cache' / 'windows.parquet')
    print('windows', W.kind.value_counts().to_dict(), 'features', F.shape, round(time.time() - t0), 's')
    Es = shuffle_order(E)
    Wi = W[W.kind == 'infer']
    Fs = window_features(Es, Wi)
    Wi.join(Fs).to_parquet(R7 / 'cache' / 'windows_shuffled.parquet')
    print('shuffled-order control windows', Fs.shape, round(time.time() - t0), 's')

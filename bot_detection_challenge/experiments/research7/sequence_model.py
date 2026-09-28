"""C7.9-C7.12 small LSTM / BiLSTM over event tokens with random-length chunk training (Maryland) and Kanpur-style
per-event kinematics. Strict protocol: per outer fold, trained only on the outer TRAINING cookies; early stopping on an
inner temporal hold-out (last 2 training days); validation cookies scored on the locked deterministic windows."""
from __future__ import annotations
import sys; from pathlib import Path
R7 = Path(__file__).resolve().parent; ROOT = R7.parent
sys.path.insert(0, str(R7))
import numpy as np, pandas as pd
import torch, torch.nn as nn
from build_windows import tokens, LENS

CONT_GROUPS = {
    'timing': ['ldt', 'miss_dt', 'gapf', 'relpos', 'reltime'],
    'pointer': ['hp', 'xs', 'ys', 'vp', 'dxs', 'dys', 'dists', 'lvel', 'lacc', 'sth', 'cth', 'sturn', 'cturn'],
    'nav': ['item_chg', 'cat_chg', 'loc_chg', 'has_item', 'newitem', 'pages'],
    'platform': ['pweb', 'pand', 'pios'],
}


def token_matrix(E: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    """-> event ids (N,), continuous matrix (N, C), cookie offsets. Scaling constants are fixed a priori (no data fit)."""
    n = E.groupby('cookie_id', sort=False).size()
    dt = np.expm1(E.ldt.values)
    x, y = E.x.values, E.y.values
    c = E.cookie_id.values; s1 = np.r_[False, c[1:] == c[:-1]]
    dx = np.where(E.vp.values > 0, x - np.r_[np.nan, x[:-1]], np.nan); dy = np.where(E.vp.values > 0, y - np.r_[np.nan, y[:-1]], np.nan)
    th = np.arctan2(dy, dx)
    turn = np.where(E.vp2.values > 0, E.absturn.values, np.nan)   # |turn|; sign not needed for sin/cos of magnitude
    it = E.item.values
    key = pd.Series(E.cookie_id.values + '|' + pd.Series(it).astype(str).values)
    newitem = (pd.notna(it) & ~key.duplicated().values).astype(float)
    cols = {
        'ldt': np.nan_to_num(E.ldt.values) / 10, 'miss_dt': np.isnan(E.ldt.values).astype(float), 'gapf': E.gap.values,
        'relpos': E.pos.values / np.repeat(n.values, n.values), 'reltime': (E.t.values % 86400) / 86400,
        'hp': E.hp.values, 'xs': np.nan_to_num(x) / 1920, 'ys': np.nan_to_num(y) / 1080, 'vp': E.vp.values,
        'dxs': np.nan_to_num(dx) / 1000, 'dys': np.nan_to_num(dy) / 1000, 'dists': np.nan_to_num(E.dist.values) / 1000,
        'lvel': np.log1p(np.nan_to_num(E.vel.values)) / 5, 'lacc': np.log1p(np.nan_to_num(E.acc.values)) / 5,
        'sth': np.where(np.isnan(th), 0, np.sin(th)), 'cth': np.where(np.isnan(th), 0, np.cos(th)),
        'sturn': np.where(np.isnan(turn), 0, np.sin(turn)), 'cturn': np.where(np.isnan(turn), 0, np.cos(turn)),
        'item_chg': E.item_chg.values, 'cat_chg': E.cat_chg.values, 'loc_chg': E.loc_chg.values, 'has_item': E.has_item.values,
        'newitem': newitem, 'pages': np.clip(np.nan_to_num(E.page.values), 0, 30) / 10,
        'pweb': (E.pl.values == 'web').astype(float), 'pand': (E.pl.values == 'android').astype(float), 'pios': (E.pl.values == 'ios').astype(float)}
    names = [k for g in CONT_GROUPS.values() for k in g]
    X = np.stack([cols[k] for k in names], 1).astype(np.float32)
    off = pd.Series(np.r_[0, n.values.cumsum()[:-1]], index=n.index)
    return E.e.values.astype(np.int64), X, off, n, names


class SeqNet(nn.Module):
    def __init__(self, n_cont, bidir, emb=12, hidden=48, dropout=0.25, use_event=True):
        super().__init__()
        self.use_event = use_event
        self.emb = nn.Embedding(9, emb)
        self.rnn = nn.LSTM((emb if use_event else 0) + n_cont, hidden, batch_first=True, bidirectional=bidir)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden * (2 if bidir else 1), 1)
        self.bidir = bidir

    def forward(self, ev, xc, lengths):
        z = torch.cat([self.emb(ev), xc], -1) if self.use_event else xc
        packed = nn.utils.rnn.pack_padded_sequence(z, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.rnn(packed)
        hcat = torch.cat([h[-2], h[-1]], -1) if self.bidir else h[-1]
        return self.head(self.drop(hcat)).squeeze(-1)


def batch(EV, XC, spans, cols):
    L = np.array([e - s for s, e in spans]); T = L.max()
    ev = np.zeros((len(spans), T), np.int64); xc = np.zeros((len(spans), T, len(cols)), np.float32)
    for i, (s, e) in enumerate(spans):
        ev[i, :e - s] = EV[s:e]; xc[i, :e - s] = XC[s:e][:, cols]
    return torch.from_numpy(ev), torch.from_numpy(xc), torch.from_numpy(L)


MAXLEN = 200   # 'full' chunks longer than this are represented by a random contiguous 200-event span (train) / the middle span (infer)


def random_spans(cookies, off, n, rng):
    out = []
    for c in cookies:
        N = int(n[c]); allowed = [L for L in LENS if L <= N] + [min(N, MAXLEN)]
        L = allowed[rng.randint(len(allowed))]; s = rng.randint(0, N - L + 1)
        out.append((off[c] + s, off[c] + s + L))
    return out


def train_model(cookies_tr, cookies_es, labels, EV, XC, off, n, cols, bidir, seed, use_event=True, max_epochs=25, patience=3, log=None, chunks_per_epoch=8):
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    net = SeqNet(len(cols), bidir, use_event=use_event)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    ytr = labels[cookies_tr].values.astype(np.float32)
    es_spans = [(off[c] + max(0, int(n[c]) - MAXLEN) // 2, off[c] + max(0, int(n[c]) - MAXLEN) // 2 + min(int(n[c]), MAXLEN)) for c in cookies_es]
    best, best_state, bad = -1, None, 0
    from sklearn.metrics import roc_auc_score
    for ep in range(max_epochs):
        net.train()
        perm = rng.permutation(np.tile(np.arange(len(cookies_tr)), chunks_per_epoch))   # each cookie: 8 random chunks per epoch
        spans = random_spans(np.asarray(cookies_tr)[perm], off, n, rng)
        for i in range(0, len(perm), 256):
            ev, xc, L = batch(EV, XC, spans[i:i + 256], cols)
            loss = lossf(net(ev, xc, L), torch.from_numpy(ytr[perm[i:i + 256]]))
            opt.zero_grad(); loss.backward(); opt.step()
        p = predict(net, EV, XC, es_spans, cols)
        auc = roc_auc_score(labels[cookies_es].values, p)
        if log is not None: log.append((ep, float(auc)))
        if auc > best + 1e-4: best, best_state, bad = auc, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience: break
    net.load_state_dict(best_state)
    return net


@torch.no_grad()
def predict(net, EV, XC, spans, cols, bs=1024):
    net.eval(); out = np.zeros(len(spans), np.float32)
    order = np.argsort([e - s for s, e in spans])          # length-bucketed batches for speed
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]
        ev, xc, L = batch(EV, XC, [spans[j] for j in idx], cols)
        out[idx] = torch.sigmoid(net(ev, xc, L)).numpy()
    return out

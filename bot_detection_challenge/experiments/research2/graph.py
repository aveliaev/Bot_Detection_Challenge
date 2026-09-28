"""Cookie<->item graph features (cycle 2, phase B).

Temporal/OOF contract (same as history.pool_features):
  target cookie of day D, E = min(D, last_labelled_day + 1)
  * labelled neighbours: cookies in `lab` (fold-train labels only) with day in [E-H, E-1]
  * unlabelled graph (2-hop): cookies of ANY part (train/val/test, no labels) with day in [D-6, D],
    i.e. events that exist before the target's window_end_ts; the target cookie itself is removed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse


def _mat(ci_part, ck_index, it_index):
    r = ck_index.get_indexer(ci_part["cookie_id"])
    c = it_index.get_indexer(ci_part["item_id"])
    return sparse.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(len(ck_index), len(it_index)))


def _topk_mean(M, k):
    if M.shape[1] == 0:
        return np.zeros(M.shape[0])
    k = min(k, M.shape[1])
    part = -np.partition(-M, k - 1, axis=1)[:, :k]
    return part.mean(axis=1)


def graph_features(D, lab: pd.Series, target_ids, H: int = 7, m: float = 2.0, p0: float = 0.08,
                   two_hop: bool = True) -> pd.DataFrame:
    ci = D.ci
    items = pd.Index(np.sort(ci["item_id"].unique()))
    labd = ci[ci["cookie_id"].isin(lab.index)]
    last = int(labd["day"].max())
    tgt = ci[ci["cookie_id"].isin(set(target_ids))]
    y_of = lab
    out = []
    for Dd in np.sort(tgt["day"].unique()):
        cur = tgt[tgt["day"] == Dd]
        E = min(Dd, last + 1)
        h = labd[(labd["day"] < E) & (labd["day"] >= E - H)]
        if len(h) == 0:
            continue
        hy = h["cookie_id"].map(y_of).values
        hb, hh = h[hy == 1], h[hy == 0]
        t_idx = pd.Index(cur["cookie_id"].unique())
        b_idx, h_idx = pd.Index(hb["cookie_id"].unique()), pd.Index(hh["cookie_id"].unique())
        T = _mat(cur, t_idx, items)
        B = _mat(hb, b_idx, items)
        Hm = _mat(hh, h_idx, items)
        tsz = np.asarray(T.sum(1)).ravel()
        bsz = np.asarray(B.sum(1)).ravel()
        hsz = np.asarray(Hm.sum(1)).ravel()
        IB = (T @ B.T).toarray()
        IH = (T @ Hm.T).toarray()
        JB = IB / (tsz[:, None] + bsz[None, :] - IB + 1e-9)
        JH = IH / (tsz[:, None] + hsz[None, :] - IH + 1e-9)
        OB = IB / (np.minimum(tsz[:, None], bsz[None, :]) + 1e-9)
        OH = IH / (np.minimum(tsz[:, None], hsz[None, :]) + 1e-9)
        # IDF-weighted (Adamic-Adar-like) similarity: rare shared items count more
        deg = np.asarray(B.sum(0)).ravel() + np.asarray(Hm.sum(0)).ravel()
        w = 1.0 / np.log(2.0 + deg)
        Tw = T.multiply(w[None, :]).tocsr()
        AB = (Tw @ B.T).toarray()
        AH = (Tw @ Hm.T).toarray()
        nbb, nbh = (IB >= 1).sum(1), (IH >= 1).sum(1)
        f = pd.DataFrame(index=t_idx)
        f["g_nbot_nb"] = nbb
        f["g_nhum_nb"] = nbh
        f["g_nbot_nb2"] = (IB >= 2).sum(1)
        f["g_nhum_nb2"] = (IH >= 2).sum(1)
        f["g_bot_nb_share"] = (nbb + m * p0) / (nbb + nbh + m)
        f["g_bot_nb2_share"] = (f["g_nbot_nb2"] + m * p0) / (f["g_nbot_nb2"] + f["g_nhum_nb2"] + m)
        f["g_jb_max"] = JB.max(1) if JB.shape[1] else 0
        f["g_jh_max"] = JH.max(1) if JH.shape[1] else 0
        f["g_jb_top3"] = _topk_mean(JB, 3)
        f["g_jh_top3"] = _topk_mean(JH, 3)
        f["g_jb_top10"] = _topk_mean(JB, 10)
        f["g_jh_top10"] = _topk_mean(JH, 10)
        f["g_j_diff"] = f["g_jb_max"] - f["g_jh_max"]
        f["g_j_top3_diff"] = f["g_jb_top3"] - f["g_jh_top3"]
        f["g_ob_max"] = OB.max(1) if OB.shape[1] else 0
        f["g_oh_max"] = OH.max(1) if OH.shape[1] else 0
        f["g_aab_max"] = AB.max(1) if AB.shape[1] else 0
        f["g_aah_max"] = AH.max(1) if AH.shape[1] else 0
        f["g_aab_sum"] = AB.sum(1)
        f["g_aah_sum"] = AH.sum(1)
        f["g_aa_share"] = (f["g_aab_sum"] + 0.1 * p0) / (f["g_aab_sum"] + f["g_aah_sum"] + 0.1)
        # kNN over labelled cookies by Jaccard: bot fraction among top-k neighbours
        J = np.hstack([JB, JH])
        lbl = np.r_[np.ones(JB.shape[1]), np.zeros(JH.shape[1])]
        for k in (5, 20):
            kk = min(k, J.shape[1])
            idx = np.argpartition(-J, kk - 1, axis=1)[:, :kk]
            top = np.take_along_axis(J, idx, 1)
            lb = lbl[idx] * (top > 0)
            f[f"g_knn{k}_botfrac"] = (lb.sum(1) + m * p0) / ((top > 0).sum(1) + m)
        out.append(f)

        if two_hop:
            # unlabelled 2-hop: s0(item)=smoothed labelled bot rate; s1(cookie)=mean s0 over items for all
            # cookies of days [Dd-6, Dd] (any part, no labels); s2(item)=mean s1 over OTHER cookies viewing it
            nb = np.asarray(B.sum(0)).ravel()
            nt = deg
            s0 = (nb + m * p0) / (nt + m)
            u = ci[(ci["day"] <= Dd) & (ci["day"] > Dd - 7)]
            u_idx = pd.Index(u["cookie_id"].unique())
            U = _mat(u, u_idx, items)
            usz = np.asarray(U.sum(1)).ravel()
            s1 = (U @ s0) / np.maximum(usz, 1)
            item_sum = U.T @ s1
            item_cnt = np.asarray(U.sum(0)).ravel()
            # remove target's own contribution
            tpos = u_idx.get_indexer(t_idx)
            s1_t = s1[tpos]
            Tm = T.tocoo()
            own_sum = s1_t[Tm.row]
            s2_vals = (item_sum[Tm.col] - own_sum) / np.maximum(item_cnt[Tm.col] - 1, 1)
            has_other = (item_cnt[Tm.col] - 1) > 0
            s2_vals = np.where(has_other, s2_vals, np.nan)
            d2 = pd.DataFrame({"r": Tm.row, "s2": s2_vals, "s0": s0[Tm.col]})
            g2 = d2.groupby("r")
            f2 = pd.DataFrame(index=np.arange(len(t_idx)))
            f2["g2_s2_mean"] = g2["s2"].mean()
            f2["g2_s2_max"] = g2["s2"].max()
            f2["g2_s2_q75"] = g2["s2"].quantile(0.75)
            f2["g2_cov"] = g2["s2"].apply(lambda s: s.notna().mean())
            f2.index = t_idx
            out[-1] = out[-1].join(f2)
    F = pd.concat(out)
    F.index.name = "cookie_id"
    return F

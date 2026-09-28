"""Time-aware cross-cookie item features.

EDA finding: bot cookies draw item_ids from a shared, temporally stable pool
(52% of 2nd-week bot items were already seen by 1st-week bots vs 9% for humans).

Strict temporal rules (no information after the cookie's window_end_ts):
  * unsupervised popularity for a cookie of day D uses in-window events of OTHER
    cookies (train+test, unlabelled) from days [D-6, D]  (<= window_end_ts);
  * supervised bot-pool statistics for a cookie of day D use only labels of
    cookies from the provided `label_pool` whose day < D (strictly earlier windows).
    In CV the pool is the training part of the fold; for test it is the whole train.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DAY0 = pd.Timestamp("2026-04-06")


def cookie_items(ev: pd.DataFrame) -> pd.DataFrame:
    ci = (ev.dropna(subset=["item_id"]).groupby(["cookie_id", "item_id"]).size()
          .rename("nv").reset_index())
    day = ev.drop_duplicates("cookie_id").set_index("cookie_id")["window_start_ts"]
    ci["day"] = ((ci["cookie_id"].map(day) - DAY0).dt.days).astype(int)
    return ci


def popularity_features(ci: pd.DataFrame, span: int = 7) -> pd.DataFrame:
    """Unsupervised: how many other cookies viewed the same items in days [D-span+1, D]."""
    days = np.sort(ci["day"].unique())
    out = []
    for D in days:
        cur = ci[ci["day"] == D]
        hist = ci[(ci["day"] <= D) & (ci["day"] > D - span)]
        cnt = hist.groupby("item_id").size()
        # raw counts: normalising by the per-day cookie count put values on a day-specific lattice
        # (adversarial AUC 0.995 = day fingerprint), so keep integer counts
        c = cur["item_id"].map(cnt).values - 1  # exclude the cookie itself
        out.append(pd.DataFrame({"cookie_id": cur["cookie_id"].values, "pop": c.astype(float)}))
    p = pd.concat(out)
    g = p.groupby("cookie_id")["pop"]
    F = pd.DataFrame({"pop_mean": g.mean(), "pop_max": g.max(), "pop_min": g.min(), "pop_median": g.median(),
                      "pop_logmean": np.log1p(p["pop"]).groupby(p["cookie_id"]).mean(),
                      "pop_zero_frac": (p["pop"] == 0).groupby(p["cookie_id"]).mean()})
    return F


def pool_features(ci: pd.DataFrame, labels: pd.Series, target_cookies, hist_days: int | None = 7,
                  m: float = 2.0, p0: float = 0.08) -> pd.DataFrame:
    """Supervised item bot-pool stats for `target_cookies`, using only labels in `labels`
    (index cookie_id -> 0/1) of cookies whose day is strictly earlier than the target day.

    History = labelled days [E-hist_days, E-1], E = min(D, last_labelled_day+1). With a fixed
    hist_days every row (train or test) sees a comparable amount of labelled history; test rows
    all use the last `hist_days` labelled days. A constant prior p0 avoids per-day value lattices.
    """
    lab = ci[ci["cookie_id"].isin(labels.index)].copy()
    lab["y"] = lab["cookie_id"].map(labels).astype(int)
    lck = lab.drop_duplicates("cookie_id")[["cookie_id", "day", "y"]]
    last = int(lck["day"].max())
    tgt = ci[ci["cookie_id"].isin(set(target_cookies))]
    out = []
    for D in np.sort(tgt["day"].unique()):
        cur = tgt[tgt["day"] == D]
        E = min(D, last + 1)
        lo = -10**9 if hist_days is None else E - hist_days
        h = lab[(lab["day"] < E) & (lab["day"] >= lo)]
        hc = lck[(lck["day"] < E) & (lck["day"] >= lo)]
        NB, NH = int(hc["y"].sum()), int((1 - hc["y"]).sum())
        if NB == 0 or NH == 0:
            continue  # no labelled history -> features stay NaN
        agg = h.groupby("item_id")["y"].agg(["sum", "size"])
        nb = cur["item_id"].map(agg["sum"]).fillna(0).values
        nh = cur["item_id"].map(agg["size"]).fillna(0).values - nb
        rate = (nb + m * p0) / (nb + nh + m)
        llr = np.log((nb + 0.5) / (nh + 0.5))  # no NB/NH -> identical value lattice for train/test
        out.append(pd.DataFrame({"cookie_id": cur["cookie_id"].values, "nb": nb, "nh": nh, "rate": rate,
                                 "llr": llr}))
    if not out:
        return pd.DataFrame(index=pd.Index([], name="cookie_id"))
    p = pd.concat(out)
    g = p.groupby("cookie_id")
    F = pd.DataFrame({
        "pool_rate_mean": g["rate"].mean(), "pool_rate_max": g["rate"].max(),
        "pool_llr_mean": g["llr"].mean(), "pool_llr_sum": g["llr"].sum(), "pool_llr_max": g["llr"].max(),
        "pool_nb_ge1_frac": (p["nb"] >= 1).groupby(p["cookie_id"]).mean(),
        "pool_nb_ge2_frac": (p["nb"] >= 2).groupby(p["cookie_id"]).mean(),
        "pool_nb_ge1_cnt": (p["nb"] >= 1).groupby(p["cookie_id"]).sum(),
        "pool_known_frac": ((p["nb"] + p["nh"]) > 0).groupby(p["cookie_id"]).mean(),
    })
    F.index.name = "cookie_id"
    return F

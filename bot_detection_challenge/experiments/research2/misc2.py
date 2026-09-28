"""Unsupervised cross-cookie temporal item features + pointer trajectory (cycle 2).

All cross-cookie quantities use only events that happened strictly BEFORE the current event
(any cookie, any part, no labels) -> available at window_end_ts of the target cookie.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cycle1"))
from features import load_events  # noqa: E402

CACHE = Path(__file__).resolve().parent / "misc2.parquet"


def misc2_features(meta) -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    ev = load_events(meta)
    F = {}
    # ---- item lag: time since item was last touched by ANOTHER cookie (strictly earlier)
    it = ev.dropna(subset=["item_id"]).sort_values(["event_ts", "cookie_id"], kind="mergesort").copy()
    it["t"] = it["event_ts"].astype("int64") // 10**6
    # last touch by another cookie: track per item last (t, cookie) and the one before with different cookie
    g = it.groupby("item_id")
    it["prev_t"] = g["t"].shift()
    it["prev_ck"] = g["cookie_id"].shift()
    # if previous touch is by same cookie, walk back using cumulative "last other" via ffill trick
    other = it["prev_ck"] != it["cookie_id"]
    it["lo_t"] = np.where(other, it["prev_t"], np.nan)
    it["lo_t"] = it.groupby(["item_id", "cookie_id"])["lo_t"].ffill()
    it["lag_h"] = (it["t"] - it["lo_t"]) / 3600
    it["n_prior_ck"] = g.cumcount()  # prior touches (any cookie) - proxy of prior exposure
    c = it.groupby("cookie_id")
    F["il_lag_med"] = c["lag_h"].median()
    F["il_lag_min"] = c["lag_h"].min()
    F["il_lag_lt1h"] = (it["lag_h"] < 1).groupby(it["cookie_id"]).mean()
    F["il_lag_lt24h"] = (it["lag_h"] < 24).groupby(it["cookie_id"]).mean()
    F["il_never_seen"] = it["lag_h"].isna().groupby(it["cookie_id"]).mean()
    F["il_prior_touch_med"] = c["n_prior_ck"].median()
    # ---- pointer trajectory
    p = ev[ev["pointer_x"].notna()].copy()
    s = p["cookie_id"].eq(p["cookie_id"].shift())
    dx = np.where(s, p["pointer_x"].diff(), np.nan)
    dy = np.where(s, p["pointer_y"].diff(), np.nan)
    ang = np.arctan2(dy, dx)
    s2 = s & p["cookie_id"].eq(p["cookie_id"].shift(2))
    dang = np.abs(np.angle(np.exp(1j * (ang - np.r_[np.nan, ang[:-1]]))))
    dang = np.where(s2, dang, np.nan)
    ts = p["event_ts"].astype("int64").values // 10**6
    dtp = np.where(s, np.r_[np.nan, np.diff(ts)], np.nan)
    vel = np.sqrt(dx ** 2 + dy ** 2) / np.maximum(dtp, 1)
    q = pd.DataFrame({"ck": p["cookie_id"].values, "dx": dx, "dy": dy, "dang": dang, "vel": vel})
    gq = q.groupby("ck")
    F["pt_axis_move"] = ((q["dx"] == 0) | (q["dy"] == 0)).where(q["dx"].notna()).groupby(q["ck"]).mean()
    F["pt_dang_mean"] = gq["dang"].mean()
    F["pt_dang_std"] = gq["dang"].std()
    F["pt_vel_med"] = gq["vel"].median()
    F["pt_vel_logstd"] = np.log1p(q["vel"]).groupby(q["ck"]).std()
    # exact coordinate seen before in another cookie (strictly earlier), excluding exact screen edges
    p = p.sort_values("event_ts", kind="mergesort")
    xy = p["pointer_x"].astype(int).astype(str) + "_" + p["pointer_y"].astype(int).astype(str)
    edge = (p["pointer_x"] == 0) | (p["pointer_y"] == 0) | (p["pointer_y"] == 1080) | (p["pointer_x"] == 1920)
    first_ck = p.assign(xy=xy).groupby("xy")["cookie_id"].transform("first")
    seen_before = (xy.duplicated() & (first_ck != p["cookie_id"])) & ~edge
    F["pt_xy_seen_before"] = seen_before.groupby(p["cookie_id"]).mean()
    out = pd.DataFrame(F).astype(float)
    out.index.name = "cookie_id"
    out.to_parquet(CACHE)
    return out

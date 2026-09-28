"""Richer item bot-likelihood aggregates + attribute-combination target encodings (cycle 2, phase C).
Same temporal/OOF contract as history.pool_features (labels of days [E-H, E-1] from fold-train only).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cycle1"))
from features import load_events  # noqa: E402

ATTR_CACHE = Path(__file__).resolve().parent / "cookie_attrs.parquet"


def cookie_attrs(meta):
    """(cookie_id, day, key) rows for attribute combinations of item events (distinct per cookie)."""
    if ATTR_CACHE.exists():
        return pd.read_parquet(ATTR_CACHE)
    ev = load_events(meta)
    ev = ev[ev["item_id"].notna()].copy()
    ev["day"] = (ev["window_start_ts"] - pd.Timestamp("2026-04-06")).dt.days
    c, l, s = ev["item_category"].fillna("NA"), ev["item_location"].fillna("NA"), ev["seller_type"].fillna("NA")
    parts = [("cl", c + "|" + l), ("cs", c + "|" + s), ("ls", l + "|" + s), ("cls", c + "|" + l + "|" + s)]
    out = pd.concat([pd.DataFrame({"cookie_id": ev["cookie_id"].values, "day": ev["day"].values,
                                   "key": k + ":" + v.values}) for k, v in parts]).drop_duplicates()
    out.to_parquet(ATTR_CACHE)
    return out


def _hist_rate(rows, lab, target_ids, key, H, m, p0):
    labd = rows[rows["cookie_id"].isin(lab.index)].copy()
    labd["y"] = labd["cookie_id"].map(lab).astype(int)
    last = int(labd["day"].max())
    tgt = rows[rows["cookie_id"].isin(set(target_ids))]
    out = []
    for D in np.sort(tgt["day"].unique()):
        cur = tgt[tgt["day"] == D]
        E = min(D, last + 1)
        h = labd[(labd["day"] < E) & (labd["day"] >= E - H)]
        if not len(h):
            continue
        agg = h.groupby(key)["y"].agg(["sum", "size"])
        nb = cur[key].map(agg["sum"]).fillna(0).values
        n = cur[key].map(agg["size"]).fillna(0).values
        out.append(pd.DataFrame({"cookie_id": cur["cookie_id"].values, "key": cur[key].values,
                                 "rate": (nb + m * p0) / (n + m), "n": n}))
    return pd.concat(out)


def pool2_features(D, lab, target_ids, H=7, m=2.0, p0=0.08, attrs=None):
    r = _hist_rate(D.ci, lab, target_ids, "item_id", H, m, p0)
    g = r.groupby("cookie_id")["rate"]
    F = pd.DataFrame({
        "p2_q25": g.quantile(0.25), "p2_q50": g.median(), "p2_q75": g.quantile(0.75), "p2_q90": g.quantile(0.9),
        "p2_std": g.std(), "p2_min": g.min(),
        "p2_top3": g.apply(lambda s: s.nlargest(3).mean()),
        "p2_gt2x": (r["rate"] > 2 * p0).groupby(r["cookie_id"]).mean(),
        "p2_gt4x": (r["rate"] > 4 * p0).groupby(r["cookie_id"]).mean(),
        "p2_logit_sum": np.log(r["rate"] / (1 - r["rate"]) / (p0 / (1 - p0))).groupby(r["cookie_id"]).sum(),
    })
    if attrs is not None:
        ra = _hist_rate(attrs, lab, target_ids, "key", H, 20.0, p0)
        ra["kind"] = ra["key"].str.split(":").str[0]
        for kind, d in ra.groupby("kind"):
            gg = d.groupby("cookie_id")["rate"]
            F[f"te_{kind}_mean"] = gg.mean()
            F[f"te_{kind}_max"] = gg.max()
    F.index.name = "cookie_id"
    return F

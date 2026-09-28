"""Event-level naive-Bayes evidence (cycle 2).

Each event contributes log p(token | bot) / p(token | human) for several token types
(gap bin x segment, pointer grid cell, search page, transition, location/category switch, ...).
Per cookie we sum / average the evidence per token type.

Leakage contract: class-conditional token counts come only from `lab` (fold-train labels).
Rows that are themselves in `lab` get their evidence from an inner 5-fold split (never from
their own events). Tokens are built from the cookie's own in-window events only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cycle1"))
from features import load_events, EV_SHORT  # noqa: E402

CACHE = Path(__file__).resolve().parent / "tokens.parquet"


def gap_bin(dt):
    edges = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 7, 9, 11, 13.5, 17, 22, 27, 35, 45, 55, 67, 82, 105,
                      150, 240, 450, 1800.5])
    return np.searchsorted(edges, dt)


def build_tokens(meta) -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    ev = load_events(meta)
    t = ev["event_ts"].astype("int64").values // 10**6
    ck = ev["cookie_id"].values
    same = np.r_[False, ck[1:] == ck[:-1]]
    dt = np.where(same, np.r_[0, np.diff(t)], np.nan)
    pl = ev["platform"].str.lower().replace({"desktop": "web", "iphone": "ios"}).values
    seg = np.where(pl == "web", "w", "m")
    en = ev["event_name"].map(EV_SHORT).values.astype(str)
    prev = np.where(same, np.concatenate([[""], en[:-1]]), "S")
    rows = []

    def add(mask, attr, tok):
        rows.append(pd.DataFrame({"cookie_id": ck[mask], "attr": attr, "tok": np.asarray(tok)[mask].astype(str)}))

    ins = same & (dt <= 1800)
    gb = gap_bin(np.nan_to_num(dt))
    add(ins, "gap", np.char.add(seg, gb.astype(str)))
    add(ins, "gap_ev", np.char.add(np.char.add(seg, gb.astype(str)), en.astype(str)))
    add(same & (dt > 1800), "longgap", np.char.add(seg, np.minimum(dt // 3600, 6).astype(int).astype(str)))
    add(np.ones(len(ev), bool), "trans", np.char.add(np.char.add(prev.astype(str), ">"), en.astype(str)))
    px, py = ev["pointer_x"].values, ev["pointer_y"].values
    hp = ~np.isnan(px)
    cell = np.where(hp, (np.nan_to_num(px) // 160).astype(int) * 100 + (np.nan_to_num(py) // 120).astype(int), -1)
    exact = np.where((px == 0) | (py == 0) | (py == 1080) | (px == 1920), "E", "")
    add(hp, "ptr", np.char.add(cell.astype(str), exact))
    add((pl == "web"), "ptr_present", hp.astype(int).astype(str))
    sp = ev["search_page"].values
    s = ~np.isnan(sp)
    add(s, "page", np.minimum(np.nan_to_num(sp), 25).astype(int).astype(str))
    # page step within same query
    q = ev["search_query"].values
    sq = s & same & np.r_[False, q[1:] == q[:-1]] & ~np.isnan(np.r_[np.nan, sp[:-1]])
    step = np.clip(np.nan_to_num(sp - np.r_[np.nan, sp[:-1]]), -3, 3).astype(int)
    add(sq, "pstep", step.astype(str))
    loc = ev["item_location"].fillna("NA").values
    add(same, "locsw", (loc == np.concatenate([["~"], loc[:-1]])).astype(int).astype(str))
    cat = ev["item_category"].fillna("NA").values
    add(same, "catsw", (cat == np.concatenate([["~"], cat[:-1]])).astype(int).astype(str))
    hour = ((ev["event_ts"] - ev["window_start_ts"]).dt.total_seconds().values // 3600).astype(int)
    add(np.ones(len(ev), bool), "hour", np.char.add(seg, hour.astype(str)))
    T = pd.concat(rows, ignore_index=True)
    T["attr"] = T["attr"].astype("category")
    T.to_parquet(CACHE)
    return T


def _llr_table(T_lab: pd.DataFrame, y_of: pd.Series, a: float = 1.0) -> pd.Series:
    yy = T_lab["cookie_id"].map(y_of).values
    c = T_lab.assign(y=yy).groupby(["attr", "tok", "y"], observed=True).size().unstack("y", fill_value=0)
    c = c.reindex(columns=[0, 1], fill_value=0)
    tot = c.groupby(level="attr", observed=True).transform("sum")
    k = c.groupby(level="attr", observed=True)[0].transform("size")
    p1 = (c[1] + a) / (tot[1] + a * k)
    p0 = (c[0] + a) / (tot[0] + a * k)
    return np.log(p1 / p0).rename("llr")


def _aggregate(T_t: pd.DataFrame, llr: pd.Series) -> pd.DataFrame:
    v = T_t.join(llr, on=["attr", "tok"])
    v["llr"] = v["llr"].fillna(0.0)
    g = v.groupby(["cookie_id", "attr"], observed=True)["llr"]
    s = g.sum().unstack("attr")
    mn = g.mean().unstack("attr")
    s.columns = [f"nb_sum_{c}" for c in s.columns]
    mn.columns = [f"nb_mean_{c}" for c in mn.columns]
    out = s.join(mn)
    sum_cols = [c for c in out.columns if c.startswith("nb_sum_")]
    out["nb_sum_total"] = out[sum_cols].fillna(0).sum(1)
    beh = [f"nb_sum_{a}" for a in ("gap", "trans", "ptr", "ptr_present", "page", "pstep", "locsw", "catsw")
           if f"nb_sum_{a}" in out]
    out["nb_sum_behav"] = out[beh].fillna(0).sum(1)
    return out


KEEP_ATTRS = ("gap", "gap_ev", "trans", "ptr", "page", "hour")


def nb_features_clean(T, lab, target_ids, n_inner: int = 5, seed: int = 0) -> pd.DataFrame:
    """NB evidence without low-cardinality token types and rounded to 0.1: removes the
    inner-OOF-table 'value lattice' that let an adversarial model tell train rows from test rows."""
    T = T[T["attr"].isin(KEEP_ATTRS)]
    T = T.assign(attr=T["attr"].astype(str).astype("category"))
    F = nb_features(T, lab, target_ids, n_inner, seed)
    F = F.drop(columns=[c for c in ("nb_sum_behav",) if c in F])
    return F.round(1)


def nb_features(T: pd.DataFrame, lab: pd.Series, target_ids, n_inner: int = 5, seed: int = 0) -> pd.DataFrame:
    lab_ids = lab.index.values
    target_ids = np.asarray(target_ids)
    tgt_in_lab = np.isin(target_ids, lab_ids)
    outs = []
    # rows not in the labelled set: use all labels
    T_lab = T[T["cookie_id"].isin(lab_ids)]
    others = target_ids[~tgt_in_lab]
    if len(others):
        outs.append(_aggregate(T[T["cookie_id"].isin(others)], _llr_table(T_lab, lab)))
    # labelled rows: inner K-fold, never their own events
    rng = np.random.RandomState(seed)
    ins = target_ids[tgt_in_lab]
    fold = rng.randint(0, n_inner, len(ins))
    for f in range(n_inner):
        hold = ins[fold == f]
        keep = lab.drop(hold)
        tbl = _llr_table(T[T["cookie_id"].isin(keep.index)], keep)
        outs.append(_aggregate(T[T["cookie_id"].isin(hold)], tbl))
    F = pd.concat(outs)
    F.index.name = "cookie_id"
    return F

"""Cycle-3 shared helpers: in-window event table, labels, B0 OOF, forensic groups."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

R3 = Path(__file__).resolve().parent
ROOT = R3.parent
sys.path.insert(0, str(ROOT / "cycle1"))
sys.path.insert(0, str(ROOT / "research2"))
from features import load_meta, load_events  # noqa: E402
from metrics import precision_at_recall  # noqa: E402

CACHE = R3 / "cache"; CACHE.mkdir(exist_ok=True)
OOF3 = R3 / "oof"; OOF3.mkdir(exist_ok=True)
DAY0 = pd.Timestamp("2026-04-06")


def events():
    p = CACHE / "ev.parquet"
    if not p.exists():
        tr, te, meta = load_meta()
        ev = load_events(meta)
        ev["t"] = (ev["event_ts"] - DAY0).dt.total_seconds()
        ev["day"] = ((ev["window_start_ts"] - DAY0).dt.days).astype(int)
        ev["pl"] = ev["platform"].str.lower().replace({"desktop": "web", "iphone": "ios"})
        first = ~ev["cookie_id"].duplicated()
        ev["dt"] = ev["t"].diff().where(~first)
        ev["prev_ev"] = ev["event_name"].shift(1).where(~first, "START")
        ev["age_h"] = (ev["event_ts"] - ev["cookie_created_at"]).dt.total_seconds() / 3600
        ev = ev.drop(columns=["window_start_ts", "window_end_ts"])
        ev.to_parquet(p)
    return pd.read_parquet(p)


def meta():
    tr, te, m = load_meta()
    m["day"] = ((m["window_start_ts"] - DAY0).dt.days).astype(int)
    return m.set_index("cookie_id")


def groups(n_flag=604):
    """Forensic groups on the B0 repeated day-block OOF (days>=2 only).
    A persistent FP: target 0, rank < 400 in BOTH repeated OOF and fwd7-OOF-if-available, i.e. stable high score
    FP: target 0 flagged at operating point; B: TP ranked in the same region as A; C: easy TP (top 250);
    D: ordinary negatives (not flagged); E: FN (not flagged)."""
    o = pd.read_csv(ROOT / "research2" / "oof_final_B0.csv").set_index("cookie_id")
    o = o[o.oof_repeated_dayblock > 0].copy()
    o["rank"] = o.oof_repeated_dayblock.rank(ascending=False, method="first").astype(int)
    flag = o["rank"] <= n_flag
    y = o.target
    g = np.full(len(o), "D", dtype=object)
    g[(y == 1) & flag] = "TP"
    g[(y == 0) & flag] = "FP"
    g[(y == 1) & ~flag] = "E"
    o["grp"] = g
    return o

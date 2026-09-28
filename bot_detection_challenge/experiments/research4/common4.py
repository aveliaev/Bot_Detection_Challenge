"""Cycle-4 helpers: load expert OOF, metrics, operating-point accounting, persistent FP."""
import sys, json
from pathlib import Path
R4 = Path(__file__).resolve().parent; ROOT = R4.parent
sys.path.insert(0, str(ROOT / 'cycle1')); sys.path.insert(0, str(ROOT / 'research2'))
import numpy as np, pandas as pd
from metrics import precision_at_recall
from rcv import paired_bootstrap

TR = pd.read_csv(ROOT / 'data' / 'train.csv')
IDS = TR.cookie_id.values; Y = TR.target.values
DAY = ((pd.to_datetime(TR.window_start_ts) - pd.Timestamp('2026-04-06')).dt.days).values
PFP = set(json.load(open(ROOT / 'research3' / 'cache' / 'persistent.json')))
PFP_MASK = np.isin(IDS, list(PFP))
SCHEMES = ['P0', 'P1', 'P2', 'P3', 'P4', 'fwd7']
EXP = ['B0', 'mouse', 'behavior', 'item', 'context']


def load(sch):
    z = np.load(R4 / 'oof' / f'experts_{sch}.npz')
    return {e: z[e] for e in z.files}


def valid_mask(E):
    return ~np.isnan(E['B0'])


def pr(y, s):
    return precision_at_recall(y, s)


def op_point(y, s, ids):
    """operating point of the official metric: TP, FP, FN and the flagged set."""
    order = np.argsort(-s, kind='mergesort'); ys = y[order]
    # tie-aware groups
    ss = s[order]; ends = np.r_[ss[1:] != ss[:-1], True]
    tp = np.cumsum(ys)[ends]; k = np.arange(1, len(ys) + 1)[ends]
    rec, prec = tp / y.sum(), tp / k
    ok = rec >= 0.7; i = np.flatnonzero(ok)[np.argmax(prec[ok])]
    n = k[i]; flagged = set(ids[order[:n]])
    return dict(P=float(prec[i]), TP=int(tp[i]), FP=int(n - tp[i]), FN=int(y.sum() - tp[i]), flagged=flagged)


def rankpct(x, mask):
    out = np.full(len(x), np.nan)
    out[mask] = pd.Series(x[mask]).rank(pct=True).values
    return out

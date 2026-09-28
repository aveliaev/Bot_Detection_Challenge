"""Cycle-2 evaluation harness: repeated day-block CV + fwd7 + paired bootstrap.

Label-dependent feature families are functions  fam(D, label_series, target_ids) -> DataFrame,
recomputed inside every fold from the fold's training labels only (and each family is itself
responsible for using only labelled days strictly earlier than the target cookie's day).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "cycle1"))
from cv import Data, POOL_COLS, make_model, WORK  # noqa: E402
from history import pool_features  # noqa: E402
from metrics import precision_at_recall, report  # noqa: E402

OOF2 = HERE / "oof"
OOF2.mkdir(exist_ok=True)
LOG = HERE / "experiments2.jsonl"
BASE_COLS = json.load(open(HERE.parent / "cycle1" / "feature_list.json"))


def partitions(day, n_rep=5):
    """P0 = contiguous 2-day blocks (cycle-1 scheme); P1.. = random pairings of days 2..13.
    days 0-1 always train (label history needed)."""
    out = [[[2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13]]]
    rng = np.random.RandomState(2024)
    for _ in range(n_rep - 1):
        d = rng.permutation(np.arange(2, 14))
        out.append([sorted(d[i:i + 2].tolist()) for i in range(0, 12, 2)])
    return [[(np.flatnonzero(~np.isin(day, b)), np.flatnonzero(np.isin(day, b))) for b in P] for P in out]


def fwd7(day):
    return [(np.flatnonzero(day < k), np.flatnonzero((day >= k) & (day < k + 7))) for k in (5, 7)]


class Harness:
    def __init__(self, n_rep=5):
        self.D = Data()
        self.D.hist_days = 7
        self.schemes = {f"P{i}": p for i, p in enumerate(partitions(self.D.day, n_rep))}
        self.schemes["fwd7"] = fwd7(self.D.day)
        self.cache = {}
        self.families = {"pool": lambda D, lab, ids: pool_features(D.ci, lab, ids, 7)}

    def add_family(self, name, fn):
        self.families[name] = fn

    def fam_frame(self, fam, scheme, k):
        key = (fam, scheme, k)
        if key not in self.cache:
            tri, _ = self.schemes[scheme][k]
            lab = self.D.labels.iloc[tri]
            self.cache[key] = self.families[fam](self.D, lab, self.D.ids)
        return self.cache[key]

    def X(self, cols, fams, scheme, k, idx, extra=None):
        D = self.D
        ids = D.ids[idx]
        base = [c for c in cols if c in D.F.columns]
        X = D.F.loc[ids, base]
        for f in fams:
            X = X.join(self.fam_frame(f, scheme, k))
        if extra is not None:
            X = X.join(extra)
        want = [c for c in cols if c in X.columns] + [c for c in X.columns if c not in cols and c not in D.F.columns]
        return X[want]

    def run(self, name, cols=None, fams=("pool",), model="cat", params=None, seeds=(0,), schemes=None,
            fit_fn=None, hypothesis="", leak_risk="", note="", verbose=True):
        """fit_fn(Xtr, ytr, Xva, seed, tri, vai, scheme, k) -> val scores (optional custom model)."""
        cols = list(BASE_COLS if cols is None else cols)
        cols = [c for c in cols if c not in POOL_COLS] + [c for c in cols if c in POOL_COLS]
        D = self.D
        schemes = schemes or list(self.schemes)
        t0 = time.time()
        res = {}
        for sch in schemes:
            oof = np.zeros(len(D.y))
            fold_sc = []
            for seed in seeds:
                o = np.full(len(D.y), np.nan)
                for k, (tri, vai) in enumerate(self.schemes[sch]):
                    Xtr, Xva = self.X(cols, fams, sch, k, tri), self.X(cols, fams, sch, k, vai)
                    if fit_fn is not None:
                        o[vai] = fit_fn(Xtr, D.y[tri], Xva, seed, tri, vai, sch, k)
                    else:
                        m = make_model(model, params, seed)
                        m.fit(Xtr, D.y[tri])
                        o[vai] = m.predict_proba(Xva)[:, 1]
                    fold_sc.append(precision_at_recall(D.y[vai], o[vai]))
                mk = ~np.isnan(o)
                oof[mk] += pd.Series(o[mk]).rank(pct=True).values / len(seeds)
            np.save(OOF2 / f"{name}__{sch}.npy", oof)
            mk = oof > 0
            r = report(D.y[mk], oof[mk])
            r.update(fold_min=float(np.min(fold_sc)), fold_std=float(np.std(fold_sc)))
            res[sch] = r
        rep = [res[s]["p_at_r70"] for s in res if s.startswith("P")]
        summ = dict(exp=name, hypothesis=hypothesis, n_feat=len(cols), fams=list(fams), model=model,
                    rep_mean=float(np.mean(rep)) if rep else None, rep_median=float(np.median(rep)) if rep else None,
                    rep_std=float(np.std(rep)) if rep else None, rep_min=float(np.min(rep)) if rep else None,
                    rep_max=float(np.max(rep)) if rep else None,
                    fwd7=res.get("fwd7", {}).get("p_at_r70"),
                    prauc_P=float(np.mean([res[s]["pr_auc"] for s in res if s.startswith("P")])) if rep else None,
                    roc_P=float(np.mean([res[s]["roc_auc"] for s in res if s.startswith("P")])) if rep else None,
                    per_scheme={s: round(res[s]["p_at_r70"], 4) for s in res},
                    sec=round(time.time() - t0, 1), leak_risk=leak_risk, note=note)
        with open(LOG, "a") as f:
            f.write(json.dumps(summ, ensure_ascii=False) + "\n")
        if verbose:
            print(f"{name:34s} rep={summ['rep_mean']:.4f}±{summ['rep_std']:.4f} "
                  f"[{summ['rep_min']:.3f},{summ['rep_max']:.3f}] fwd7={summ['fwd7']:.4f} "
                  f"PR-AUC={summ['prauc_P']:.4f} ROC={summ['roc_P']:.4f} {summ['per_scheme']} {summ['sec']}s",
                  flush=True)
        return summ


def repeated_oof(name, n_rep=5):
    """average rank over the repeated day-block partitions -> one OOF vector (days>=2)."""
    arr = [np.load(OOF2 / f"{name}__P{i}.npy") for i in range(n_rep)]
    return np.mean(arr, axis=0)


def paired_bootstrap(y, a, b, n=2000, seed=0):
    """P(P@R(a) > P@R(b)) and CI of the difference, resampling cookies (stratified)."""
    mk = (a > 0) & (b > 0)
    y, a, b = y[mk], a[mk], b[mk]
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rng = np.random.RandomState(seed)
    d = np.empty(n)
    for i in range(n):
        idx = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        d[i] = precision_at_recall(y[idx], a[idx]) - precision_at_recall(y[idx], b[idx])
    return dict(p_better=float((d > 0).mean()), diff_mean=float(d.mean()),
                ci95=(float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))),
                point=float(precision_at_recall(y, a) - precision_at_recall(y, b)))


def bootstrap_ci(y, a, n=2000, seed=0):
    mk = a > 0
    y, a = y[mk], a[mk]
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rng = np.random.RandomState(seed)
    v = [precision_at_recall(y[idx], a[idx]) for idx in
         (np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))] for _ in range(n))]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

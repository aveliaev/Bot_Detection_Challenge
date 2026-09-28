"""CV harness: fold schemes, fold-aware history features, models, OOF bookkeeping."""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from features import build_features, load_events, load_meta, family_map
from history import cookie_items, pool_features, popularity_features, DAY0
from metrics import precision_at_recall, report

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "work"
WORK.mkdir(exist_ok=True)
OOF_DIR = WORK / "oof"
OOF_DIR.mkdir(exist_ok=True)


# ----------------------------------------------------------------------------- data
class Data:
    def __init__(self, rebuild=False):
        self.tr, self.te, self.meta = load_meta()
        fp, cp = WORK / "features.parquet", WORK / "cookie_items.parquet"
        if rebuild or not fp.exists() or not cp.exists():
            ev = load_events(self.meta)
            build_features(ev, self.meta).to_parquet(fp)
            cookie_items(ev).to_parquet(cp)
        self.F = pd.read_parquet(fp)
        self.ci = pd.read_parquet(cp)
        pop = popularity_features(self.ci)
        self.F = self.F.join(pop)
        self.y = self.tr["target"].values
        self.ids = self.tr["cookie_id"].values
        self.day = ((self.tr["window_start_ts"] - DAY0).dt.days).values
        self.labels = self.tr.set_index("cookie_id")["target"]
        self.base_cols = list(self.F.columns)
        self._pool_cache = {}
        self.hist_days = 7

    def folds(self, scheme: str, seed: int = 0):
        """list of (train_idx, val_idx) over train rows."""
        d = self.day
        if scheme == "dayblock":  # days 0-1 always train; 6 validation blocks of 2 days
            return [(np.flatnonzero(~np.isin(d, b)), np.flatnonzero(np.isin(d, b)))
                    for b in ([2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13])]
        if scheme == "forward":   # pure forward holdout(s) with expanding window
            return [(np.flatnonzero(d < k), np.flatnonzero((d >= k) & (d < k + 2))) for k in (6, 8, 10, 12)]
        if scheme == "fwd7":      # test-like: 7 validation days after the last labelled day (gap 1..7)
            return [(np.flatnonzero(d < k), np.flatnonzero((d >= k) & (d < k + 7))) for k in (5, 7)]
        if scheme == "strat":
            skf = StratifiedKFold(5, shuffle=True, random_state=seed)
            return list(skf.split(d, self.y))
        raise ValueError(scheme)

    def pool(self, scheme, k, tr_idx, val_idx, seed=0):
        key = (scheme, k, seed, self.hist_days)
        if key not in self._pool_cache:
            lab = self.labels.iloc[tr_idx]
            P = pool_features(self.ci, lab, self.ids, self.hist_days)  # train rows use only earlier train-part labels
            self._pool_cache[key] = P
        return self._pool_cache[key]

    def pool_test(self):
        return pool_features(self.ci, self.labels, self.te["cookie_id"].values, self.hist_days)

    def pool_full_train(self):
        """pool features for all train rows using all (earlier) train labels -> final model training."""
        return pool_features(self.ci, self.labels, self.ids, self.hist_days)


POOL_COLS = ["pool_rate_mean", "pool_rate_max", "pool_llr_mean", "pool_llr_sum", "pool_llr_max",
             "pool_nb_ge1_frac", "pool_nb_ge2_frac", "pool_nb_ge1_cnt", "pool_known_frac"]


# ----------------------------------------------------------------------------- models
def make_model(name, params=None, seed=0, pos_weight=1.0):
    params = dict(params or {})
    if name == "lgb":
        import lightgbm as lgb
        p = dict(n_estimators=600, learning_rate=0.02, num_leaves=15, min_child_samples=40, subsample=0.8,
                 subsample_freq=1, colsample_bytree=0.5, reg_lambda=5.0, verbose=-1, random_state=seed,
                 scale_pos_weight=pos_weight, n_jobs=4)
        p.update(params)
        return lgb.LGBMClassifier(**p)
    if name == "cat":
        from catboost import CatBoostClassifier
        p = dict(iterations=1500, learning_rate=0.03, depth=6, l2_leaf_reg=5, rsm=0.5, random_seed=seed,
                 verbose=0, thread_count=4, scale_pos_weight=pos_weight)
        p.update(params)
        return CatBoostClassifier(**p)
    if name == "xgb":
        import xgboost as xgb
        p = dict(n_estimators=600, learning_rate=0.02, max_depth=4, min_child_weight=5, subsample=0.8,
                 colsample_bytree=0.5, reg_lambda=5.0, random_state=seed, n_jobs=4, tree_method="hist",
                 scale_pos_weight=pos_weight, eval_metric="logloss")
        p.update(params)
        return xgb.XGBClassifier(**p)
    if name == "et":
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.impute import SimpleImputer
        p = dict(n_estimators=800, min_samples_leaf=5, max_features=0.3, n_jobs=4, random_state=seed,
                 class_weight={0: 1, 1: pos_weight})
        p.update(params)
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), ExtraTreesClassifier(**p))
    if name == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        p = dict(max_iter=400, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=5.0,
                 random_state=seed, class_weight={0: 1, 1: pos_weight})
        p.update(params)
        return HistGradientBoostingClassifier(**p)
    if name == "lr":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import QuantileTransformer
        p = dict(C=0.05, max_iter=3000, class_weight={0: 1, 1: pos_weight})
        p.update(params)
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                             QuantileTransformer(n_quantiles=200, output_distribution="normal"),
                             LogisticRegression(**p))
    raise ValueError(name)


def get_X(D: Data, cols, idx=None, pool=None):
    X = D.F.loc[D.ids if idx is None else D.ids[idx]]
    use_pool = [c for c in cols if c in POOL_COLS]
    X = X[[c for c in cols if c not in POOL_COLS]]
    if use_pool:
        X = X.join(pool[use_pool] if pool is not None else None)
        X = X[cols]
    return X


def run_cv(D: Data, cols, model="lgb", params=None, scheme="dayblock", seeds=(0,), pos_weight=1.0,
           name=None, save=True, verbose=True, sample_weight_fn=None):
    t0 = time.time()
    oof = np.zeros(len(D.y))
    fold_scores = []
    for seed in seeds:
        folds = D.folds(scheme, seed)
        o = np.full(len(D.y), np.nan)
        for k, (tri, vai) in enumerate(folds):
            P = D.pool(scheme, k, tri, vai, seed) if any(c in POOL_COLS for c in cols) else None
            Xtr, Xva = get_X(D, cols, tri, P), get_X(D, cols, vai, P)
            mdl = make_model(model, params, seed, pos_weight)
            sw = sample_weight_fn(D, tri) if sample_weight_fn else None
            if sw is not None:
                if model in ("et", "lr"):
                    mdl.fit(Xtr, D.y[tri], **{mdl.steps[-1][0] + "__sample_weight": sw})
                else:
                    mdl.fit(Xtr, D.y[tri], sample_weight=sw)
            else:
                mdl.fit(Xtr, D.y[tri])
            o[vai] = mdl.predict_proba(Xva)[:, 1]
            fold_scores.append(precision_at_recall(D.y[vai], o[vai]))
        mask = ~np.isnan(o)
        # average ranks across seeds to be robust to calibration differences
        oof[mask] += pd.Series(o[mask]).rank(pct=True).values / len(seeds)
    mask = oof > 0
    r = report(D.y[mask], oof[mask])
    r.update(fold_mean=float(np.mean(fold_scores)), fold_std=float(np.std(fold_scores)),
             fold_min=float(np.min(fold_scores)), n_feat=len(cols), sec=round(time.time() - t0, 1))
    if save and name:
        np.save(OOF_DIR / f"{name}__{scheme}.npy", oof)
    if verbose:
        print(f"{name or model:40s} {scheme:9s} P@R70={r['p_at_r70']:.4f} fold={r['fold_mean']:.4f}±{r['fold_std']:.4f} "
              f"min={r['fold_min']:.3f} PR-AUC={r['pr_auc']:.4f} ROC={r['roc_auc']:.4f} nf={len(cols)} {r['sec']}s",
              flush=True)
    return r, oof


def log_experiment(row: dict, path=WORK / "experiments.jsonl"):
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")

"""Phase H/I: hard-negative stage-2, difficulty weighting, pairwise ranking — all nested OOF.

For every outer fold, an inner 4-fold CV on the outer-train part gives OOF stage-1 scores for
training rows (a row's score never comes from a model that saw it). These inner scores define
hard negatives / borderline positives for training stage-2 and difficulty weights.
"""
import sys

sys.path.insert(0, "research2")
from rcv import *  # noqa
from sklearn.model_selection import StratifiedKFold

H = Harness()
D = H.D
FAST = dict(iterations=700, learning_rate=0.06)
inner_cache = {}


def inner_oof(Xtr, ytr, seed, key):
    if key in inner_cache:
        return inner_cache[key]
    o = np.zeros(len(ytr))
    for a, b in StratifiedKFold(4, shuffle=True, random_state=seed).split(Xtr, ytr):
        m = make_model("cat", FAST, seed)
        m.fit(Xtr.iloc[a], ytr[a])
        o[b] = m.predict_proba(Xtr.iloc[b])[:, 1]
    inner_cache[key] = o
    return o


def region_mask(scores, y_or_none, frac):
    thr = np.quantile(scores, 1 - frac)
    return scores >= thr


def stage2_fit(lam, frac=0.15):
    def fit(Xtr, ytr, Xva, seed, tri, vai, sch, k):
        A = make_model("cat", None, seed)
        A.fit(Xtr, ytr)
        sA = A.predict_proba(Xva)[:, 1]
        io = inner_oof(Xtr, ytr, seed, (sch, k, seed))
        reg_tr = region_mask(io, ytr, frac)          # top-15% of training by OOF stage-1 score
        B = make_model("cat", None, seed)
        B.fit(Xtr[reg_tr], ytr[reg_tr])
        reg_va = region_mask(sA, None, frac)
        out = sA.copy()  # outside the region: stage-1 score (< 1)
        rA = pd.Series(sA[reg_va]).rank(pct=True).values
        rB = pd.Series(B.predict_proba(Xva[reg_va])[:, 1]).rank(pct=True).values
        # keep region above everything else; reorder inside the region
        out[reg_va] = 1.0 + (rA + lam * rB) / (1 + lam)
        return out
    return fit


def weight_fit(w_neg=3.0, w_pos=2.0):
    def fit(Xtr, ytr, Xva, seed, tri, vai, sch, k):
        io = inner_oof(Xtr, ytr, seed, (sch, k, seed))
        thr = np.quantile(io, 1 - 0.15)
        w = np.ones(len(ytr))
        w[(ytr == 0) & (io >= thr)] = w_neg
        w[(ytr == 1) & (io < np.quantile(io[ytr == 1], 0.5))] = w_pos  # borderline / missed positives
        m = make_model("cat", None, seed)
        m.fit(Xtr, ytr, sample_weight=w)
        return m.predict_proba(Xva)[:, 1]
    return fit


def rank_fit(Xtr, ytr, Xva, seed, tri, vai, sch, k):
    import xgboost as xgb
    m = xgb.XGBRanker(objective="rank:pairwise", n_estimators=600, learning_rate=0.03, max_depth=4,
                      subsample=0.8, colsample_bytree=0.5, reg_lambda=5, random_state=seed, n_jobs=4,
                      lambdarank_pair_method="mean", lambdarank_num_pair_per_sample=8)
    m.fit(Xtr, ytr, qid=np.zeros(len(ytr), int))
    return m.predict(Xva)


if __name__ == "__main__":
    which = sys.argv[1]
    if which == "stage2":
        H.run("H1_stage2_lam1", fit_fn=stage2_fit(1.0), hypothesis="stage-2 TP vs hard-neg in top-15% region",
              leak_risk="medium: nested inner OOF")
        H.run("H2_stage2_lam05", fit_fn=stage2_fit(0.5), hypothesis="stage-2, lambda 0.5",
              leak_risk="medium: nested inner OOF")
    if which == "weights":
        H.run("H3_difficulty_w", fit_fn=weight_fit(), hypothesis="difficulty weights from inner OOF",
              leak_risk="medium: nested inner OOF")
    if which == "rank":
        H.run("I2_xgb_pairwise_mean", fit_fn=rank_fit, hypothesis="pairwise ranking objective",
              leak_risk="low")

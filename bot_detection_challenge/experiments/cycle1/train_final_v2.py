"""Cycle-2 final pipeline (model B0): CatBoost on the top-120 behavioural features + all
time-aware item bot-pool features, averaged over 5 seeds.

Why B0 and not a bigger blend: in cycle 2 every candidate (graph, NB evidence, platform models,
stage-2, ranking, blends) stayed within noise of B0 on repeated day-block CV (0.9003 +- 0.0035)
and none improved the test-like fwd7 check (0.9023). See research2/REPORT_CYCLE2.md.

Run from bot_detection_challenge/:  python solution/train_final_v2.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cv import Data, POOL_COLS, make_model, ROOT  # noqa: E402

SEEDS = (0, 1, 2, 3, 4)
HIST_DAYS = 7


def main():
    D = Data(rebuild=True)
    D.hist_days = HIST_DAYS
    base = [c for c in json.load(open(Path(__file__).resolve().parent / "feature_list.json")) if c not in POOL_COLS]
    P_tr, P_te = D.pool_full_train(), D.pool_test()
    Xtr = D.F.loc[D.ids, base].join(P_tr)
    Xte = D.F.loc[D.te["cookie_id"].values, base].join(P_te)[Xtr.columns]
    score = np.zeros(len(Xte))
    for seed in SEEDS:
        m = make_model("cat", None, seed)
        m.fit(Xtr, D.y)
        score += pd.Series(m.predict_proba(Xte)[:, 1]).rank(pct=True).values / len(SEEDS)
        print("seed", seed, "done", flush=True)
    score = (score - score.min()) / (score.max() - score.min())
    sub = pd.DataFrame({"cookie_id": D.te["cookie_id"].values, "score": score})
    test = D.te
    assert len(sub) == len(test)
    assert sub.cookie_id.nunique() == len(test)
    assert sub.score.notna().all()
    assert sub.score.between(0, 1).all()
    assert set(sub.cookie_id) == set(test.cookie_id)
    sub.to_csv(ROOT / "submission.csv", index=False)
    print("saved submission.csv", sub.shape)


if __name__ == "__main__":
    main()

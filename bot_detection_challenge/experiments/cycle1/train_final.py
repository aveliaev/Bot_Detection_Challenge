"""Final pipeline: features -> full-train fit of ensemble members -> rank blend -> submission.csv

Run from bot_detection_challenge/:  python solution/train_final.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cv import Data, POOL_COLS, make_model, WORK, ROOT  # noqa: E402
from config import MEMBERS, SEEDS, HIST_DAYS  # noqa: E402


def rank01(x):
    return pd.Series(x).rank(pct=True, method="average").values


def main():
    D = Data(rebuild=True)
    D.hist_days = HIST_DAYS
    cols = json.load(open(Path(__file__).resolve().parent / "feature_list.json"))
    # pool features: train rows use strictly earlier train labels (fixed-length history),
    # test rows use the last HIST_DAYS labelled days (all of them precede the test period)
    P_tr, P_te = D.pool_full_train(), D.pool_test()
    Xtr = D.F.loc[D.ids].join(P_tr)[cols]
    Xte = D.F.loc[D.te["cookie_id"].values].join(P_te)[cols]
    y = D.y
    blend = np.zeros(len(Xte))
    wsum = 0.0
    for m in MEMBERS:
        pred = np.zeros(len(Xte))
        for seed in SEEDS:
            mdl = make_model(m["model"], m.get("params"), seed, m.get("pos_weight", 1.0))
            mdl.fit(Xtr, y)
            pred += rank01(mdl.predict_proba(Xte)[:, 1]) / len(SEEDS)
        print(f"{m['name']}: fitted {len(SEEDS)} seeds", flush=True)
        blend += m["weight"] * pred
        wsum += m["weight"]
    score = blend / wsum
    # map to [0,1]; ranks preserve ordering, ties only where members tie everywhere
    score = (score - score.min()) / (score.max() - score.min())
    sub = pd.DataFrame({"cookie_id": D.te["cookie_id"].values, "score": score})
    test = D.te
    assert len(sub) == len(test)
    assert sub.cookie_id.nunique() == len(test)
    assert sub.score.notna().all()
    assert sub.score.between(0, 1).all()
    assert set(sub.cookie_id) == set(test.cookie_id)
    out = ROOT / "submission.csv"
    sub.to_csv(out, index=False)
    print("saved", out, sub.shape, "n unique scores", sub.score.nunique())


if __name__ == "__main__":
    main()

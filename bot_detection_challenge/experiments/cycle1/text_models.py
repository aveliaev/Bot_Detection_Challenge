"""Sparse TF-IDF + linear models on per-cookie documents.

docs:
  ua  - concatenated distinct user agents of the cookie (char n-grams)
  seq - event sequence tokens: event short name + log-binned gap to previous event,
        plus uni/bi/tri-grams over these tokens (word n-grams)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from features import EV_SHORT


def gap_bin(dt):
    b = np.full(len(dt), "s", dtype=object)  # start
    edges = [(0, 0.5, "z"), (0.5, 2.5, "a"), (2.5, 12.5, "b"), (12.5, 30.5, "c"), (30.5, 90.5, "d"),
             (90.5, 300.5, "e"), (300.5, 1800.5, "f"), (1800.5, 1e12, "g")]
    for lo, hi, n in edges:
        b[(dt >= lo) & (dt < hi)] = n
    return b


def build_docs(ev: pd.DataFrame) -> pd.DataFrame:
    ev = ev.sort_values(["cookie_id", "event_ts", "eid"], kind="mergesort")
    t = ev["event_ts"].astype("int64").values // 10**6
    first = ~ev["cookie_id"].duplicated().values
    dt = np.r_[np.nan, np.diff(t)].astype(float)
    dt[first] = np.nan
    tok = ev["event_name"].map(EV_SHORT).values + "_" + gap_bin(dt)
    seq = pd.Series(tok, index=ev["cookie_id"].values).groupby(level=0).agg(" ".join)
    ua = ev.drop_duplicates(["cookie_id", "user_agent"]).groupby("cookie_id")["user_agent"].agg(" || ".join)
    return pd.DataFrame({"seq": seq, "ua": ua})


class TextLR:
    def __init__(self, kind="seq", C=0.5, seed=0):
        self.kind, self.C, self.seed = kind, C, seed

    def _vec(self):
        if self.kind == "ua":
            return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=5, sublinear_tf=True)
        return TfidfVectorizer(analyzer="word", token_pattern=r"\S+", ngram_range=(1, 3), min_df=5,
                               sublinear_tf=True, max_features=50000)

    def fit(self, docs: pd.Series, y):
        self.v = self._vec()
        X = self.v.fit_transform(docs.values)
        self.m = LogisticRegression(C=self.C, max_iter=3000, class_weight="balanced", random_state=self.seed)
        self.m.fit(X, y)
        return self

    def predict_proba(self, docs: pd.Series):
        return self.m.predict_proba(self.v.transform(docs.values))

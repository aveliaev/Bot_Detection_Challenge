"""Bot detection (Avito): final solution in one file.

Usage (from bot_detection_challenge/):
    python3 solution.py              # train on all of train, write submission.csv  (~2 min on 4 CPU)
    python3 solution.py --validate   # temporal validation: baselines vs the final model  (~10 min)

Pipeline
    1. events: keep only events inside the cookie's [window_start_ts, window_end_ts), drop exact duplicates, sort
    2. behavioural features of each cookie from its own in-window events (pointer, pauses, sessions,
       event mix, items / categories / cities, search, user agent)
    3. cross-cookie item features, strictly time-aware:
         * popularity   - how many other cookies (unlabelled, train + test) viewed the same items in days [D-6, D]
         * bot item pool - how many LABELLED bots / humans viewed the same items, labels of earlier days only
    4. CatBoost x 5 seeds, rank-averaged -> score in [0, 1] -> submission.csv

Every random component has a fixed seed (CatBoost random_seed 0..4, validation partitions RandomState(2024),
RandomForest baseline random_state 0).  cookie_id is only a join key, never a feature.
"""
from __future__ import annotations

import argparse
import warnings

from pathlib import Path

import numpy as np
import pandas as pd

from metric import precision_at_recall  # official metric implementation

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DAY0 = pd.Timestamp("2026-04-06")      # first train day; day index = (window_start_ts - DAY0).days
SEEDS = (0, 1, 2, 3, 4)
POOL_HISTORY_DAYS = 7                  # labelled history for the bot item pool (item lifetime is ~8 days)
CATBOOST = dict(iterations=1500, learning_rate=0.03, depth=6, l2_leaf_reg=5, rsm=0.5,
                verbose=0, thread_count=4, scale_pos_weight=1.0)

EVENTS = ["item_view", "search_results_view", "photo_swipe", "favorite_add", "seller_page_view",
          "contact_phone_show", "login", "contact_chat_open", "contact_message_sent"]
EV_SHORT = dict(zip(EVENTS, ["iv", "srv", "ps", "fav", "sp", "phone", "login", "chat", "msg"]))
DEDUP_KEY = ["cookie_id", "event_ts", "eid", "event_name", "platform", "user_agent", "item_id", "item_category",
             "item_location", "seller_type", "search_query", "search_page", "pointer_x", "pointer_y"]

# Behavioural features used by the model: top-115 of ~300 candidates by CatBoost importance (cycle 1, dropping
# redundant families did not hurt, see experiments/). The ORDER is part of the model: CatBoost samples columns
# (rsm=0.5) by position, so changing it changes the submission.
FEATURES = [
    "ptr_x_std", "ptr_x_max", "sdt_q90", "ptr_exact_edge_frac", "ptr_dist_mean", "frac_contact", "cat_nuniq",
    "cat_ent", "ev_per_item", "sdt_log_mean", "page_mean", "item_repeat_ratio", "sess_rate", "ptr_dist_std",
    "ddt_mean_rel", "sdt_band_90_1800", "loc_nuniq", "loc_switch_frac", "loc_ent", "cookie_age_at_first_h",
    "sdt_cv", "active_min_per_event", "ptr_x_gt1300_frac", "ptr_box_frac", "mean_per_active_min",
    "cat_switch_frac", "sdt_band_12_30", "sdt_median", "secbin_nent", "ddt_median", "sdt_log_std",
    "ddt_lt2_frac", "page_median", "dt_lt60", "dt_lt30", "sdt_mean", "ua_rarity", "loc_top1", "dt_q75",
    "item_nent", "hour_sin", "sdt_band_30_90", "item_ent", "sdt_std", "cat_top1", "act_on_viewed_frac",
    "bg_iv_sp", "cookie_age_d", "sdt_log_skew", "dt_iqr", "query_len_mean", "ptr_zero_frac", "dtbin_ent",
    "sdt_band_2_20", "item_top1", "ptr_dist_med", "ua_browser_ver", "frac_engage", "r_fav_iv", "ptr_n_x_gt1300",
    "page_std", "dt_median", "dtexact_uniq_ratio", "bg_iv_iv", "pop_logmean", "loc_nent", "dwell_med_iv",
    "dt_lt120", "dt_std", "n_gap_5_30min", "created_hour", "sel_pro_frac", "t_first", "item_id_std",
    "mean_per_active_5min", "item_id_mean", "sess_mean_dur", "sdt_q10", "dt_cv", "dtbin_top1", "ua_len",
    "bg_sp_iv", "t_last", "query_latin", "frac_sp", "sess_mean_n", "ptr_y_std", "ptr_y_extreme_frac",
    "ptr_x_mean", "bg_srv_iv", "page_max", "ptr_y_center_dev", "pop_mean", "sel_private_frac", "cat_nent",
    "same_ev_adj_frac", "dwell_med_srv", "sel_ent", "ptr_none_web", "log_cookie_age_d", "bigram_ent", "frac_iv",
    "ua_app_ver", "frac_phone", "item_id_absdiff_med", "frac_ps", "bg_iv_ps", "dt_q90", "hour_cos",
    "mean_per_active_hour", "loc_big4_frac", "frac_browse", "ptr_edge_frac", "bg_fav_iv", "sess_max_dur",
]
POOL_COLS = ["pool_rate_mean", "pool_rate_max", "pool_llr_mean", "pool_llr_sum", "pool_llr_max",
             "pool_nb_ge1_frac", "pool_nb_ge2_frac", "pool_nb_ge1_cnt", "pool_known_frac"]


# ============================================================================ 1. data
def load_data():
    """-> train, test, meta (train + test), in-window events."""
    kw = dict(parse_dates=["cookie_created_at", "window_start_ts", "window_end_ts"])
    tr = pd.read_csv(DATA / "train.csv", **kw)
    te = pd.read_csv(DATA / "test.csv", **kw)
    meta = pd.concat([tr.assign(part="tr"), te.assign(part="te", target=np.nan)], ignore_index=True)

    ev = pd.read_csv(DATA / "events.csv.gz")
    ev["event_ts"] = pd.to_datetime(ev["event_ts"])
    # ~1.5% exact duplicate rows, same rate in train and test and proportional to activity -> logging artefact
    ev = ev.drop_duplicates(subset=DEDUP_KEY)
    ev = ev.merge(meta[["cookie_id", "window_start_ts", "window_end_ts", "cookie_created_at"]], on="cookie_id")
    # Train contains events AFTER window_end_ts (42% of bot events vs 11% of human, incl. captcha_shown),
    # test does not -> using them is target leakage. Only the observation window is used.
    ev = ev[(ev.event_ts >= ev.window_start_ts) & (ev.event_ts < ev.window_end_ts)]
    # file rows are not ordered; eid breaks ties of equal timestamps
    ev = ev.sort_values(["cookie_id", "event_ts", "eid"], kind="mergesort").reset_index(drop=True)
    return tr, te, meta, ev


# ============================================================================ 2. per-cookie features
def _entropy(df, gcol, vcol, prefix):
    """Per group: number of distinct values, entropy, normalised entropy, share of the top value (NaN ignored)."""
    d = df[[gcol, vcol]].dropna()
    c = d.groupby([gcol, vcol], observed=True).size().rename("c").reset_index()
    p = c["c"] / c.groupby(gcol)["c"].transform("sum")
    c["h"], c["p"] = -p * np.log(p), p
    g = c.groupby(gcol)
    out = pd.DataFrame({f"{prefix}_nuniq": g.size(), f"{prefix}_ent": g["h"].sum(), f"{prefix}_top1": g["p"].max()})
    n = out[f"{prefix}_nuniq"]
    out[f"{prefix}_nent"] = np.where(n > 1, out[f"{prefix}_ent"] / np.log(n.clip(lower=2)), 0.0)
    out[f"{prefix}_uniq_ratio"] = n / g["c"].sum()
    return out


def cookie_features(ev: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Behavioural features of every cookie from its own in-window events only.
    Missing values stay NaN (no pointer / no search / single event ...): CatBoost handles NaN natively and
    "the value is absent" is itself informative (e.g. web cookies without pointer: 18% of bots vs ~6%)."""
    ev = ev.copy()
    cid = ev["cookie_id"]
    g = ev.groupby("cookie_id", sort=True)
    F = pd.DataFrame(index=pd.Index(np.sort(meta["cookie_id"].values), name="cookie_id"))

    ev["t"] = ev["event_ts"].astype("int64") // 10**6                                  # seconds
    ev["tw"] = ev["t"] - ev["window_start_ts"].astype("int64") // 10**6                # seconds since window start
    first_row = ~cid.duplicated()
    ev["dt"] = np.where(first_row, np.nan, ev["t"] - ev["t"].shift(1))                  # pause before the event
    ev["prev_ev"] = np.where(first_row, "START", ev["event_name"].shift(1))
    n = g.size()

    # ---- activity: timing inside the window, activity density, sessions (gap > 30 min), cookie age
    F["t_first"] = g["tw"].min() / 3600
    F["t_last"] = g["tw"].max() / 3600
    for w, name in [(60, "min"), (300, "5min"), (3600, "hour")]:
        per_bin = ev.groupby([cid, ev["tw"] // w]).size()
        F[f"mean_per_active_{name}"] = per_bin.groupby(level=0).mean()
        if name == "min":
            F["active_min_per_event"] = per_bin.groupby(level=0).size() / n
    ev["sess"] = (first_row | (ev["dt"] > 1800)).cumsum()
    sg = ev.groupby("sess").agg(cookie_id=("cookie_id", "first"), s_n=("t", "size"),
                                s_dur=("t", lambda x: x.max() - x.min()))
    sgc = sg.groupby("cookie_id")
    F["sess_mean_n"] = sgc["s_n"].mean()
    F["sess_mean_dur"] = sgc["s_dur"].mean()
    F["sess_max_dur"] = sgc["s_dur"].max()
    F["sess_rate"] = (sg["s_n"] / (sg["s_dur"] + 60)).groupby(sg["cookie_id"]).max()
    F["n_gap_5_30min"] = ev.assign(x=(ev.dt > 300) & (ev.dt <= 1800)).groupby("cookie_id")["x"].sum()
    m = meta.set_index("cookie_id").loc[F.index]
    F["cookie_age_d"] = ((m["window_start_ts"] - m["cookie_created_at"]).dt.total_seconds() / 86400).values
    F["cookie_age_at_first_h"] = ((g["event_ts"].min() - m["cookie_created_at"]).dt.total_seconds() / 3600).values
    F["created_hour"] = m["cookie_created_at"].dt.hour.values + m["cookie_created_at"].dt.minute.values / 60
    F["log_cookie_age_d"] = np.log1p(F["cookie_age_d"].clip(lower=0))
    hr = ev["tw"] / 3600 * 2 * np.pi / 24
    F["hour_sin"] = np.sin(hr).groupby(cid).mean()
    F["hour_cos"] = np.cos(hr).groupby(cid).mean()
    F["secbin_nent"] = _entropy(ev.assign(s=ev["t"] % 60 // 6), "cookie_id", "s", "secbin")["secbin_nent"]

    # ---- pauses between events. EDA: 58% of bot in-session pauses are 3..20 s (humans 20%), pauses of 0..2 s
    # are RARER for bots (scripted minimum delay), and bot pauses are more regular.
    d = ev.dropna(subset=["dt"])
    gd = d.groupby("cookie_id")["dt"]
    F["dt_median"], F["dt_std"] = gd.median(), gd.std()
    F["dt_q75"], F["dt_q90"] = gd.quantile(0.75), gd.quantile(0.9)
    F["dt_iqr"] = F["dt_q75"] - gd.quantile(0.25)
    F["dt_cv"] = F["dt_std"] / (gd.mean() + 1e-9)
    for th in [30, 60, 120]:
        F[f"dt_lt{th}"] = (d["dt"] < th).groupby(d["cookie_id"]).mean()
    ds = d[d["dt"] <= 1800]                                                    # in-session pauses
    gds = ds.groupby("cookie_id")["dt"]
    F["sdt_mean"], F["sdt_median"], F["sdt_std"] = gds.mean(), gds.median(), gds.std()
    F["sdt_cv"] = F["sdt_std"] / (F["sdt_mean"] + 1e-9)
    F["sdt_q10"], F["sdt_q90"] = gds.quantile(0.1), gds.quantile(0.9)
    ld = np.log1p(ds["dt"]).groupby(ds["cookie_id"])
    F["sdt_log_std"], F["sdt_log_mean"], F["sdt_log_skew"] = ld.std(), ld.mean(), ld.skew()
    for lo, hi in [(2.5, 20.5), (12.5, 30.5), (30.5, 90.5), (90.5, 1800.5)]:
        F[f"sdt_band_{int(lo)}_{int(hi)}"] = ((ds["dt"] > lo) & (ds["dt"] < hi)).groupby(ds["cookie_id"]).mean()
    for e_ in ["item_view", "search_results_view"]:
        F[f"dwell_med_{EV_SHORT[e_]}"] = ds[ds["event_name"] == e_].groupby("cookie_id")["dt"].median()
    # regularity: entropy of log-binned pauses, share of distinct exact pauses, change between successive pauses
    sd = ds.assign(r=ds["dt"].clip(upper=600))
    e = _entropy(sd.assign(b=(np.log1p(sd["r"]) * 4).round()), "cookie_id", "b", "dtbin")
    F = F.join(e[["dtbin_ent", "dtbin_top1"]])
    F = F.join(_entropy(sd, "cookie_id", "r", "dtexact")[["dtexact_uniq_ratio"]])
    dd = ds.copy()
    dd["ddt"] = np.where(dd["cookie_id"].eq(dd["cookie_id"].shift(1)), (dd["dt"] - dd["dt"].shift(1)).abs(), np.nan)
    dd["rel"] = dd["ddt"] / (dd[["dt"]].join(dd["dt"].shift(1).rename("p")).max(axis=1) + 1)
    gdd = dd.dropna(subset=["ddt"]).groupby("cookie_id")
    F["ddt_mean_rel"], F["ddt_median"] = gdd["rel"].mean(), gdd["ddt"].median()
    F["ddt_lt2_frac"] = (dd["ddt"] < 2).groupby(dd["cookie_id"]).mean()

    # ---- event mix and transitions
    ec = pd.crosstab(ev["cookie_id"], ev["event_name"])
    cnt = {EV_SHORT[e_]: (ec[e_] if e_ in ec else 0) for e_ in EVENTS}
    for s in ["iv", "sp", "phone", "ps"]:
        F[f"frac_{s}"] = cnt[s] / n
    cnt = pd.DataFrame(cnt).reindex(F.index).fillna(0)
    contact = cnt["phone"] + cnt["chat"] + cnt["msg"]
    F["frac_contact"] = contact / n
    F["r_fav_iv"] = cnt["fav"] / (cnt["iv"] + 1.0)
    F["frac_engage"] = (cnt["ps"] + cnt["fav"] + contact + cnt["login"]) / n
    F["frac_browse"] = (cnt["iv"] + cnt["srv"] + cnt["sp"]) / n
    tr = ev[~first_row]
    big = tr["prev_ev"].map(EV_SHORT) + ">" + tr["event_name"].map(EV_SHORT)
    bc = pd.crosstab(tr["cookie_id"], big)
    for b in ["iv>iv", "iv>sp", "sp>iv", "srv>iv", "iv>ps", "fav>iv"]:
        F["bg_" + b.replace(">", "_")] = bc[b] / bc.sum(axis=1)
    F["bigram_ent"] = _entropy(tr.assign(b=big), "cookie_id", "b", "bigram")["bigram_ent"]
    F["same_ev_adj_frac"] = (tr["prev_ev"] == tr["event_name"]).groupby(tr["cookie_id"]).mean()

    # ---- items, categories, cities, sellers. Bots browse more cities / categories and switch between them more.
    it = ev.dropna(subset=["item_id"])
    gi = it.groupby("cookie_id")
    n_items = gi["item_id"].nunique()
    F["item_repeat_ratio"] = 1 - n_items / gi.size()
    F["ev_per_item"] = gi.size() / n_items
    viewed = it[it.event_name == "item_view"][["cookie_id", "item_id"]].drop_duplicates().assign(v=1)
    act = it[it.event_name != "item_view"].merge(viewed, on=["cookie_id", "item_id"], how="left")
    F["act_on_viewed_frac"] = act["v"].fillna(0).groupby(act["cookie_id"]).mean()
    # item_id grows with listing creation -> its level / spread is a proxy of the age of the listings browsed
    F["item_id_mean"], F["item_id_std"] = gi["item_id"].mean(), gi["item_id"].std()
    step = (it["item_id"] - it["item_id"].shift(1)).where(it["cookie_id"].eq(it["cookie_id"].shift(1)))
    F["item_id_absdiff_med"] = step.abs().groupby(it["cookie_id"]).median()
    for col, p in [("item_category", "cat"), ("item_location", "loc"), ("seller_type", "sel"), ("item_id", "item")]:
        F = F.join(_entropy(ev, "cookie_id", col, p)[[f"{p}_nuniq", f"{p}_ent", f"{p}_nent", f"{p}_top1"]])
    for col, p in [("item_location", "loc"), ("item_category", "cat")]:
        x = ev.dropna(subset=[col])
        sw = x[col].ne(x[col].shift(1)) & x["cookie_id"].eq(x["cookie_id"].shift(1))
        F[f"{p}_switch_frac"] = sw.groupby(x["cookie_id"]).mean()
    F["sel_pro_frac"] = (it["seller_type"] == "pro").groupby(it["cookie_id"]).mean()
    F["sel_private_frac"] = (it["seller_type"] == "private").groupby(it["cookie_id"]).mean()
    big4 = ev["item_location"].isin(["moskva", "sankt-peterburg", "novosibirsk", "ekaterinburg"])
    F["loc_big4_frac"] = big4.groupby(cid).mean()

    # ---- search: bots go deeper into result pages (mean page 3.7 vs 2.0)
    s = ev[ev.event_name == "search_results_view"]
    gs = s.groupby("cookie_id")["search_page"]
    F["page_mean"], F["page_median"], F["page_max"], F["page_std"] = gs.mean(), gs.median(), gs.max(), gs.std()
    F["query_len_mean"] = s["search_query"].str.len().groupby(s["cookie_id"]).mean()
    F["query_latin"] = s["search_query"].str.contains(r"[a-z]", na=False).groupby(s["cookie_id"]).mean()

    # ---- pointer (web only). EDA: human coordinates are ~uniform over 1920x1080; bot coordinates concentrate in a
    # central box (x < ~1250 for 90% of events) plus exact 0 / 1080 / 1920 edges.
    # `platform` spelling is random per event (WEB/Web/web/desktop) -> compare lower-cased values.
    web = ev["platform"].str.lower().isin(["web", "desktop"])
    has_ptr = ev["pointer_x"].notna()
    pw = has_ptr[web]
    is_web = web.groupby(cid).max().astype(int).reindex(F.index)
    ptr_frac_web = pw.groupby(ev.loc[web, "cookie_id"]).mean().reindex(F.index)
    F["ptr_none_web"] = ((ptr_frac_web == 0) & (is_web == 1)).astype(int)
    p = ev[has_ptr]
    px, py, pc = p["pointer_x"], p["pointer_y"], p["cookie_id"]
    F["ptr_x_mean"], F["ptr_x_std"], F["ptr_x_max"] = px.groupby(pc).mean(), px.groupby(pc).std(), px.groupby(pc).max()
    F["ptr_y_std"] = py.groupby(pc).std()
    F["ptr_zero_frac"] = ((px == 0) | (py == 0)).groupby(pc).mean()
    F["ptr_edge_frac"] = ((py >= 1080) | (px >= 1920)).groupby(pc).mean()
    F["ptr_exact_edge_frac"] = ((px == 0) | (py == 0) | (py == 1080) | (px == 1920)).groupby(pc).mean()
    F["ptr_x_gt1300_frac"] = (px > 1300).groupby(pc).mean()
    F["ptr_n_x_gt1300"] = (px > 1300).groupby(pc).sum()
    F["ptr_y_extreme_frac"] = (((py > 0) & (py < 100)) | ((py > 950) & (py < 1080))).groupby(pc).mean()
    F["ptr_box_frac"] = ((px >= 300) & (px <= 1200) & (py >= 300) & (py <= 900)).groupby(pc).mean()
    F["ptr_y_center_dev"] = (py - 540).abs().groupby(pc).mean()
    same = pc.eq(pc.shift(1))
    dist = np.sqrt((px - px.shift(1)).where(same) ** 2 + (py - py.shift(1)).where(same) ** 2)
    F["ptr_dist_mean"], F["ptr_dist_std"], F["ptr_dist_med"] = (dist.groupby(pc).mean(), dist.groupby(pc).std(),
                                                                dist.groupby(pc).median())

    # ---- user agent: weak on its own (scraper libraries are only 19-27% of bots), kept as context
    ua = ev["user_agent"].fillna("")
    F["ua_len"] = ua.str.len().groupby(cid.values).max()
    F["ua_browser_ver"] = (ua.str.extract(r"(?:Chrome|Firefox|HeadlessChrome|rv)[/:](\d+)")[0].astype(float)
                           .groupby(cid.values).max())
    F["ua_app_ver"] = ua.str.extract(r"^Avito/(\d+)")[0].astype(float).groupby(cid.values).max()
    ua_freq = ev.drop_duplicates(["cookie_id", "user_agent"])["user_agent"].value_counts()   # unlabelled counts
    F["ua_rarity"] = np.log(ev["user_agent"].map(ua_freq)).groupby(cid).min()

    return F.replace([np.inf, -np.inf], np.nan)


# ============================================================================ 3. cross-cookie item features
def cookie_items(ev: pd.DataFrame) -> pd.DataFrame:
    """(cookie_id, item_id, number of views, day of the cookie's window)."""
    ci = ev.dropna(subset=["item_id"]).groupby(["cookie_id", "item_id"]).size().rename("nv").reset_index()
    day = ev.drop_duplicates("cookie_id").set_index("cookie_id")["window_start_ts"]
    ci["day"] = ((ci["cookie_id"].map(day) - DAY0).dt.days).astype(int)
    return ci


def popularity_features(ci: pd.DataFrame, span: int = 7) -> pd.DataFrame:
    """Unsupervised: how many OTHER cookies viewed the same items during days [D-span+1, D] (no labels,
    no events after the cookie's own day). Raw counts on purpose: normalising by the number of cookies per day
    put values on a day-specific lattice that separated train from test (adversarial AUC 0.995)."""
    out = []
    for D in np.sort(ci["day"].unique()):
        cur = ci[ci["day"] == D]
        cnt = ci[(ci["day"] <= D) & (ci["day"] > D - span)].groupby("item_id").size()
        out.append(pd.DataFrame({"cookie_id": cur["cookie_id"].values,
                                 "pop": (cur["item_id"].map(cnt).values - 1).astype(float)}))  # minus itself
    p = pd.concat(out)
    return pd.DataFrame({"pop_mean": p.groupby("cookie_id")["pop"].mean(),
                         "pop_logmean": np.log1p(p["pop"]).groupby(p["cookie_id"]).mean()})


def pool_features(ci: pd.DataFrame, labels: pd.Series, target_cookies, hist_days: int = POOL_HISTORY_DAYS,
                  m: float = 2.0, p0: float = 0.08) -> pd.DataFrame:
    """Bot item pool. EDA: bots draw item_ids from a shared, stable pool (52% of 2nd-week bot items were already
    viewed by 1st-week bots vs 9% for humans). For a target cookie of day D this counts, per viewed item, the
    labelled bots (nb) and humans (nh) that viewed it, using ONLY cookies from `labels` whose day is strictly
    earlier: days [E-hist_days, E-1], E = min(D, last labelled day + 1).
    In validation `labels` = training part of the fold; for test = the whole train.
    Fixed history length and a constant prior p0 keep train and test values on the same scale."""
    lab = ci[ci["cookie_id"].isin(labels.index)].copy()
    lab["y"] = lab["cookie_id"].map(labels).astype(int)
    lck = lab.drop_duplicates("cookie_id")[["cookie_id", "day", "y"]]
    last = int(lck["day"].max())
    tgt = ci[ci["cookie_id"].isin(set(target_cookies))]
    out = []
    for D in np.sort(tgt["day"].unique()):
        cur = tgt[tgt["day"] == D]
        E = min(D, last + 1)
        h = lab[(lab["day"] < E) & (lab["day"] >= E - hist_days)]
        hc = lck[(lck["day"] < E) & (lck["day"] >= E - hist_days)]
        if hc["y"].sum() == 0 or (1 - hc["y"]).sum() == 0:
            continue                                   # no labelled history yet -> features stay NaN
        agg = h.groupby("item_id")["y"].agg(["sum", "size"])
        nb = cur["item_id"].map(agg["sum"]).fillna(0).values
        nh = cur["item_id"].map(agg["size"]).fillna(0).values - nb
        out.append(pd.DataFrame({"cookie_id": cur["cookie_id"].values, "nb": nb, "nh": nh,
                                 "rate": (nb + m * p0) / (nb + nh + m), "llr": np.log((nb + 0.5) / (nh + 0.5))}))
    p = pd.concat(out)
    g = p.groupby("cookie_id")
    F = pd.DataFrame({
        "pool_rate_mean": g["rate"].mean(), "pool_rate_max": g["rate"].max(),
        "pool_llr_mean": g["llr"].mean(), "pool_llr_sum": g["llr"].sum(), "pool_llr_max": g["llr"].max(),
        "pool_nb_ge1_frac": (p["nb"] >= 1).groupby(p["cookie_id"]).mean(),
        "pool_nb_ge2_frac": (p["nb"] >= 2).groupby(p["cookie_id"]).mean(),
        "pool_nb_ge1_cnt": (p["nb"] >= 1).groupby(p["cookie_id"]).sum(),
        "pool_known_frac": ((p["nb"] + p["nh"]) > 0).groupby(p["cookie_id"]).mean(),
    })
    F.index.name = "cookie_id"
    return F


# ============================================================================ 4. model
class Dataset:
    def __init__(self):
        self.tr, self.te, meta, ev = load_data()
        ci = cookie_items(ev)
        self.ci = ci
        self.F = cookie_features(ev, meta).join(popularity_features(ci))[FEATURES]
        self.n_events = ev.groupby("cookie_id").size()
        self.n_items = ev.groupby("cookie_id")["item_id"].nunique()
        self.ids = self.tr["cookie_id"].values
        self.y = self.tr["target"].values
        self.labels = self.tr.set_index("cookie_id")["target"]
        self.day = (self.tr["window_start_ts"] - DAY0).dt.days.values

    def X(self, ids, labels):
        """Features of `ids`; the bot pool uses only `labels` (and only days earlier than each cookie's day)."""
        return self.F.loc[ids].join(pool_features(self.ci, labels, ids))


def catboost(seed):
    from catboost import CatBoostClassifier
    return CatBoostClassifier(random_seed=seed, **CATBOOST)


def train_and_predict(D: Dataset, tr_idx, pred_ids, seeds=SEEDS):
    """Fit on train rows tr_idx (pool from their labels only), return rank-averaged scores for pred_ids."""
    labels = D.labels.iloc[tr_idx]
    Xtr = D.X(D.ids[tr_idx], labels)
    Xp = D.X(pred_ids, labels)[Xtr.columns]
    score = np.zeros(len(pred_ids))
    for seed in seeds:
        m = catboost(seed).fit(Xtr, D.y[tr_idx])
        score += pd.Series(m.predict_proba(Xp)[:, 1]).rank(pct=True).values / len(seeds)
    return score


def make_submission(D: Dataset):
    test_ids = D.te["cookie_id"].values
    score = train_and_predict(D, np.arange(len(D.ids)), test_ids)
    score = (score - score.min()) / (score.max() - score.min())
    sub = pd.DataFrame({"cookie_id": test_ids, "score": score})
    assert len(sub) == len(D.te) and sub.cookie_id.is_unique and set(sub.cookie_id) == set(test_ids)
    assert sub.score.notna().all() and sub.score.between(0, 1).all()
    sub.to_csv(ROOT / "submission.csv", index=False)
    print("saved submission.csv", sub.shape)


# ============================================================================ 5. validation
def validation_schemes(day):
    """Test lies strictly after train (1..7 days after the last label), so validation is by days, never random.
    Days 0-1 always stay in training (the bot pool needs labelled history).
      P0..P4 : repeated day-block CV - 12 days (2..13) split into 6 blocks of 2 days; P0 = consecutive pairs,
               P1..P4 = random pairings (RandomState 2024). Spread between P0..P4 shows the metric noise.
      fwd7   : test-like forward check - train on days < k, validate on the next 7 days (k = 5, 7)."""
    parts = [[[2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13]]]
    rng = np.random.RandomState(2024)
    for _ in range(4):
        d = rng.permutation(np.arange(2, 14))
        parts.append([sorted(d[i:i + 2].tolist()) for i in range(0, 12, 2)])
    schemes = {f"P{i}": [(np.flatnonzero(~np.isin(day, b)), np.flatnonzero(np.isin(day, b))) for b in blocks]
               for i, blocks in enumerate(parts)}
    schemes["fwd7"] = [(np.flatnonzero(day < k), np.flatnonzero((day >= k) & (day < k + 7))) for k in (5, 7)]
    return schemes


def validate(D: Dataset):
    """Out-of-fold P@R>=0.70 per scheme. Every label-dependent feature (the bot pool) is rebuilt inside each fold
    from the fold's training labels only. One seed per fit to keep the run time reasonable."""
    from sklearn.ensemble import RandomForestClassifier

    def quickstart(tri, vai):          # organisers' quickstart baseline: RandomForest on 2 activity counts
        X = pd.DataFrame({"n_events": D.n_events, "n_items": D.n_items}).reindex(D.ids).fillna(0)
        m = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=0)
        return m.fit(X.iloc[tri], D.y[tri]).predict_proba(X.iloc[vai])[:, 1]

    def no_pool(tri, vai):             # same CatBoost, per-cookie features only (no labelled item pool)
        m = catboost(0).fit(D.F.loc[D.ids[tri]], D.y[tri])
        return m.predict_proba(D.F.loc[D.ids[vai]])[:, 1]

    def final(tri, vai):               # the final model
        return train_and_predict(D, tri, D.ids[vai], seeds=(0,))

    models = {"quickstart baseline (RF, n_events + n_items)": quickstart,
              "CatBoost, cookie features only": no_pool,
              "final: CatBoost + bot item pool": final}
    rows = []
    for name, fit in models.items():
        r = {"model": name}
        for sch, folds in validation_schemes(D.day).items():
            oof = np.full(len(D.y), np.nan)
            for tri, vai in folds:     # fwd7 folds overlap: the later fold (more history) overwrites
                oof[vai] = fit(tri, vai)
            ok = ~np.isnan(oof)
            r[sch] = precision_at_recall(D.y[ok], oof[ok])
        rep = [r[f"P{i}"] for i in range(5)]
        r["repeated mean"], r["repeated std"] = np.mean(rep), np.std(rep)
        rows.append(r)
        print(pd.DataFrame([r]).round(4).to_string(index=False), flush=True)
    print("\nP@R>=0.70 (out-of-fold). Constant score = bot share = 0.081.")
    print(pd.DataFrame(rows).set_index("model").round(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--validate", action="store_true", help="run temporal validation instead of the submission")
    args = ap.parse_args()
    data = Dataset()
    if args.validate:
        validate(data)
    else:
        make_submission(data)

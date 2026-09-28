"""Feature engineering for cookie-level bot detection.

Only events with window_start_ts <= event_ts < window_end_ts are used
(train contains post-window events incl. captcha_shown -> target leakage).
Exact duplicate event rows (~1.5%, same rate in train/test) are removed.
All features are computed per cookie from its own in-window events; the only
cross-cookie statistics (item popularity / labelled bot item pool) are in history.py
and are strictly time-aware.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
EVENTS = ["item_view", "search_results_view", "photo_swipe", "favorite_add", "seller_page_view",
          "contact_phone_show", "login", "contact_chat_open", "contact_message_sent"]
EV_SHORT = dict(zip(EVENTS, ["iv", "srv", "ps", "fav", "sp", "phone", "login", "chat", "msg"]))
KEY = ["cookie_id", "event_ts", "eid", "event_name", "platform", "user_agent", "item_id", "item_category",
       "item_location", "seller_type", "search_query", "search_page", "pointer_x", "pointer_y"]


def load_meta():
    kw = dict(parse_dates=["cookie_created_at", "window_start_ts", "window_end_ts"])
    tr = pd.read_csv(DATA / "train.csv", **kw)
    te = pd.read_csv(DATA / "test.csv", **kw)
    meta = pd.concat([tr.assign(part="tr"), te.assign(part="te", target=np.nan)], ignore_index=True)
    return tr, te, meta


def load_events(meta):
    ev = pd.read_csv(DATA / "events.csv.gz")
    ev["event_ts"] = pd.to_datetime(ev["event_ts"])
    ev = ev.drop_duplicates(subset=KEY)
    ev = ev.merge(meta[["cookie_id", "window_start_ts", "window_end_ts", "cookie_created_at"]], on="cookie_id")
    ev = ev[(ev.event_ts >= ev.window_start_ts) & (ev.event_ts < ev.window_end_ts)]
    ev = ev.sort_values(["cookie_id", "event_ts", "eid"], kind="mergesort").reset_index(drop=True)
    return ev


# ----------------------------------------------------------------------------- helpers
def _entropy_from_counts(df, gcol, vcol, prefix):
    """entropy, normalized entropy, top1 share, nunique of vcol within gcol (NaN ignored)."""
    d = df[[gcol, vcol]].dropna()
    c = d.groupby([gcol, vcol], observed=True).size().rename("c").reset_index()
    tot = c.groupby(gcol)["c"].transform("sum")
    p = c["c"] / tot
    c["h"] = -p * np.log(p)
    c["p"] = p
    g = c.groupby(gcol)
    out = pd.DataFrame({
        f"{prefix}_nuniq": g.size(),
        f"{prefix}_ent": g["h"].sum(),
        f"{prefix}_top1": g["p"].max(),
    })
    n = out[f"{prefix}_nuniq"]
    out[f"{prefix}_nent"] = np.where(n > 1, out[f"{prefix}_ent"] / np.log(n.clip(lower=2)), 0.0)
    out[f"{prefix}_uniq_ratio"] = n / g["c"].sum()
    return out


def _run_lengths(key_series, same_prev):
    """max run length of consecutive equal values per cookie given boolean same_prev."""
    run_id = (~same_prev).cumsum()
    rl = run_id.groupby(run_id).transform("size")
    return rl


def parse_ua(ua: pd.Series) -> pd.DataFrame:
    u = ua.fillna("")
    out = pd.DataFrame(index=ua.index)
    lib = u.str.contains(r"^(?:python-|node-fetch|curl/|Go-http|Scrapy|wget|Java/|axios)", regex=True)
    out["ua_is_lib"] = lib.astype(int)
    out["ua_headless"] = u.str.contains("HeadlessChrome").astype(int)
    out["ua_app"] = u.str.startswith("Avito/").astype(int)
    out["ua_yabro"] = u.str.contains("YaBrowser").astype(int)
    out["ua_firefox"] = u.str.contains("Firefox").astype(int)
    out["ua_mac"] = u.str.contains("Macintosh").astype(int)
    out["ua_win"] = u.str.contains("Windows").astype(int)
    out["ua_linux_x11"] = u.str.contains("X11").astype(int)
    out["ua_android"] = u.str.contains("Android").astype(int)
    out["ua_iphone"] = u.str.contains("iPhone").astype(int)
    out["ua_mobile_safari"] = u.str.contains("Mobile Safari").astype(int)
    out["ua_len"] = u.str.len()
    ver = u.str.extract(r"(?:Chrome|Firefox|HeadlessChrome|rv)[/:](\d+)")[0].astype(float)
    out["ua_browser_ver"] = ver
    out["ua_app_ver"] = u.str.extract(r"^Avito/(\d+)")[0].astype(float)
    out["ua_android_ver"] = u.str.extract(r"Android (\d+)")[0].astype(float)
    out["ua_ios_ver"] = u.str.extract(r"iOS (\d+)")[0].astype(float)
    return out


warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# ----------------------------------------------------------------------------- main
def build_features(ev: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    ev = ev.copy()
    cid = ev["cookie_id"]
    g = ev.groupby("cookie_id", sort=True)
    F = pd.DataFrame(index=meta["cookie_id"].sort_values().values)
    F.index.name = "cookie_id"

    ts = ev["event_ts"].astype("int64") // 10**6  # seconds (datetime64[us])
    ev["t"] = ts
    ws = ev["window_start_ts"].astype("int64") // 10**6
    ev["tw"] = ev["t"] - ws  # seconds since window start
    first_row = ~cid.duplicated()
    prev_t = ev["t"].shift(1)
    ev["dt"] = np.where(first_row, np.nan, ev["t"] - prev_t)
    prev_ev = ev["event_name"].shift(1)
    ev["prev_ev"] = np.where(first_row, "START", prev_ev)

    # ---------------- A. activity
    n = g.size()
    F["n_events"] = n
    F["n_event_types"] = g["event_name"].nunique()
    F["t_first"] = g["tw"].min() / 3600
    F["t_last"] = g["tw"].max() / 3600
    F["duration_s"] = g["t"].max() - g["t"].min()
    F["duration_h"] = F["duration_s"] / 3600
    F["ev_per_hour_active"] = n / (F["duration_s"] / 3600 + 1 / 60)
    F["log_n"] = np.log1p(n)
    for w, name in [(60, "min"), (300, "5min"), (900, "15min"), (3600, "hour")]:
        b = ev["tw"] // w
        cnt = ev.groupby([cid, b]).size()
        F[f"n_active_{name}"] = cnt.groupby(level=0).size()
        F[f"max_per_{name}"] = cnt.groupby(level=0).max()
        F[f"mean_per_active_{name}"] = cnt.groupby(level=0).mean()
    F["frac_active_hours"] = F["n_active_hour"] / 24
    F["active_min_per_event"] = F["n_active_min"] / n

    # sessions: gap > 30 min
    ev["new_sess"] = first_row | (ev["dt"] > 1800)
    ev["sess"] = ev["new_sess"].cumsum()
    sg = ev.groupby("sess").agg(cookie_id=("cookie_id", "first"), s_n=("t", "size"),
                                s_dur=("t", lambda x: x.max() - x.min()))
    sgc = sg.groupby("cookie_id")
    F["n_sessions"] = sgc.size()
    F["sess_mean_n"] = sgc["s_n"].mean()
    F["sess_max_n"] = sgc["s_n"].max()
    F["sess_mean_dur"] = sgc["s_dur"].mean()
    F["sess_max_dur"] = sgc["s_dur"].max()
    F["sess_rate"] = (sg["s_n"] / (sg["s_dur"] + 60)).groupby(sg["cookie_id"]).max()
    # gaps between 5 and 30 min (typical "break" region)
    F["n_gap_5_30min"] = ev.assign(x=(ev.dt > 300) & (ev.dt <= 1800)).groupby("cookie_id")["x"].sum()

    # cookie age
    m = meta.set_index("cookie_id").loc[F.index]
    F["cookie_age_d"] = ((m["window_start_ts"] - m["cookie_created_at"]).dt.total_seconds() / 86400).values
    F["cookie_age_at_first_h"] = (
        (g["event_ts"].min() - m["cookie_created_at"]).dt.total_seconds() / 3600).values
    F["created_in_window"] = (m["cookie_created_at"] >= m["window_start_ts"]).astype(int).values
    F["created_hour"] = m["cookie_created_at"].dt.hour.values + m["cookie_created_at"].dt.minute.values / 60
    F["log_cookie_age_d"] = np.log1p(F["cookie_age_d"].clip(lower=0))

    # ---------------- B. inter-event times
    d = ev.dropna(subset=["dt"])
    gd = d.groupby("cookie_id")["dt"]
    F["dt_min"] = gd.min()
    F["dt_max"] = gd.max()
    F["dt_mean"] = gd.mean()
    F["dt_median"] = gd.median()
    F["dt_std"] = gd.std()
    for q in [0.1, 0.25, 0.75, 0.9]:
        F[f"dt_q{int(q*100)}"] = gd.quantile(q)
    F["dt_iqr"] = F["dt_q75"] - F["dt_q25"]
    F["dt_cv"] = F["dt_std"] / (F["dt_mean"] + 1e-9)
    for th in [1, 2, 5, 10, 30, 60, 120, 300]:
        F[f"dt_lt{th}"] = (d["dt"] < th).groupby(d["cookie_id"]).mean()
    F["dt_zero_n"] = (d["dt"] == 0).groupby(d["cookie_id"]).sum()
    # in-session deltas (<= 30 min): human think-time distribution vs scripted
    ds = d[d["dt"] <= 1800]
    gds = ds.groupby("cookie_id")["dt"]
    F["sdt_mean"] = gds.mean()
    F["sdt_median"] = gds.median()
    F["sdt_std"] = gds.std()
    F["sdt_cv"] = F["sdt_std"] / (F["sdt_mean"] + 1e-9)
    ld = np.log1p(ds["dt"])
    F["sdt_log_std"] = ld.groupby(ds["cookie_id"]).std()
    F["sdt_log_mean"] = ld.groupby(ds["cookie_id"]).mean()
    F["sdt_log_skew"] = ld.groupby(ds["cookie_id"]).skew()
    sd = ds.assign(r=ds["dt"].clip(upper=600))
    # EDA: bot in-session gaps concentrate in 3..20 s and are rarely 0..2 s (scripted min delay)
    for lo, hi in [(0.5, 2.5), (2.5, 12.5), (2.5, 20.5), (12.5, 30.5), (30.5, 90.5), (90.5, 1800.5)]:
        F[f"sdt_band_{int(lo)}_{int(hi)}"] = ((ds["dt"] > lo) & (ds["dt"] < hi)).groupby(ds["cookie_id"]).mean()
    F["sdt_n_3_20"] = ((ds["dt"] >= 3) & (ds["dt"] <= 20)).groupby(ds["cookie_id"]).sum()
    for e_ in ["item_view", "search_results_view", "photo_swipe"]:
        q = ds[ds["event_name"] == e_]
        F[f"dwell_med_{EV_SHORT[e_]}"] = q.groupby("cookie_id")["dt"].median()
    F["sdt_q10"] = gds.quantile(0.1)
    F["sdt_q90"] = gds.quantile(0.9)
    # regularity: entropy of rounded dt, share of most common dt, successive-dt differences
    e = _entropy_from_counts(sd.assign(b=(np.log1p(sd["r"]) * 4).round()), "cookie_id", "b", "dtbin")
    F = F.join(e[["dtbin_ent", "dtbin_top1", "dtbin_nuniq"]])
    e2 = _entropy_from_counts(sd, "cookie_id", "r", "dtexact")
    F = F.join(e2[["dtexact_top1", "dtexact_uniq_ratio"]])
    dd = ds.copy()
    same_ck = dd["cookie_id"].eq(dd["cookie_id"].shift(1))
    dd["ddt"] = np.where(same_ck, (dd["dt"] - dd["dt"].shift(1)).abs(), np.nan)
    dd["rel"] = dd["ddt"] / (dd[["dt"]].join(dd["dt"].shift(1).rename("p")).max(axis=1) + 1)
    gdd = dd.dropna(subset=["ddt"]).groupby("cookie_id")
    F["ddt_mean_rel"] = gdd["rel"].mean()
    F["ddt_median"] = gdd["ddt"].median()
    F["ddt_lt2_frac"] = (dd["ddt"] < 2).groupby(dd["cookie_id"]).mean()

    # ---------------- C. event types
    ec = pd.crosstab(ev["cookie_id"], ev["event_name"])
    for e_ in EVENTS:
        s = EV_SHORT[e_]
        c = ec[e_] if e_ in ec else 0
        F[f"cnt_{s}"] = c
        F[f"frac_{s}"] = c / n
    F = F.fillna({f"cnt_{EV_SHORT[e_]}": 0 for e_ in EVENTS})
    eps = 1.0
    F["r_srv_iv"] = F["cnt_srv"] / (F["cnt_iv"] + eps)
    F["r_ps_iv"] = F["cnt_ps"] / (F["cnt_iv"] + eps)
    F["r_fav_iv"] = F["cnt_fav"] / (F["cnt_iv"] + eps)
    F["r_sp_iv"] = F["cnt_sp"] / (F["cnt_iv"] + eps)
    F["cnt_contact"] = F["cnt_phone"] + F["cnt_chat"] + F["cnt_msg"]
    F["frac_contact"] = F["cnt_contact"] / n
    F["r_contact_iv"] = F["cnt_contact"] / (F["cnt_iv"] + eps)
    F["r_msg_chat"] = F["cnt_msg"] / (F["cnt_chat"] + eps)
    F["frac_engage"] = (F["cnt_ps"] + F["cnt_fav"] + F["cnt_contact"] + F["cnt_login"]) / n
    F["frac_browse"] = (F["cnt_iv"] + F["cnt_srv"] + F["cnt_sp"]) / n
    F["has_login"] = (F["cnt_login"] > 0).astype(int)
    F["has_contact"] = (F["cnt_contact"] > 0).astype(int)
    e = _entropy_from_counts(ev, "cookie_id", "event_name", "evn")
    F = F.join(e[["evn_ent", "evn_nent", "evn_top1"]])
    first_ev = g["event_name"].first()
    F["first_is_srv"] = (first_ev == "search_results_view").astype(int)
    F["first_is_iv"] = (first_ev == "item_view").astype(int)
    F["first_is_login"] = (first_ev == "login").astype(int)

    # ---------------- D. sequences / transitions
    tr = ev[~first_row]
    big = tr["prev_ev"].map(EV_SHORT) + ">" + tr["event_name"].map(EV_SHORT)
    bc = pd.crosstab(tr["cookie_id"], big)
    nt = bc.sum(axis=1)
    top_bigrams = big.value_counts().index[:40]
    for b in top_bigrams:
        F[f"bg_{b}"] = bc[b] / nt
    e = _entropy_from_counts(tr.assign(b=big), "cookie_id", "b", "bigram")
    F = F.join(e[["bigram_nuniq", "bigram_ent", "bigram_nent", "bigram_top1", "bigram_uniq_ratio"]])
    same = tr["prev_ev"] == tr["event_name"]
    F["same_ev_adj_frac"] = same.groupby(tr["cookie_id"]).mean()
    rl = _run_lengths(ev["cookie_id"], (ev["event_name"] == ev["prev_ev"]) & ~first_row)
    F["max_ev_run"] = rl.groupby(ev["cookie_id"]).max()
    F["max_ev_run_frac"] = F["max_ev_run"] / n
    # "funnel" semantics: item_view preceded by search / by item / photo after view
    F["iv_after_srv"] = ((tr["event_name"] == "item_view") & (tr["prev_ev"] == "search_results_view")).groupby(
        tr["cookie_id"]).sum() / (F["cnt_iv"] + eps)
    F["srv_after_srv"] = ((tr["event_name"] == "search_results_view") & (
        tr["prev_ev"] == "search_results_view")).groupby(tr["cookie_id"]).sum() / (F["cnt_srv"] + eps)
    # trigram entropy
    pp = ev["event_name"].shift(2)
    tri_ok = ev["cookie_id"].eq(ev["cookie_id"].shift(2))
    tri = (pp.map(EV_SHORT) + ">" + ev["prev_ev"].map(EV_SHORT) + ">" + ev["event_name"].map(EV_SHORT))[tri_ok]
    e = _entropy_from_counts(pd.DataFrame({"cookie_id": ev["cookie_id"][tri_ok], "t": tri}), "cookie_id", "t", "trigram")
    F = F.join(e[["trigram_nent", "trigram_top1", "trigram_uniq_ratio"]])

    # ---------------- E. items / categories / locations / seller
    it = ev.dropna(subset=["item_id"])
    gi = it.groupby("cookie_id")
    F["n_item_ev"] = gi.size()
    F["n_items"] = gi["item_id"].nunique()
    F["item_repeat_ratio"] = 1 - F["n_items"] / F["n_item_ev"]
    F["ev_per_item"] = F["n_item_ev"] / F["n_items"]
    F["max_ev_same_item"] = it.groupby(["cookie_id", "item_id"]).size().groupby(level=0).max()
    same_item = it["item_id"].eq(it["item_id"].shift(1)) & it["cookie_id"].eq(it["cookie_id"].shift(1))
    F["consec_same_item"] = same_item.groupby(it["cookie_id"]).mean()
    # revisits: item seen, then other items, then again
    it2 = it.assign(k=it.groupby(["cookie_id", "item_id"]).cumcount())
    F["item_revisit_nonconsec"] = ((it2["k"] > 0) & ~same_item).groupby(it2["cookie_id"]).sum()
    # item-level actions per viewed item (photo/fav/contact on items that were viewed first)
    iv_items = it[it.event_name == "item_view"][["cookie_id", "item_id"]].drop_duplicates().assign(v=1)
    act = it[it.event_name != "item_view"].merge(iv_items, on=["cookie_id", "item_id"], how="left")
    F["act_on_viewed_frac"] = act["v"].fillna(0).groupby(act["cookie_id"]).mean()
    # NOTE: item popularity / bot-pool features are time-aware and live in history.py
    F["item_id_mean"] = gi["item_id"].mean()
    F["item_id_std"] = gi["item_id"].std()
    # sequential item id crawling
    did = (it["item_id"] - it["item_id"].shift(1)).where(it["cookie_id"].eq(it["cookie_id"].shift(1)))
    F["item_id_absdiff_med"] = did.abs().groupby(it["cookie_id"]).median()
    F["item_id_small_step"] = ((did.abs() > 0) & (did.abs() <= 10)).groupby(it["cookie_id"]).mean()

    for col, p in [("item_category", "cat"), ("item_location", "loc"), ("seller_type", "sel"), ("item_id", "item")]:
        e = _entropy_from_counts(ev, "cookie_id", col, p)
        F = F.join(e[[f"{p}_nuniq", f"{p}_ent", f"{p}_nent", f"{p}_top1"]])
    # location switches between consecutive events
    lo = ev.dropna(subset=["item_location"])
    sw = lo["item_location"].ne(lo["item_location"].shift(1)) & lo["cookie_id"].eq(lo["cookie_id"].shift(1))
    F["loc_switch_frac"] = sw.groupby(lo["cookie_id"]).mean()
    ca = ev.dropna(subset=["item_category"])
    sw = ca["item_category"].ne(ca["item_category"].shift(1)) & ca["cookie_id"].eq(ca["cookie_id"].shift(1))
    F["cat_switch_frac"] = sw.groupby(ca["cookie_id"]).mean()
    F["sel_pro_frac"] = (it["seller_type"] == "pro").groupby(it["cookie_id"]).mean()
    F["sel_private_frac"] = (it["seller_type"] == "private").groupby(it["cookie_id"]).mean()
    big_city = ev["item_location"].isin(["moskva", "sankt-peterburg", "novosibirsk", "ekaterinburg"])
    F["loc_big4_frac"] = big_city.groupby(ev["cookie_id"]).mean()

    # ---------------- F. search
    s = ev[ev.event_name == "search_results_view"].copy()
    gs = s.groupby("cookie_id")
    F["n_queries"] = gs["search_query"].nunique()
    F["query_repeat_ratio"] = 1 - F["n_queries"] / F["cnt_srv"].replace(0, np.nan)
    F["page_mean"] = gs["search_page"].mean()
    F["page_median"] = gs["search_page"].median()
    F["page_max"] = gs["search_page"].max()
    F["page_std"] = gs["search_page"].std()
    F["page1_frac"] = (s["search_page"] == 1).groupby(s["cookie_id"]).mean()
    F["page_gt5_frac"] = (s["search_page"] > 5).groupby(s["cookie_id"]).mean()
    F["page_gt10_frac"] = (s["search_page"] > 10).groupby(s["cookie_id"]).mean()
    F["n_pages_uniq"] = gs["search_page"].nunique()
    same_q = s["search_query"].eq(s["search_query"].shift(1)) & s["cookie_id"].eq(s["cookie_id"].shift(1))
    dp = (s["search_page"] - s["search_page"].shift(1)).where(same_q)
    F["page_next_frac"] = (dp == 1).groupby(s["cookie_id"]).sum() / (F["cnt_srv"] - 1).clip(lower=1)
    F["page_jump_frac"] = (dp.abs() > 1).groupby(s["cookie_id"]).sum() / (F["cnt_srv"] - 1).clip(lower=1)
    F["page_same_frac"] = (dp == 0).groupby(s["cookie_id"]).sum() / (F["cnt_srv"] - 1).clip(lower=1)
    F["same_query_consec"] = same_q.groupby(s["cookie_id"]).sum() / (F["cnt_srv"] - 1).clip(lower=1)
    F["query_len_mean"] = s["search_query"].str.len().groupby(s["cookie_id"]).mean()
    F["query_tok_mean"] = s["search_query"].str.split().str.len().groupby(s["cookie_id"]).mean()
    F["query_has_digit"] = s["search_query"].str.contains(r"\d", na=False).groupby(s["cookie_id"]).mean()
    F["query_latin"] = s["search_query"].str.contains(r"[a-z]", na=False).groupby(s["cookie_id"]).mean()
    e = _entropy_from_counts(s, "cookie_id", "search_query", "q")
    F = F.join(e[["q_ent", "q_top1"]])
    # search category vs viewed-item categories (cookie-internal, no cross-cookie info):
    # does the cookie view items of the categories it searches in
    qcats = s[["cookie_id", "item_category"]].rename(columns={"item_category": "qc"}).dropna().drop_duplicates()
    icats = it[["cookie_id", "item_category"]].dropna().drop_duplicates().rename(columns={"item_category": "qc"}).assign(h=1)
    qm = qcats.merge(icats, on=["cookie_id", "qc"], how="left")
    F["qcat_in_items"] = qm["h"].fillna(0).groupby(qm["cookie_id"]).mean()
    F["n_query_cats"] = qcats.groupby("cookie_id").size()

    # ---------------- G. pointer
    web = ev["platform"].str.lower().isin(["web", "desktop"])
    ev["has_ptr"] = ev["pointer_x"].notna()
    pw = ev[web]
    F["is_web"] = web.groupby(ev["cookie_id"]).max().astype(int)
    F["ptr_frac_web"] = pw["has_ptr"].groupby(pw["cookie_id"]).mean()
    F["ptr_n"] = ev["has_ptr"].groupby(ev["cookie_id"]).sum()
    F["ptr_none_web"] = ((F["ptr_frac_web"] == 0) & (F["is_web"] == 1)).astype(int)
    p = ev[ev["has_ptr"]].copy()
    gp = p.groupby("cookie_id")
    F["ptr_x_mean"] = gp["pointer_x"].mean()
    F["ptr_y_mean"] = gp["pointer_y"].mean()
    F["ptr_x_std"] = gp["pointer_x"].std()
    F["ptr_y_std"] = gp["pointer_y"].std()
    F["ptr_x_max"] = gp["pointer_x"].max()
    F["ptr_y_max"] = gp["pointer_y"].max()
    F["ptr_zero_frac"] = ((p["pointer_x"] == 0) | (p["pointer_y"] == 0)).groupby(p["cookie_id"]).mean()
    F["ptr_edge_frac"] = ((p["pointer_y"] >= 1080) | (p["pointer_x"] >= 1920)).groupby(p["cookie_id"]).mean()
    F["ptr_round10_frac"] = ((p["pointer_x"] % 10 == 0) & (p["pointer_y"] % 10 == 0)).groupby(p["cookie_id"]).mean()
    # EDA: bot coordinates concentrate in a central box (x<~1250) plus exact 0 / 1080 / 1920 edges,
    # human coordinates are ~uniform over 1920x1080
    px, py = p["pointer_x"], p["pointer_y"]
    F["ptr_x_gt1300_frac"] = (px > 1300).groupby(p["cookie_id"]).mean()
    F["ptr_x_lt100_frac"] = ((px > 0) & (px < 100)).groupby(p["cookie_id"]).mean()
    F["ptr_y_extreme_frac"] = (((py > 0) & (py < 100)) | ((py > 950) & (py < 1080))).groupby(p["cookie_id"]).mean()
    F["ptr_box_frac"] = ((px >= 300) & (px <= 1200) & (py >= 300) & (py <= 900)).groupby(p["cookie_id"]).mean()
    F["ptr_exact_edge_frac"] = ((px == 0) | (py == 0) | (py == 1080) | (px == 1920)).groupby(p["cookie_id"]).mean()
    F["ptr_n_x_gt1300"] = (px > 1300).groupby(p["cookie_id"]).sum()
    p["xy"] = p["pointer_x"].astype(int).astype(str) + "_" + p["pointer_y"].astype(int).astype(str)
    e = _entropy_from_counts(p, "cookie_id", "xy", "ptrxy")
    F = F.join(e[["ptrxy_uniq_ratio", "ptrxy_top1"]])
    sp = p["cookie_id"].eq(p["cookie_id"].shift(1))
    dx = (p["pointer_x"] - p["pointer_x"].shift(1)).where(sp)
    dy = (p["pointer_y"] - p["pointer_y"].shift(1)).where(sp)
    dist = np.sqrt(dx ** 2 + dy ** 2)
    F["ptr_dist_mean"] = dist.groupby(p["cookie_id"]).mean()
    F["ptr_dist_std"] = dist.groupby(p["cookie_id"]).std()
    F["ptr_dist_med"] = dist.groupby(p["cookie_id"]).median()
    F["ptr_zero_move"] = (dist == 0).groupby(p["cookie_id"]).sum() / sp.groupby(p["cookie_id"]).sum().replace(0, np.nan)
    F["ptr_x_center_dev"] = (p["pointer_x"] - 960).abs().groupby(p["cookie_id"]).mean()
    F["ptr_y_center_dev"] = (p["pointer_y"] - 540).abs().groupby(p["cookie_id"]).mean()
    # pointer by event type (e.g. clicks on contact buttons at fixed positions)
    for e_ in ["item_view", "search_results_view"]:
        q = p[p.event_name == e_]
        F[f"ptr_y_std_{EV_SHORT[e_]}"] = q.groupby("cookie_id")["pointer_y"].std()

    # ---------------- H. user agent
    uaf = parse_ua(ev["user_agent"])
    uaf["cookie_id"] = ev["cookie_id"].values
    gu = uaf.groupby("cookie_id")
    for c in uaf.columns.drop("cookie_id"):
        F[c] = gu[c].mean() if c.startswith("ua_is") or c in ("ua_headless",) else gu[c].max()
    F["ua_nuniq"] = g["user_agent"].nunique()
    F["ua_switches"] = (ev["user_agent"].ne(ev["user_agent"].shift(1)) & ~first_row).groupby(ev["cookie_id"]).sum()
    F["ua_switch_rate"] = F["ua_switches"] / (n - 1).clip(lower=1)
    ua_freq = ev.drop_duplicates(["cookie_id", "user_agent"])["user_agent"].value_counts()
    F["ua_rarity"] = np.log(ev["user_agent"].map(ua_freq)).groupby(ev["cookie_id"]).min()
    pl = ev["platform"].str.lower().replace({"desktop": "web", "iphone": "ios"})
    plc = pl.groupby(ev["cookie_id"]).first()
    F["pl_android"] = (plc == "android").astype(int)
    F["pl_ios"] = (plc == "ios").astype(int)
    F["pl_web"] = (plc == "web").astype(int)
    # raw platform spelling diversity (technical noise; kept only for leakage/ablation checks)
    F["tech_platform_spellings"] = g["platform"].nunique()

    # ---------------- I. time of day
    ev["hour"] = (ev["tw"] // 3600).astype(int)
    hc = pd.crosstab(ev["cookie_id"], ev["hour"])
    hp = hc.div(hc.sum(axis=1), axis=0)
    F["hour_ent"] = -(hp * np.log(hp.where(hp > 0, 1))).sum(axis=1)
    F["hour_top1"] = hp.max(axis=1)
    F["frac_night_0_6"] = hp.loc[:, [h for h in range(0, 6) if h in hp]].sum(axis=1)
    F["frac_day_9_18"] = hp.loc[:, [h for h in range(9, 18) if h in hp]].sum(axis=1)
    F["frac_eve_18_24"] = hp.loc[:, [h for h in range(18, 24) if h in hp]].sum(axis=1)
    F["hour_mean"] = ev.groupby("cookie_id")["tw"].mean() / 3600
    hr = ev["tw"] / 3600 * 2 * np.pi / 24
    F["hour_sin"] = np.sin(hr).groupby(ev["cookie_id"]).mean()
    F["hour_cos"] = np.cos(hr).groupby(ev["cookie_id"]).mean()
    sec = ev["t"] % 60
    F["sec0_frac"] = (sec == 0).groupby(ev["cookie_id"]).mean()
    e = _entropy_from_counts(ev.assign(s=sec // 6), "cookie_id", "s", "secbin")
    F["secbin_nent"] = e["secbin_nent"]
    F["dow"] = m["window_start_ts"].dt.dayofweek.values  # for shift analysis only

    # ---------------- J. missingness
    ivl = ev[ev.event_name != "login"]
    F["miss_cat_frac"] = ivl["item_category"].isna().groupby(ivl["cookie_id"]).mean()
    F["miss_loc_frac"] = ivl["item_location"].isna().groupby(ivl["cookie_id"]).mean()
    F["miss_sel_frac"] = it["seller_type"].isna().groupby(it["cookie_id"]).mean()

    # ---------------- H14. last-N-minutes of cookie's own activity and of window
    tl = ev.groupby("cookie_id")["t"].transform("max")
    for mins in [5, 15, 30, 60]:
        x = ev["t"] >= tl - mins * 60
        F[f"n_last{mins}m_act"] = x.groupby(ev["cookie_id"]).sum()
    F["frac_last60m_act"] = F["n_last60m_act"] / n
    for mins in [60, 180]:
        x = ev["tw"] >= 86400 - mins * 60
        F[f"n_last{mins}m_win"] = x.groupby(ev["cookie_id"]).sum()
    tf = ev.groupby("cookie_id")["t"].transform("min")
    F["n_first15m_act"] = (ev["t"] < tf + 900).groupby(ev["cookie_id"]).sum()

    F = F.replace([np.inf, -np.inf], np.nan)
    F.columns = [c.replace(">", "_") for c in F.columns]
    return F


def family_map(cols):
    fam = {}
    for c in cols:
        if c.startswith(("dt_", "sdt_", "dtbin", "dtexact", "ddt_", "dwell_")):
            f = "B_intervals"
        elif c.startswith(("bg_", "bigram", "trigram", "same_ev_adj", "max_ev_run", "iv_after", "srv_after")):
            f = "D_sequence"
        elif c.startswith(("cnt_", "frac_", "r_", "has_", "evn_", "first_is")) and not c.startswith(
                ("frac_night", "frac_day", "frac_eve", "frac_active", "frac_last")):
            f = "C_events"
        elif c.startswith(("n_item", "n_items", "item", "max_ev_same", "consec_same", "act_on", "cat_", "loc_",
                           "sel_", "q_cat", "qcat")) and not c.startswith("q_"):
            f = "E_items"
        elif c.startswith(("n_queries", "query", "page", "n_pages", "same_query", "q_", "n_query")):
            f = "F_search"
        elif c.startswith(("ptr", "is_web")):
            f = "G_pointer"
        elif c.startswith(("ua_", "pl_", "tech_")):
            f = "H_ua_platform"
        elif c.startswith(("hour", "frac_night", "frac_day", "frac_eve", "sec", "dow")):
            f = "I_time"
        elif c.startswith("miss_"):
            f = "J_missing"
        elif c.startswith(("n_last", "frac_last", "n_first15")):
            f = "L_lastN"
        elif c.startswith(("cookie_age", "created", "log_cookie")):
            f = "A2_cookie_age"
        else:
            f = "A_activity"
        fam[c] = f
    return fam


if __name__ == "__main__":
    import time
    t0 = time.time()
    tr, te, meta = load_meta()
    ev = load_events(meta)
    F = build_features(ev, meta)
    out = Path(__file__).resolve().parent.parent / "work" / "features.parquet"
    F.to_parquet(out)
    print(F.shape, f"{time.time()-t0:.1f}s")

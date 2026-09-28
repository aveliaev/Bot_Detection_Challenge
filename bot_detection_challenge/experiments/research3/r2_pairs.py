"""C3.2 matched-pair analysis: FP_i -> nearest TP_j (and reverse) in B0 feature space; paired raw differences."""
import sys, json; sys.path.insert(0, 'research3')
from common import *
from cv import Data, POOL_COLS
from history import pool_features
from scipy.stats import wilcoxon
from sklearn.preprocessing import QuantileTransformer
pd.set_option('display.width', 250)
D = Data()
cols = [c for c in json.load(open(ROOT / 'cycle1' / 'feature_list.json')) if c not in POOL_COLS]
X = D.F.loc[D.ids, cols].join(pool_features(D.ci, D.labels, D.ids, 7))
o = pd.read_parquet(CACHE / 'groups_opt.parquet')
persist = CACHE / 'persistent.json'
A_ids = json.load(open(persist)) if persist.exists() else list(o[o.grp == 'A'].index)
print('A size', len(A_ids))
Z = pd.DataFrame(QuantileTransformer(n_quantiles=200, output_distribution='normal').fit_transform(X.fillna(X.median())), index=X.index, columns=X.columns)
tp_ids = o[o.grp.isin(['B', 'C'])].index
# raw profile
ev = events(); C = pd.read_parquet(CACHE / 'ptr_cookie.parquet')
g = ev.groupby('cookie_id')
d = ev[ev['dt'].notna() & (ev['dt'] <= 1800)]
R = pd.DataFrame({'n': g.size(), 'age_d': g.age_h.min() / 24, 'n_loc': g.item_location.nunique(), 'n_cat': g.item_category.nunique(),
                  'n_items': g.item_id.nunique(), 'page_max': g.search_page.max(),
                  'dt_med': d.groupby('cookie_id')['dt'].median(), 'f30_90': d.groupby('cookie_id')['dt'].apply(lambda x: ((x > 30) & (x <= 90)).mean()),
                  'f3_10': d.groupby('cookie_id')['dt'].apply(lambda x: ((x >= 3) & (x <= 10)).mean()),
                  'f_lt3': d.groupby('cookie_id')['dt'].apply(lambda x: (x < 3).mean()),
                  'hour_span': g.t.apply(lambda x: (x.max() - x.min()) / 3600),
                  'f_srv': g.event_name.apply(lambda x: (x == 'search_results_view').mean()),
                  'f_engage': g.event_name.apply(lambda x: x.isin(['favorite_add', 'photo_swipe', 'contact_phone_show', 'contact_chat_open', 'contact_message_sent', 'login']).mean()),
                  'uniq_q': g.search_query.nunique(), 'ua_ff': g.user_agent.first().str.contains('Firefox').astype(int)})
R = R.join(C[['sx', 'sy', 'edge_frac']]).join(X[['pool_nb_ge1_frac', 'pool_llr_mean']])
def pairs(src, dst, name):
    dist = ((Z.loc[src].values[:, None, :] - Z.loc[dst].values[None, :, :]) ** 2).sum(-1)
    j = dist.argmin(1)
    a, b = R.loc[src].reset_index(drop=True), R.loc[dst[j]].reset_index(drop=True)
    rows = []
    for c in R.columns:
        dd = (a[c] - b[c]).dropna()
        if len(dd) < 8 or (dd != 0).sum() < 5: continue
        p = wilcoxon(dd).pvalue
        rows.append((c, len(dd), a[c].median(), b[c].median(), (dd > 0).mean(), p))
    T = pd.DataFrame(rows, columns=['raw', 'n', f'med_{name[0]}', f'med_{name[1]}', f'P({name[0]}>{name[1]})', 'p_wilcoxon']).sort_values('p_wilcoxon')
    print(f'\n== pairs {name[0]} -> nearest {name[1]} (n={len(src)}), median NN dist {np.median(dist.min(1)):.1f}')
    print(T.round(3).to_string(index=False))
    return T
T1 = pairs(np.array(A_ids), np.array(tp_ids), ('FP', 'TP'))
# reverse: TP near threshold (rank 400-604) -> nearest negative among flagged/near-flagged negatives (rank<1200)
tpn = o[(o.target == 1) & o['rank'].between(400, 604)].index.values
negs = o[(o.target == 0) & (o['rank'] < 1200)].index.values
T2 = pairs(tpn, negs, ('TP', 'NEG'))
T1.to_csv(R3 / 'r2_pairs_fp_tp.csv', index=False); T2.to_csv(R3 / 'r2_pairs_tp_neg.csv', index=False)

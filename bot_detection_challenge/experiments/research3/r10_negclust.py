"""C3.10 negative subtype clustering: cluster labelled humans; where do persistent FP fall? stable across weeks?"""
import sys, json, warnings; sys.path.insert(0, 'research3'); warnings.filterwarnings('ignore')
from common import *
from rep import build, block_cols
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
pd.set_option('display.width', 250)
R = build(); m = meta(); pfp = json.load(open(CACHE / 'persistent.json'))
neg = m.index[(m.target == 0)]
pos = m.index[m.target == 1]
cols = block_cols(R, ['ev', 'tr', 'dt', 'px', 'py', 'pm', 'cat', 'pg', 'ua', 'pl', 'm'])
sc = StandardScaler().fit(R.loc[neg, cols]); pca = PCA(15, random_state=0).fit(sc.transform(R.loc[neg, cols]))
Z = lambda i: pca.transform(sc.transform(R.loc[i, cols]))
wk = (m.loc[neg, 'day'] >= 7).values
for k in [4, 6, 8, 12, 16]:
    km = KMeans(k, n_init=10, random_state=0).fit(Z(neg))
    lab = pd.Series(km.labels_, index=neg)
    lp = pd.Series(km.predict(Z(pos)), index=pos)
    t = pd.DataFrame({'n_neg': lab.value_counts(), 'n_pFP': lab[pfp].value_counts(), 'n_bots_assigned': lp.value_counts()}).fillna(0).astype(int)
    t['pFP_share_of_cluster'] = (t.n_pFP / t.n_neg).round(3)
    t['wk2_share'] = pd.Series(wk, index=neg).groupby(lab).mean().round(2)
    print(f'\nk={k}: max pFP concentration cluster'); print(t.sort_values('n_pFP', ascending=False).head(4))
# GMM density: are pFP outliers among humans?
g = GaussianMixture(8, covariance_type='diag', random_state=0).fit(Z(neg))
ll = pd.Series(g.score_samples(Z(neg)), index=neg)
print('\npFP log-density percentile among humans (in-sample):', ll.rank(pct=True)[pfp].describe()[['25%', '50%', '75%']].round(3).to_dict())
gb = GaussianMixture(8, covariance_type='diag', random_state=0).fit(Z(pos))
lb = pd.Series(gb.score_samples(Z(m.index[m.target.notna()])), index=m.index[m.target.notna()])
print('pFP log-density under BOT GMM, percentile among bots:', ((lb[pos].values[None, :] < lb[pfp].values[:, None]).mean(1)).round(2).tolist())

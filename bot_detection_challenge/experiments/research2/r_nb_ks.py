import sys; sys.path.insert(0,'research2')
from rcv import *
from nb import build_tokens, nb_features
from scipy.stats import ks_2samp
D=Data(); T=build_tokens(D.meta)
Ntr=nb_features(T,D.labels,D.ids).reindex(D.ids); Nte=nb_features(T,D.labels,D.te.cookie_id.values).reindex(D.te.cookie_id)
ks=pd.Series({c:ks_2samp(Ntr[c].dropna(),Nte[c].dropna()).statistic for c in Ntr.columns}).sort_values(ascending=False)
print(ks.round(3).head(12))
print(Ntr[['nb_mean_hour','nb_sum_hour']].describe().T.round(3)); print(Nte[['nb_mean_hour','nb_sum_hour']].describe().T.round(3))

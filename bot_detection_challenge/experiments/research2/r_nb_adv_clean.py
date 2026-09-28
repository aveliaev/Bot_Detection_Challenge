import sys; sys.path.insert(0,'research2')
from rcv import *
from nb import build_tokens, nb_features_clean
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp
D=Data(); T=build_tokens(D.meta)
Ntr=nb_features_clean(T,D.labels,D.ids).reindex(D.ids); Nte=nb_features_clean(T,D.labels,D.te.cookie_id.values).reindex(D.te.cookie_id)
print('max KS train vs test over NB cols:',max(ks_2samp(Ntr[c].dropna(),Nte[c].dropna()).statistic for c in Ntr.columns))
def adv(A,yy):
    o=np.zeros(len(yy))
    for a,b in StratifiedKFold(5,shuffle=True,random_state=0).split(A,yy):
        m=lgb.LGBMClassifier(n_estimators=300,learning_rate=.05,num_leaves=15,verbose=-1); m.fit(A.iloc[a],yy[a]); o[b]=m.predict_proba(A.iloc[b])[:,1]
    return roc_auc_score(yy,o)
yy=np.r_[np.zeros(len(Ntr)),np.ones(len(Nte))]
A=pd.concat([Ntr,Nte]); print('adv raw',round(adv(A,yy),4))
print('adv rounded 1dp',round(adv(A.round(1),yy),4))
print('adv sums only (nb_sum_*) rounded',round(adv(A[[c for c in A if c.startswith('nb_sum')]].round(1),yy),4))
# inside a CV fold: training rows (inner-OOF) vs validation rows -> same mechanism?
tri=np.flatnonzero(D.day<12); vai=np.flatnonzero(D.day>=12)
N2=nb_features_clean(T,D.labels.iloc[tri],D.ids).reindex(D.ids); y2=np.isin(np.arange(len(D.ids)),vai).astype(int)
print('adv within fold: train rows vs val rows raw',round(adv(N2,y2),4),' rounded',round(adv(N2.round(1),y2),4))

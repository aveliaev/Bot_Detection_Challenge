import sys; sys.path.insert(0,'research2')
from rcv import *
from misc2 import misc2_features
from sklearn.metrics import roc_auc_score
H=Harness(); M=misc2_features(H.D.meta)
y=H.D.y; X=M.reindex(H.D.ids)
for c in X.columns:
    a=roc_auc_score(y,X[c].fillna(-1)); print(f'{c:20s} AUC={max(a,1-a):.3f}')
# adversarial train vs test for this family
import lightgbm as lgb
from sklearn.model_selection import cross_val_predict
A=pd.concat([M.reindex(H.D.ids),M.reindex(H.D.te.cookie_id)]); yy=np.r_[np.zeros(len(H.D.ids)),np.ones(len(H.D.te))]
pa=cross_val_predict(lgb.LGBMClassifier(n_estimators=200,verbose=-1),A,yy,cv=5,method='predict_proba')[:,1]; print('adversarial AUC misc2',round(roc_auc_score(yy,pa),3))
H.add_family('misc2',lambda D,lab,ids: M)
H.run('D1_cat_misc2',fams=('pool','misc2'),hypothesis='item freshness lag + pointer trajectory + cross-cookie exact xy',leak_risk='low: past-only, no labels')

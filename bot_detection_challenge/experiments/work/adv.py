import sys; sys.path.insert(0,'cycle1')
from cv import *
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
D=Data()
Ptr=D.pool_full_train(); Pte=D.pool_test()
cols=[c for c in D.base_cols if c not in ('dow',)]+POOL_COLS
X=pd.concat([D.F.loc[D.tr.cookie_id].join(Ptr),D.F.loc[D.te.cookie_id].join(Pte)])[cols]
yy=np.r_[np.zeros(len(D.tr)),np.ones(len(D.te))]
def adv(cols):
    o=np.zeros(len(yy)); imp=np.zeros(len(cols))
    for tri,vai in StratifiedKFold(5,shuffle=True,random_state=0).split(X,yy):
        m=lgb.LGBMClassifier(n_estimators=300,learning_rate=0.05,num_leaves=15,colsample_bytree=0.5,verbose=-1,random_state=0,n_jobs=4)
        m.fit(X.iloc[tri][cols],yy[tri]); o[vai]=m.predict_proba(X.iloc[vai][cols])[:,1]; imp+=m.booster_.feature_importance('gain')
    return roc_auc_score(yy,o),pd.Series(imp,cols).sort_values(ascending=False),o
auc,imp,o=adv(cols); print('adversarial AUC all',round(auc,4)); print((imp/imp.sum()).head(10).round(3))
nop=[c for c in cols if not c.startswith(('pop_','pool_'))]
print('no pop/pool',round(adv(nop)[0],4)); print('pop only',round(adv([c for c in cols if c.startswith('pop_')])[0],4)); print('pool only',round(adv(POOL_COLS)[0],4))
# restrict to train days>=6 vs test
day=((D.tr.window_start_ts-DAY0).dt.days).values
keep=np.r_[day>=6,np.ones(len(D.te),bool)]
Xk=X[keep]; yk=yy[keep]; o=np.zeros(len(yk))
for tri,vai in StratifiedKFold(5,shuffle=True,random_state=0).split(Xk,yk):
    m=lgb.LGBMClassifier(n_estimators=300,learning_rate=0.05,num_leaves=15,colsample_bytree=0.5,verbose=-1,random_state=0,n_jobs=4)
    m.fit(Xk.iloc[tri],yk[tri]); o[vai]=m.predict_proba(Xk.iloc[vai])[:,1]
print('train days>=6 vs test, all feats',round(roc_auc_score(yk,o),4))

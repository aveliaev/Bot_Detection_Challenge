import sys; sys.path.insert(0,'research2')
from rcv import *
import xgboost as xgb
H=Harness(); D=H.D; tri,vai=H.schemes['P0'][2]
Xtr=H.X(BASE_COLS,('pool',),'P0',2,tri); Xva=H.X(BASE_COLS,('pool',),'P0',2,vai)
for kw in [dict(),dict(lambdarank_pair_method='mean',lambdarank_num_pair_per_sample=8),dict(lambdarank_pair_method='mean',lambdarank_num_pair_per_sample=32)]:
    m=xgb.XGBRanker(objective='rank:pairwise',n_estimators=600,learning_rate=0.03,max_depth=4,subsample=0.8,colsample_bytree=0.5,reg_lambda=5,random_state=0,n_jobs=2,**kw)
    m.fit(Xtr,D.y[tri],qid=np.zeros(len(tri),int)); p=m.predict(Xva); print(kw,round(precision_at_recall(D.y[vai],p),4))
m=make_model('xgb',None,0); m.set_params(n_jobs=2); m.fit(Xtr,D.y[tri]); print('xgb clf',round(precision_at_recall(D.y[vai],m.predict_proba(Xva)[:,1]),4))

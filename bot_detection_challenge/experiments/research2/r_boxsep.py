import sys; sys.path.insert(0,'research2')
from rcv import *
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
D=Data(); y=D.y; X=D.F.loc[D.ids].join(pool_features(D.ci,D.labels,D.ids,7))
box=((X.pl_web==1)&(X.ptr_n>=3)&(X.ptr_x_std<330)).values
Xb=X[box]; yb=y[box]
cols=[c for c in X.columns if c not in('dow','tech_platform_spellings')]
aucs=[]
for seed in range(5):
    o=np.zeros(len(yb))
    for a,b in StratifiedKFold(5,shuffle=True,random_state=seed).split(Xb,yb):
        m=make_model('cat',dict(iterations=500,learning_rate=0.03,depth=4),seed); m.fit(Xb.iloc[a][cols],yb[a]); o[b]=m.predict_proba(Xb.iloc[b][cols])[:,1]
    aucs.append(roc_auc_score(yb,o))
print('within box-group CV AUC (label separability):',np.round(aucs,3),'mean',round(np.mean(aucs),3),'n',len(yb),'bots',yb.sum())
# global OOF score within box group
o=repeated_oof('B0_cat_top120'); mk=(o>0)&box
print('global model AUC inside box group',round(roc_auc_score(y[mk],o[mk]),3))

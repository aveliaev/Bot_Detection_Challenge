import sys; sys.path.insert(0,'research2')
from rcv import *
from graph import graph_features
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp
D=Data(); y=D.y
o=repeated_oof('B0_cat_top120'); mk=o>0
thr=np.sort(o[mk])[::-1][600]
lab=D.labels[D.day<14]
X=D.F.loc[D.ids].join(pool_features(D.ci,D.labels,D.ids,7)).join(graph_features(D,D.labels,D.ids))
X['s']=o
grp=np.select([(y==1)&(o>=thr),(y==0)&(o>=thr),(y==1)&(o<thr)&mk],['TP','FP','FN'],'TN'); grp[~mk]='NA'
X['grp']=grp; print(pd.Series(grp).value_counts().to_dict())
A=X[X.grp.isin(['TP','FP'])]; t=(A.grp=='TP').values
rows=[]
for c in X.columns.drop(['grp','s']):
    v=A[c].values.astype(float)
    if np.isnan(v).all(): continue
    a=roc_auc_score(t,np.nan_to_num(v,nan=-999)); ks=ks_2samp(v[t][~np.isnan(v[t])],v[~t][~np.isnan(v[~t])]).statistic if (~np.isnan(v[~t])).sum()>3 else np.nan
    rows.append((c,max(a,1-a),ks,np.nanmedian(v[t]),np.nanmedian(v[~t]),np.nanmedian(X.loc[X.grp=='TN',c])))
R=pd.DataFrame(rows,columns=['f','auc_TPvsFP','ks','med_TP','med_FP','med_TN']).sort_values('auc_TPvsFP',ascending=False)
pd.set_option('display.width',200); print(R.head(25).round(3).to_string(index=False))
# shallow tree TP vs FP, CV AUC
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import HistGradientBoostingClassifier
cols=[c for c in X.columns if c not in('grp','s')]
print('TP vs FP classifier CV AUC',cross_val_score(HistGradientBoostingClassifier(max_depth=3,max_iter=200,learning_rate=.05),A[cols],t,cv=5,scoring='roc_auc').mean())
# score-region: in rank 200-800 ,bot vs human classifier AUC w/o score

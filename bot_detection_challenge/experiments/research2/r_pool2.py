import sys; sys.path.insert(0,'research2')
from rcv import *
from pool2 import pool2_features, cookie_attrs
from history import pool_features
from sklearn.metrics import roc_auc_score
H=Harness(); A=cookie_attrs(H.D.meta)
tri,vai=H.schemes['P0'][2]; lab=H.D.labels.iloc[tri]
G=pool2_features(H.D,lab,H.D.ids,attrs=A); y=H.D.labels.loc[G.index].values; va=np.isin(G.index,H.D.ids[vai])
for c in G.columns:
    a=roc_auc_score(y[va],G[c].fillna(-1).values[va]); print(f'{c:16s} AUC={max(a,1-a):.3f}')
# alpha sweep on existing pool features (single feature pool_rate_mean AUC on val)
for m in [0.5,2,5,20]:
    P=pool_features(H.D.ci,lab,H.D.ids,7,m=m); a=roc_auc_score(H.D.labels.loc[P.index][np.isin(P.index,H.D.ids[vai])],P.pool_rate_mean[np.isin(P.index,H.D.ids[vai])]); print('alpha',m,'pool_rate_mean AUC',round(a,4))
H.add_family('pool2',lambda D,lab,ids: pool2_features(D,lab,ids,attrs=A))
H.run('C2_cat_pool2',fams=('pool','pool2'),hypothesis='richer item bot-likelihood aggregates + attr-combo TE',leak_risk='medium: temporal OOF like pool')

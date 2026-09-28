import sys,time; sys.path.insert(0,'research2')
from rcv import *
from graph import graph_features
H=Harness()
t=time.time(); tri,vai=H.schemes['P0'][0]; G=graph_features(H.D,H.D.labels.iloc[tri],H.D.ids); print('one fold',round(time.time()-t,1),'s',G.shape)
from sklearn.metrics import roc_auc_score
from metrics import precision_at_recall
y=H.D.labels.loc[G.index].values; va=np.isin(G.index,H.D.ids[vai])
for c in G.columns:
    x=G[c].fillna(-1).values[va]; a=roc_auc_score(y[va],x); print(f'{c:18s} AUC={a:.3f} P@R={precision_at_recall(y[va],x if a>=.5 else -x):.3f}')
H.add_family('graph',graph_features)
H.run('B1_lgb_graph',model='lgb',fams=('pool','graph'),hypothesis='cookie-item graph neighbours/Jaccard/2hop',leak_risk='medium: OOF+temporal by construction')
H.run('B1_cat_graph',fams=('pool','graph'),hypothesis='cookie-item graph neighbours/Jaccard/2hop',leak_risk='medium: OOF+temporal by construction')

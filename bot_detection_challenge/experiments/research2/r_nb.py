import sys,time; sys.path.insert(0,'research2')
from rcv import *
from nb import build_tokens, nb_features
from sklearn.metrics import roc_auc_score
H=Harness(); t=time.time(); T=build_tokens(H.D.meta); print('tokens',T.shape,T.attr.value_counts().to_dict(),round(time.time()-t,1))
tri,vai=H.schemes['P0'][0]; t=time.time()
G=nb_features(T,H.D.labels.iloc[tri],H.D.ids); print('one fold',round(time.time()-t,1),G.shape)
y=H.D.labels.loc[G.index].values; va=np.isin(G.index,H.D.ids[vai])
for c in G.columns:
    x=G[c].fillna(0).values[va]; a=roc_auc_score(y[va],x); print(f'{c:22s} AUC={a:.3f} P@R={precision_at_recall(y[va],x if a>=.5 else -x):.3f}')
H.add_family('nb',lambda D,lab,ids: nb_features(T,lab,ids))
H.run('C1_lgb_nb',model='lgb',fams=('pool','nb'),hypothesis='event-level naive-Bayes evidence accumulation',leak_risk='medium: inner-OOF for train rows')
H.run('C1_cat_nb',fams=('pool','nb'),hypothesis='event-level naive-Bayes evidence accumulation',leak_risk='medium: inner-OOF for train rows')

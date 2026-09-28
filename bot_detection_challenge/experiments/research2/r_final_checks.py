import sys; sys.path.insert(0,'research2')
from rcv import *
from nb import build_tokens, nb_features
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
D=Data(); y=D.y; T=build_tokens(D.meta)
# adversarial: NB family (train rows inner-OOF with all train labels; test rows with all train labels)
Ntr=nb_features(T,D.labels,D.ids); Nte=nb_features(T,D.labels,D.te.cookie_id.values)
A=pd.concat([Ntr.reindex(D.ids),Nte.reindex(D.te.cookie_id)]); yy=np.r_[np.zeros(len(D.ids)),np.ones(len(D.te))]
o=np.zeros(len(yy))
for a,b in StratifiedKFold(5,shuffle=True,random_state=0).split(A,yy):
    m=lgb.LGBMClassifier(n_estimators=300,learning_rate=.05,num_leaves=15,verbose=-1); m.fit(A.iloc[a],yy[a]); o[b]=m.predict_proba(A.iloc[b])[:,1]
print('adversarial AUC NB family:',round(roc_auc_score(yy,o),4))
# segment breakdown + CI for final vs B0
def bl(ws,sch):
    arrs=[(w,np.load(OOF2/f'{n}__{sch}.npy')) for n,w in ws.items()]; s=sum(w*pd.Series(a).rank(pct=True).values for w,a in arrs)/sum(ws.values()); s[arrs[0][1]==0]=0; return s
FIN={'B0_cat_top120':.5,'C1_cat_nb':.5}
fin=np.mean([bl(FIN,f'P{i}') for i in range(5)],0); b0=repeated_oof('B0_cat_top120')
np.save(OOF2/'FINAL_repeated_oof.npy',fin)
X=D.F.loc[D.ids]
for nm,o in [('B0',b0),('FINAL',fin)]:
    mk=o>0; lo,hi=bootstrap_ci(y,o,n=1000)
    seg={s:round(precision_at_recall(y[mk&(X[s].values==1)],o[mk&(X[s].values==1)]),3) for s in ['pl_web','pl_android','pl_ios']}
    print(nm,'P@R',round(precision_at_recall(y[mk],o[mk]),4),'CI95',(round(lo,3),round(hi,3)),seg,{k:round(v,4) for k,v in report(y[mk],o[mk]).items()})
# per-fold P@R of final (P0 partition + fwd7)
for sch in ['P0','fwd7']:
    f=bl(FIN,sch); schemes=__import__('rcv').partitions(D.day)[0] if sch=='P0' else __import__('rcv').fwd7(D.day)
    print(sch,[round(precision_at_recall(y[v],f[v]),3) for _,v in schemes])

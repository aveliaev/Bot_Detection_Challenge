import sys; sys.path.insert(0,'cycle1')
from cv import *
from metrics import operating_point
D=Data(); y=D.y
W={'S2_cat_top120_s234':.6,'S2_et_top120':.2,'S2_xgb_top120':.2}
X=D.F.loc[D.ids]
for s in ['dayblock','fwd7']:
    O={n:np.load(OOF_DIR/f'{n}__{s}.npy') for n in W}; m=np.all([o>0 for o in O.values()],0)
    b=sum(w*pd.Series(O[n][m]).rank(pct=True).values for n,w in W.items())
    r=report(y[m],b); op=operating_point(y[m],b)
    print(s,{k:round(v,4) for k,v in r.items()},op)
    # fold stability
    for tri,vai in D.folds(s):
        k=np.zeros(len(y),bool); k[vai]=True; k=k[m]; print('   fold P@R',round(precision_at_recall(y[m][k],b[k]),3),'bots',int(y[m][k].sum()))
    if s=='dayblock':
        pass
        Z=X[m].copy(); Z['s']=b; Z['y']=y[m]; thr=op['threshold']
        Z['grp']=np.select([(Z.y==1)&(Z.s>=thr),(Z.y==1)&(Z.s<thr),(Z.y==0)&(Z.s>=thr)],['TP','FN','FP'],'TN')
        print(Z.grp.value_counts().to_dict())
        for seg in ['pl_web','pl_android','pl_ios']:
            k=Z[seg]==1; print(f'  {seg}: n={k.sum()} bots={int(Z.y[k].sum())} recall@thr={((Z.s>=thr)&(Z.y==1))[k].sum()/Z.y[k].sum():.3f} FP={int(((Z.s>=thr)&(Z.y==0))[k].sum())} P@R_in_seg={precision_at_recall(Z.y[k],Z.s[k]):.3f}')
        cols=['n_events','ptr_none_web','ptr_x_std','ptr_x_max','sdt_median','dt_lt30','loc_nuniq','page_mean','cookie_age_d','pool_nb_ge1_frac'] 
        P=D.pool('dayblock',0,*D.folds('dayblock')[0])
        Z=Z.join(P[['pool_nb_ge1_frac']])
        print(Z.groupby('grp')[cols].median().T.round(3))
        # hardest FP around threshold
        fp=Z[(Z.grp=='FP')].sort_values('s',ascending=False)
        print('FP: web',round(fp.pl_web.mean(),2),'ptr_none_web',round(fp.ptr_none_web.mean(),2),'young(<7d)',round((fp.cookie_age_d<7).mean(),2), 'vs TN young',round((Z[Z.grp=="TN"].cookie_age_d<7).mean(),2))

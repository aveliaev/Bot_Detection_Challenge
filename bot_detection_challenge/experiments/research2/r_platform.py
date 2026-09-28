import sys; sys.path.insert(0,'research2')
from rcv import *
H=Harness(); D=H.D
web=D.F['pl_web']
def seg_fit(Xtr,ytr,Xva,seed,tri,vai,sch,k):
    out=np.zeros(len(Xva)); wtr=web.loc[Xtr.index].values==1; wva=web.loc[Xva.index].values==1
    for m_tr,m_va in [(wtr,wva),(~wtr,~wva)]:
        cols=[c for c in Xtr.columns if not (c.startswith('ptr') and not m_tr[0] and False)]
        mdl=make_model('cat',None,seed); mdl.fit(Xtr[m_tr],ytr[m_tr]); out[m_va]=mdl.predict_proba(Xva[m_va])[:,1]
    return out
def mob_nopointer_fit(Xtr,ytr,Xva,seed,tri,vai,sch,k):
    # global model for web + mobile model without pointer features
    out=np.zeros(len(Xva)); wtr=web.loc[Xtr.index].values==1; wva=web.loc[Xva.index].values==1
    g=make_model('cat',None,seed); g.fit(Xtr,ytr); out[:]=g.predict_proba(Xva)[:,1]
    cols=[c for c in Xtr.columns if not c.startswith(('ptr','pt_','is_web','pl_'))]
    mm=make_model('cat',None,seed); mm.fit(Xtr.loc[~wtr,cols],ytr[~wtr]); pm=mm.predict_proba(Xva.loc[~wva,cols])[:,1]
    out[~wva]=0.5*out[~wva]+0.5*pm
    return out
H.run('G1_cat_segment_models',fit_fn=seg_fit,hypothesis='separate web / mobile models',leak_risk='low')
H.run('G2_cat_global_plus_mobile',fit_fn=mob_nopointer_fit,hypothesis='global + mobile-only (no pointer) average on mobile',leak_risk='low')
# segment-level P@R of baseline vs G1
y=D.y
for nm in ['B0_cat_top120','G1_cat_segment_models','G2_cat_global_plus_mobile']:
    o=repeated_oof(nm); mk=o>0
    r=[precision_at_recall(y[mk&(web.loc[D.ids].values==v)],o[mk&(web.loc[D.ids].values==v)]) for v in (1,0)]
    print(nm,'overall',round(precision_at_recall(y[mk],o[mk]),4),'web',round(r[0],3),'mobile',round(r[1],3))

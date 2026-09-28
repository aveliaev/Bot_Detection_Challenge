import sys; sys.path.insert(0,'cycle1')
from cv import *
from text_models import build_docs, TextLR
D=Data()
dp=WORK/'docs.parquet'
if not dp.exists():
    ev=load_events(D.meta); build_docs(ev).to_parquet(dp)
docs=pd.read_parquet(dp)
for kind,C in [('ua',0.5),('seq',0.1),('seq',0.5)]:
    for sch in ['dayblock','fwd7']:
        o=np.full(len(D.y),np.nan); fs=[]
        for tri,vai in D.folds(sch):
            m=TextLR(kind,C).fit(docs.loc[D.ids[tri],kind],D.y[tri]); o[vai]=m.predict_proba(docs.loc[D.ids[vai],kind])[:,1]
            fs.append(precision_at_recall(D.y[vai],o[vai]))
        mk=~np.isnan(o); r=report(D.y[mk],o[mk]); r.update(fold_mean=np.mean(fs),fold_std=np.std(fs),fold_min=np.min(fs),sec=0,exp=f'X_{kind}_C{C}',scheme=sch)
        o2=np.zeros(len(o)); o2[mk]=pd.Series(o[mk]).rank(pct=True).values; np.save(OOF_DIR/f"X_{kind}_C{C}__{sch}.npy",o2)
        log_experiment(r); print(r['exp'],sch,round(r['p_at_r70'],4),round(r['pr_auc'],4),round(r['roc_auc'],4),flush=True)

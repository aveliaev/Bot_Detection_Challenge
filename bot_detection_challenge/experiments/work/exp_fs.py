import sys; sys.path.insert(0,'cycle1')
from cv import *
D=Data(); allc=json.load(open(WORK/'cols_all.json')); fam=json.load(open(WORK/'family.json'))
# importance: average CatBoost importance over dayblock fold models (seed 0)
imp=pd.Series(0.0,index=allc)
for k,(tri,vai) in enumerate(D.folds('dayblock')):
    P=D.pool('dayblock',k,tri,vai); m=make_model('cat',None,0); m.fit(get_X(D,allc,tri,P),D.y[tri])
    imp+=pd.Series(m.get_feature_importance(),index=allc)
imp=imp.sort_values(ascending=False); imp.to_csv(WORK/'cat_importance.csv')
print(imp.head(30).round(2).to_string())
print(imp.groupby(imp.index.map(fam)).sum().sort_values(ascending=False).round(1))
for K in [60,120,200]:
    cols=list(imp.index[:K])
    for sch in ['dayblock','fwd7']:
        r,_=run_cv(D,cols,model='cat',name=f'S_cat_top{K}',scheme=sch,seeds=(0,1)); r.update(exp=f'S_cat_top{K}',scheme=sch); log_experiment(r)
drop={'L_lastN','H_ua_platform','D_sequence'}
cols=[c for c in allc if fam[c] not in drop or c in ('pl_web','pl_android','pl_ios')]
for sch in ['dayblock','fwd7']:
    r,_=run_cv(D,cols,model='cat',name='S_cat_drop_L_H_D',scheme=sch,seeds=(0,1)); r.update(exp='S_cat_drop_L_H_D',scheme=sch); log_experiment(r)

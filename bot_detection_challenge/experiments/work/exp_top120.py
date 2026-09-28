import sys; sys.path.insert(0,'cycle1')
from cv import *
D=Data(); imp=pd.read_csv(WORK/'cat_importance.csv',index_col=0).iloc[:,0]
allc=json.load(open(WORK/'cols_all.json'))
cols=list(imp.index[:120]); json.dump(cols,open(ROOT/'cycle1'/'feature_list.json','w'),indent=0)
for model,nm,p,seeds in [('cat','S2_cat_top120_s234',None,(2,3,4)),('cat','S2_cat_all_s234',None,(2,3,4)),('et','S2_et_top120',dict(min_samples_leaf=3),(0,1)),('xgb','S2_xgb_top120',None,(0,1,2))]:
    c=allc if 'all' in nm else cols
    for sch in ['dayblock','fwd7']:
        r,_=run_cv(D,c,model=model,params=p,name=nm,scheme=sch,seeds=seeds); r.update(exp=nm,scheme=sch); log_experiment(r)

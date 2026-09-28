import sys; sys.path.insert(0,'cycle1')
from cv import *
D=Data(); fam=json.load(open(WORK/'family.json'))
allc=[c for c in fam if c not in {'dow','tech_platform_spellings'}]
json.dump(allc,open(WORK/'cols_all.json','w'))
for model,seeds in [('lr',(0,)),('et',(0,1,2)),('hgb',(0,1,2)),('xgb',(0,1,2)),('cat',(0,1,2)),('lgb',(0,1,2))]:
    for sch in ['dayblock','fwd7']:
        r,_=run_cv(D,allc,model=model,name=f'M_{model}',scheme=sch,seeds=seeds); r.update(exp=f'M_{model}',scheme=sch); log_experiment(r)

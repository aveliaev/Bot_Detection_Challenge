import sys; sys.path.insert(0,'cycle1')
from cv import *
D=Data(rebuild=True)
fam=family_map(D.base_cols)
for c in D.base_cols:
    if c.startswith('pop_'): fam[c]='P_popularity'
for c in POOL_COLS: fam[c]='Q_pool'
old=json.load(open(WORK/'family.json'))
json.dump(fam,open(WORK/'family.json','w'))
EXCL={'dow','tech_platform_spellings'}
allc=[c for c in fam if c not in EXCL]; oldc=[c for c in old if c not in EXCL]
new=[c for c in allc if c not in old]; print('new',new)
for nm,cols in [('G_old',oldc),('G_new',allc),('G_new_noband',[c for c in allc if not c.startswith(('sdt_band','sdt_n_3','dwell'))]),('G_new_noptrzone',[c for c in allc if c not in new or not c.startswith('ptr')])]:
    for sch in ['dayblock','fwd7']:
        r,_=run_cv(D,cols,name=nm,scheme=sch,seeds=(0,1,2)); r.update(exp=nm,scheme=sch); log_experiment(r)

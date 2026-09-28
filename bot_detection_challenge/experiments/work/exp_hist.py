import sys; sys.path.insert(0,'cycle1')
from cv import *
D=Data(); fam=json.load(open(WORK/'family.json'))
fam={c:f for c,f in fam.items() if c!='pool_nbn_mean' and c!='pool_lift_mean'}
allc=[c for c in fam if c not in ('dow','tech_platform_spellings')]
for hd in [7,None,4,10]:
    D.hist_days=hd
    for sch in ['dayblock','forward']:
        r,_=run_cv(D,allc,name=f'H_hist{hd}',scheme=sch,seeds=(0,1,2)); r.update(exp=f'H_hist{hd}',scheme=sch); log_experiment(r)

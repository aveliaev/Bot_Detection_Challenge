import sys; sys.path.insert(0,'cycle1')
from cv import *
D=Data(); fam=json.load(open(WORK/'family.json'))
EXCL={'dow','tech_platform_spellings'}
fams=sorted(set(fam.values()))
allc=[c for c in fam if c not in EXCL]
r,_=run_cv(D,allc,name='AB_all',seeds=(0,1,2)); r.update(exp='AB_all'); log_experiment(r)
for f in fams:
    cols=[c for c in allc if fam[c]!=f]
    r,_=run_cv(D,cols,name=f'AB_minus_{f}',seeds=(0,1,2)); r.update(exp=f'AB_minus_{f}'); log_experiment(r)

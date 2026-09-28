import sys; sys.path.insert(0,'cycle1')
from cv import *
D=Data(); fam=json.load(open(WORK/'family.json'))
fam={c:f for c,f in fam.items() if c not in('pool_nbn_mean','pool_lift_mean')}
json.dump(fam,open(WORK/'family.json','w'))
allc=[c for c in fam if c not in ('dow','tech_platform_spellings')]
nopool=[c for c in allc if fam[c]!='Q_pool']
nohist=[c for c in allc if fam[c] not in('Q_pool','P_popularity')]
for nm,cols in [('F_all',allc),('F_nopool',nopool),('F_nohist',nohist)]:
    for sch in ['fwd7','dayblock']:
        r,o=run_cv(D,cols,name=nm,scheme=sch,seeds=(0,1,2)); r.update(exp=nm,scheme=sch); log_experiment(r)
        if sch=='fwd7':
            # P@R by gap (days after last label), fold k=7 only
            vai=np.flatnonzero((D.day>=7)); 
            for g in [(7,8,9),(10,11,12,13)]:
                mk=np.isin(D.day,g); print('   gap days',g,round(precision_at_recall(D.y[mk],o[mk]),4),int(D.y[mk].sum()))

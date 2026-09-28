import sys; sys.path.insert(0,'cycle1')
from cv import *
D=Data(rebuild=True)
fam=family_map(D.base_cols)
for c in D.base_cols:
    if c.startswith('pop_'): fam[c]='P_popularity'
for c in POOL_COLS: fam[c]='Q_pool'
json.dump(fam,open(WORK/'family.json','w'))
fams=sorted(set(fam.values())); print({f:sum(v==f for v in fam.values()) for f in fams})
EXCL={'dow','tech_platform_spellings'}
def cols_of(*fs): return [c for c,f in fam.items() if f in fs and c not in EXCL]
res=[]
def go(name,cols,**kw):
    for sch in kw.pop('schemes',['dayblock']):
        r,_=run_cv(D,cols,name=name,scheme=sch,**kw); r.update(exp=name,scheme=sch); log_experiment(r); res.append(r)
go('E0_quickstart',['n_events','n_items'],schemes=['dayblock','forward','strat'])
go('E1_activity',cols_of('A_activity'),schemes=['dayblock','forward','strat'])
base=['A_activity']
for f in ['B_intervals','C_events','D_sequence','E_items','F_search','G_pointer','H_ua_platform','I_time','J_missing','L_lastN','A2_cookie_age','P_popularity','Q_pool']:
    go(f'E2_A+{f}',cols_of(*base,f))
allf=[f for f in fams]
go('E3_all',cols_of(*allf),schemes=['dayblock','forward','strat'])
go('E3_all_nopool',cols_of(*[f for f in allf if f!='Q_pool']),schemes=['dayblock','forward','strat'])

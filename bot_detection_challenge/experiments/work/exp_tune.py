import sys; sys.path.insert(0,'cycle1')
from cv import *
D=Data(); allc=json.load(open(WORK/'cols_all.json'))
cfgs=[
 ('cat','T_cat_d4',dict(depth=4),1),
 ('cat','T_cat_d8',dict(depth=8,iterations=1200),1),
 ('cat','T_cat_l2_15',dict(l2_leaf_reg=15),1),
 ('cat','T_cat_lr015_it3000',dict(learning_rate=0.015,iterations=3000),1),
 ('cat','T_cat_pw3',{},3),
 ('cat','T_cat_rsm02',dict(rsm=0.2),1),
 ('et','T_et_leaf3',dict(min_samples_leaf=3),1),
 ('et','T_et_leaf10',dict(min_samples_leaf=10),1),
 ('et','T_et_mf05',dict(max_features=0.5),1),
 ('lgb','T_lgb_slow',dict(learning_rate=0.01,n_estimators=1500,num_leaves=7,min_child_samples=30,colsample_bytree=0.3),1),
 ('lgb','T_lgb_pw3',{},3),
]
for model,nm,p,pw in cfgs:
    for sch in ['dayblock','fwd7']:
        r,_=run_cv(D,allc,model=model,params=p,name=nm,scheme=sch,seeds=(0,1),pos_weight=pw); r.update(exp=nm,scheme=sch,params=p,pw=pw); log_experiment(r)

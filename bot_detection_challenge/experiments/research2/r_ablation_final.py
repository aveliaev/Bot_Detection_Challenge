import sys; sys.path.insert(0,'research2')
from rcv import *
H=Harness(); fam=json.load(open(WORK/'family.json'))
nopool=[c for c in BASE_COLS if c not in POOL_COLS]
H.run('AB2_minus_pool',cols=nopool,fams=(),hypothesis='ablation: no labelled item pool',leak_risk='low')
H.run('AB2_minus_pointer',cols=[c for c in BASE_COLS if fam.get(c)!='G_pointer'],hypothesis='ablation: no pointer family',leak_risk='low')
H.run('AB2_minus_intervals',cols=[c for c in BASE_COLS if fam.get(c)!='B_intervals'],hypothesis='ablation: no inter-event family',leak_risk='low')
H.run('AB2_minus_pop',cols=[c for c in BASE_COLS if fam.get(c)!='P_popularity'],hypothesis='ablation: no unlabelled popularity',leak_risk='low')

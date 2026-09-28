import sys,warnings; warnings.filterwarnings('ignore'); sys.path.insert(0,'cycle1')
import pandas as pd,numpy as np
from features import load_meta,load_events
from sklearn.metrics import roc_auc_score
tr,te,meta=load_meta(); ev=load_events(meta)
ev=ev.merge(meta[['cookie_id','target','part']],on='cookie_id')
it=ev.dropna(subset=['item_id'])
ci=it.drop_duplicates(['cookie_id','item_id'])
per=ci.groupby('item_id').agg(nck=('cookie_id','size'),nbot=('target',lambda s:(s==1).sum()),nhum=('target',lambda s:(s==0).sum()),nte=('part',lambda s:(s=='te').sum()))
print(per.describe().T)
print('items viewed by >=1 bot',(per.nbot>0).sum(),'of',len(per))
print(per.nck.value_counts().sort_index().head(15))
# popularity among humans only (excluding self) predicts?
tt=ci[ci.part=='tr'].merge(per,on='item_id')
tt['hum_other']=tt.nhum-(tt.target==0); tt['bot_other']=tt.nbot-(tt.target==1); tt['te_n']=tt.nte
g=tt.groupby('cookie_id').agg(t=('target','first'),hum=('hum_other','mean'),bot=('bot_other','mean'),ten=('te_n','mean'))
for c in ['hum','bot','ten']: print(c,roc_auc_score(g.t,g[c]))
# item id range by target
print(ci[ci.part=='tr'].groupby('target').item_id.describe())
# is there a bot item "pool"?  items popular: rank
top=per.sort_values('nck',ascending=False).head(10); print(top)
# per-day cookie count
print(ev.groupby(ev.window_start_ts.dt.date).cookie_id.nunique())

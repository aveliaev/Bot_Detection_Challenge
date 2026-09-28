import sys,warnings; warnings.filterwarnings('ignore'); sys.path.insert(0,'cycle1')
import pandas as pd,numpy as np
from features import load_meta,load_events
from sklearn.metrics import roc_auc_score
tr,te,meta=load_meta(); ev=load_events(meta)
ev=ev.merge(meta[['cookie_id','target','part']],on='cookie_id')
ev['day']=(ev.window_start_ts-pd.Timestamp('2026-04-06')).dt.days
ci=ev.dropna(subset=['item_id']).drop_duplicates(['cookie_id','item_id'])[['cookie_id','item_id','day','target','part']]
# pairs of cookies sharing items
p=ci.merge(ci,on='item_id',suffixes=('','_o')); p=p[p.cookie_id!=p.cookie_id_o]
pb=p[(p.target==1)&(p.target_o==1)]
print('bot-bot share pairs day diff'); print((pb.day-pb.day_o).abs().value_counts().sort_index().head(15))
ph=p[(p.target==0)&(p.target_o==0)]; print('hum-hum day diff'); print((ph.day-ph.day_o).abs().value_counts(normalize=True).sort_index().head(8))
# stability: bot items in days 0-6 vs bots in days 7-13 : fraction of late-bot items seen by early bots, vs humans
early=set(ci[(ci.target==1)&(ci.day<7)].item_id)
late=ci[(ci.day>=7)&(ci.part=='tr')]
print(late.assign(h=late.item_id.isin(early)).groupby('target').h.mean())
# how concentrated is the pool: number of distinct items seen by bots vs by humans per view
b=ci[ci.target==1]; h=ci[ci.target==0]
print('bots: views',len(b),'distinct',b.item_id.nunique(),' humans views',len(h),'distinct',h.item_id.nunique())
# bot item pool: items where bot fraction high; item_id ranges?
per=ci[ci.part=='tr'].groupby('item_id').agg(nb=('target','sum'),n=('target','size'))
print(per[per.nb>=3].index.to_series().describe())
# connected components of bots via shared items
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
bc=b.cookie_id.astype('category'); bi=b.item_id.astype('category')
M=sp.csr_matrix((np.ones(len(b)),(bc.cat.codes,bi.cat.codes)))
A=(M@M.T)>0
nc,lab=connected_components(A); print('bot components',nc, pd.Series(lab).value_counts().head(5).tolist())
# share of bot's items that are in pool of OTHER bots

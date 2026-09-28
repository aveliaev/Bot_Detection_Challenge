import sys,re; sys.path.insert(0,'work')
from common_load import *
pd.set_option('display.width',250); pd.set_option('display.max_columns',50); pd.set_option('display.max_rows',300)
tr,te,meta,m=load()
m=m.sort_values(['cookie_id','event_ts','row'])
cols=['event_ts','event_name','platform','item_id','item_category','item_location','seller_type','search_query','search_page','pointer_x','pointer_y']
rng=np.random.RandomState(0)
for tg in [1,0]:
    ids=meta[(meta.target==tg)].cookie_id.values; 
    for cid in rng.choice(ids,3):
        d=m[m.cookie_id==cid]; print('=== target',tg,cid,len(d),'created',meta.set_index('cookie_id').cookie_created_at[cid]); print(d[cols].head(25).to_string(index=False))

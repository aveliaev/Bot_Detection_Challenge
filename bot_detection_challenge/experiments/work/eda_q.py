import sys,warnings; warnings.filterwarnings('ignore'); sys.path.insert(0,'cycle1')
import pandas as pd,numpy as np
from features import load_meta,load_events
tr,te,meta=load_meta(); ev=load_events(meta)
ev=ev.merge(meta[['cookie_id','target','part']],on='cookie_id'); t=ev[ev.part=='tr']
s=t[t.event_name=='search_results_view']
q=s.drop_duplicates(['cookie_id','search_query']).groupby('search_query').target.agg(['mean','size']).sort_values('mean')
print(q.head(5),q.tail(5),'spread std',q['mean'].std(), 'binomial std approx',np.sqrt(.08*.92/q['size'].mean()))
l=t.drop_duplicates(['cookie_id','item_location']).groupby('item_location').target.agg(['mean','size']).sort_values('mean'); print(l.head(3),l.tail(3))
c=t.drop_duplicates(['cookie_id','item_category']).groupby('item_category').target.agg(['mean','size']).sort_values('mean'); print(c)
# page distribution
print(pd.crosstab(s.search_page.clip(upper=15),s.target,normalize='columns').round(3).T)
# item attributes consistent per item?
it=ev.dropna(subset=['item_id'])
print('cat per item nunique',it.groupby('item_id').item_category.nunique().value_counts().head(), it.groupby('item_id').item_location.nunique().value_counts().head())

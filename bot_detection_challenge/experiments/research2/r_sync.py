import sys; sys.path.insert(0,'research2'); sys.path.insert(0,'cycle1')
from rcv import *
from features import load_events
D=Data(); meta=D.meta.copy(); ev=load_events(meta)
ev=ev.merge(meta[['cookie_id','target','part']],on='cookie_id')
ev['tw']=(ev.event_ts-ev.window_start_ts).dt.total_seconds()
t=ev[ev.part=='tr']
print('first-minute events by target:'); print(t.assign(fm=t.tw<120).groupby('target').fm.mean())
f=t.groupby('cookie_id').agg(first=('tw','min'),y=('target','first'))
print(f.groupby(pd.cut(f['first'],[-1,60,300,3600,6*3600,86400])).y.agg(['mean','size']))
# same-second collisions across cookies
c=ev.groupby('event_ts').cookie_id.nunique(); ev['ts_ncookies']=ev.event_ts.map(c)
t=ev[ev.part=='tr']; print(t.groupby('target').ts_ncookies.describe())
# cookie creation bursts
m=meta.copy(); m['cs']=m.cookie_created_at.astype('int64')//10**6
m=m.sort_values('cs'); cs=m.cs.values
for w in [10,60,600]:
    m[f'n_created_{w}s']=np.searchsorted(cs,cs+w,side='right')-np.searchsorted(cs,cs-w,side='left')-1
mt=m[m.part=='tr']
for w in [10,60,600]: print(w, mt.groupby(mt[f'n_created_{w}s'].clip(upper=5)).target.agg(['mean','size']).T.round(3).to_string())
# created seconds pattern
print(mt.groupby(mt.cookie_created_at.dt.second==0).target.agg(['mean','size']))
print(mt.groupby((m.cookie_created_at-m.window_start_ts).dt.total_seconds().loc[mt.index].pipe(lambda s: pd.cut(s/86400,[-400,-30,-7,-1,0,1]))).target.agg(['mean','size']))

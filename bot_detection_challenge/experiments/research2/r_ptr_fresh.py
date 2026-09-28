import sys; sys.path.insert(0,'research2'); sys.path.insert(0,'cycle1')
from rcv import *
from features import load_events
D=Data(); meta=D.meta; ev=load_events(meta)
ev=ev.merge(meta[['cookie_id','target','part']],on='cookie_id')
p=ev[ev.pointer_x.notna()].copy(); p['xy']=p.pointer_x.astype(int).astype(str)+'_'+p.pointer_y.astype(int).astype(str)
# how many distinct cookies share exact xy
n=p.groupby('xy').cookie_id.nunique(); p['xy_nck']=p.xy.map(n)-1
t=p[p.part=='tr']; print('exact xy shared with other cookies (mean #):'); print(t.groupby('target').xy_nck.describe())
# expected under uniformity: humans cover 1920x1080 -> collisions rare; bots box
# bot-bot exact xy sharing vs bot-human
nb=t[t.target==1].groupby('xy').cookie_id.nunique(); t['n_bot_other']=t.xy.map(nb).fillna(0)-(t.target==1)
print(t.groupby('target').n_bot_other.mean())
# trajectory: consecutive identical dx,dy patterns? angle distribution
p=p.sort_values(['cookie_id','event_ts']); s=p.cookie_id.eq(p.cookie_id.shift())
p['dx']=np.where(s,p.pointer_x.diff(),np.nan); p['dy']=np.where(s,p.pointer_y.diff(),np.nan)
t=p[(p.part=='tr')&p.dx.notna()]
print('dx==0 or dy==0 frac',t.groupby('target').apply(lambda d:((d.dx==0)|(d.dy==0)).mean()))
print('x mod 10 == 0',p[p.part=='tr'].groupby('target').apply(lambda d:(d.pointer_x%10==0).mean()), 'x mod 2',p[p.part=='tr'].groupby('target').apply(lambda d:(d.pointer_x%2==0).mean()))
# fresh items: time since item last seen by another cookie, and first seen
it=ev.dropna(subset=['item_id']).sort_values('event_ts')
it['prev_seen']=it.groupby('item_id').event_ts.shift()
it['prev_ck']=it.groupby('item_id').cookie_id.shift()
it['lag_h']=(it.event_ts-it.prev_seen).dt.total_seconds()/3600
t=it[(it.part=='tr')&(it.window_start_ts>=pd.Timestamp('2026-04-10'))]
print(t.groupby('target').lag_h.describe())
print(pd.crosstab(pd.cut(t.lag_h.fillna(1e6),[-1,0.1,1,6,24,72,1e7]),t.target,normalize='columns').round(3))

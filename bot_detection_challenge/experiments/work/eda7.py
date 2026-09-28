import sys; sys.path.insert(0,'work')
from common_load import *
tr,te,meta,m=load()
ev=pd.read_csv('data/events.csv.gz',usecols=['cookie_id','event_ts','event_name']); ev['event_ts']=pd.to_datetime(ev.event_ts)
e=ev.merge(meta,on='cookie_id')
e['pos']=np.select([e.event_ts>=e.window_end_ts],['after'],'in')
print(pd.crosstab([e.event_name],[e.part,e.pos]))
c=e[e.event_name=='captcha_shown']; print(c.groupby(['part','pos']).cookie_id.nunique(), c.groupby('cookie_id').target.first().value_counts())
# lag of after events
a=e[e.pos=='after']; print(((a.event_ts-a.window_end_ts).dt.total_seconds()/3600).describe())
print('test events any date outside window?', e[(e.part=='te')].pos.value_counts())

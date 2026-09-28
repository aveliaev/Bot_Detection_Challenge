import pandas as pd, numpy as np
def load():
    tr=pd.read_csv('data/train.csv',parse_dates=['cookie_created_at','window_start_ts','window_end_ts'])
    te=pd.read_csv('data/test.csv',parse_dates=['cookie_created_at','window_start_ts','window_end_ts'])
    ev=pd.read_csv('data/events.csv.gz'); ev['event_ts']=pd.to_datetime(ev.event_ts)
    ev['row']=np.arange(len(ev))
    meta=pd.concat([tr.assign(part='tr'),te.assign(part='te',target=np.nan)],ignore_index=True)
    m=ev.merge(meta[['cookie_id','window_start_ts','window_end_ts','cookie_created_at','part','target']],on='cookie_id')
    m=m[(m.event_ts>=m.window_start_ts)&(m.event_ts<m.window_end_ts)].copy()
    return tr,te,meta,m

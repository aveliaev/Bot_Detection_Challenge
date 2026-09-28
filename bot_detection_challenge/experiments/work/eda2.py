import pandas as pd, numpy as np
pd.set_option('display.width',250); pd.set_option('display.max_columns',50)
tr=pd.read_csv('data/train.csv',parse_dates=['cookie_created_at','window_start_ts','window_end_ts'])
te=pd.read_csv('data/test.csv',parse_dates=['cookie_created_at','window_start_ts','window_end_ts'])
ev=pd.read_csv('data/events.csv.gz')
ev['event_ts']=pd.to_datetime(ev.event_ts,errors='coerce')
print('bad ts',ev.event_ts.isna().sum(), ev.event_ts.min(), ev.event_ts.max())
print(ev.dtypes)
meta=pd.concat([tr.assign(part='tr'),te.assign(part='te',target=np.nan)])
m=ev.merge(meta,on='cookie_id',how='left')
print('events w/o meta',m.part.isna().sum())
m['pos']=np.select([m.event_ts<m.window_start_ts,m.event_ts>=m.window_end_ts],['before','after'],'in')
print(pd.crosstab(m.part,m.pos))
print('before/after by target'); print(pd.crosstab(m.target,m.pos,normalize='index'))
print('before created?',(m.event_ts<m.cookie_created_at).sum())
# per-cookie: has events outside window
oc=m.groupby('cookie_id').agg(n=('pos','size'),n_in=('pos',lambda s:(s=='in').sum()),n_out=('pos',lambda s:(s!='in').sum()))
meta=meta.merge(oc,on='cookie_id',how='left').fillna({'n':0,'n_in':0,'n_out':0})
print('cookies with 0 in-window events', (meta.n_in==0).groupby(meta.part).mean())
print(meta.groupby('part')[['n','n_in','n_out']].describe().T)
t=meta[meta.part=='tr']
print('target vs has_out', t.groupby(t.n_out>0).target.agg(['mean','size']))
# duplicates
d=ev.duplicated().sum(); print('exact dup rows',d)
dd=m[m.pos=='in']; print('dups in-window',dd.duplicated(subset=ev.columns.tolist()).sum())
dup=dd[dd.duplicated(subset=ev.columns.tolist(),keep=False)]
print('cookies with dups by target',dup.groupby('cookie_id').target.first().value_counts(), dup.groupby('cookie_id').part.first().value_counts())
# row order sorted?
print('file sorted by cookie,ts?', (ev.groupby('cookie_id').event_ts.apply(lambda s:s.is_monotonic_increasing)).mean())
# platform normalization
print(pd.crosstab(ev.platform,ev.platform.str.lower()))
# events per cookie distributions by target
q=[.05,.25,.5,.75,.95,.99]
print(t.groupby('target').n_in.quantile(q).unstack())
print('events/cookie test in window',meta[meta.part=='te'].n_in.quantile(q))

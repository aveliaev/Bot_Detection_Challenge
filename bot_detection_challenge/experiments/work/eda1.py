import pandas as pd, numpy as np
pd.set_option('display.width',250); pd.set_option('display.max_columns',50)
tr=pd.read_csv('data/train.csv',parse_dates=['cookie_created_at','window_start_ts','window_end_ts'])
te=pd.read_csv('data/test.csv',parse_dates=['cookie_created_at','window_start_ts','window_end_ts'])
ev=pd.read_csv('data/events.csv.gz',dtype=str,keep_default_na=False)
print(tr.shape,te.shape,ev.shape)
print(tr.dtypes); print(ev.columns.tolist())
print('target',tr.target.value_counts())
print('cookie uniq',tr.cookie_id.nunique(),te.cookie_id.nunique(),'overlap',len(set(tr.cookie_id)&set(te.cookie_id)))
for d,n in [(tr,'tr'),(te,'te')]:
    print(n,'windows'); print(d.groupby('window_start_ts').size().to_string())
    print((d.window_end_ts-d.window_start_ts).value_counts())
    print(n,'created range',d.cookie_created_at.min(),d.cookie_created_at.max())
print(tr.groupby('window_start_ts').target.agg(['mean','size']))
# events raw string inspection
for c in ev.columns:
    s=ev[c]; print(f'--- {c}: nunique={s.nunique()} empty={(s=="").mean():.3f}'); print(s.value_counts().head(8).to_string()[:900])

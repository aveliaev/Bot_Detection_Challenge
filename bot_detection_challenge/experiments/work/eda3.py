import sys; sys.path.insert(0,'work')
from common_load import *
pd.set_option('display.width',250); pd.set_option('display.max_columns',50); pd.set_option('display.max_rows',200)
tr,te,meta,m=load()
t=m[m.part=='tr']
def rate(col,df=t,top=20):
    g=df.groupby(col,dropna=False).agg(n=('target','size'),ck=('cookie_id','nunique'),bot_ev=('target','mean'))
    print(g.sort_values('n',ascending=False).head(top))
rate('event_name'); rate('platform')
print(pd.crosstab(t.event_name,t.eid))
# field fill by event_name
cols=['item_id','item_category','item_location','seller_type','search_query','search_page','pointer_x','pointer_y']
print(m.groupby('event_name')[cols].apply(lambda d:d.notna().mean()).round(2))
print(m.groupby('platform')[['pointer_x']].apply(lambda d:d.notna().mean()).round(2))
print(pd.crosstab(m.event_name,m.platform.str.lower()))
# cookie-level platform consistency
cp=m.groupby('cookie_id').agg(np_=('platform','nunique'),npl=('platform',lambda s:s.str.lower().nunique()),nua=('user_agent','nunique'),target=('target','first'),part=('part','first'))
print(cp.groupby('part')[['np_','npl','nua']].describe().T)
print(cp[cp.part=='tr'].groupby('npl').target.agg(['mean','size'])); print(cp[cp.part=='tr'].groupby(cp.nua.clip(0,4)).target.agg(['mean','size']))
print(cp[cp.part=='tr'].groupby(cp.np_.clip(0,4)).target.agg(['mean','size']))
# dup rows per cookie vs target
key=['cookie_id','event_ts','eid','event_name','platform','user_agent','item_id','item_category','item_location','seller_type','search_query','search_page','pointer_x','pointer_y']
m['dup']=m.duplicated(subset=key)
dc=m.groupby('cookie_id').agg(nd=('dup','sum'),n=('dup','size'),target=('target','first'),part=('part','first'))
print(dc[dc.part=='tr'].groupby(dc.nd.clip(0,3)).target.agg(['mean','size']))
print('dup frac by part',dc.groupby('part').nd.sum()/dc.groupby('part').n.sum())
# same ts dup ignoring other fields
m['dupts']=m.duplicated(subset=['cookie_id','event_ts'])
print('same ts non-exact', (m.dupts & ~m.dup).sum())

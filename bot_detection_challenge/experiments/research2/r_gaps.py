import sys; sys.path.insert(0,'research2'); sys.path.insert(0,'cycle1')
from rcv import *
from features import load_events
D=Data(); meta=D.meta; ev=load_events(meta)
ev=ev.merge(meta[['cookie_id','target']],on='cookie_id'); ev=ev[ev.target.notna()]
ev['t']=ev.event_ts.astype('int64')//10**6; f=~ev.cookie_id.duplicated(); ev['dt']=ev.t.diff(); ev.loc[f,'dt']=np.nan
F=D.F; ev['web']=ev.cookie_id.map(F.pl_web); ev['noptr']=ev.cookie_id.map(F.ptr_none_web)
d=ev[ev.dt.notna()&(ev.dt<=1800)]
bins=[-1,0,1,2,3,4,5,6,8,10,12,15,20,25,30,40,50,60,75,90,120,180,300,600,1800]
seg=np.where(d.web==1,'web','mob')
tab=pd.crosstab(pd.cut(d.dt,bins),[seg,d.target],normalize='columns').round(3)
print(tab)

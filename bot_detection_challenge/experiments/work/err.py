import sys; sys.path.insert(0,'cycle1')
from cv import *
from metrics import operating_point
D=Data(); fam=json.load(open(WORK/'family.json'))
o=np.load(OOF_DIR/'F_all__dayblock.npy'); m=o>0
y=D.y; op=operating_point(y[m],o[m]); print(op)
X=D.F.loc[D.ids].copy(); X=X.join(D.pool('dayblock',0,*D.folds('dayblock')[0])[POOL_COLS]) if False else X
X['s']=o; X['y']=y; X=X[m]
thr=op['threshold']
X['grp']=np.select([(X.y==1)&(X.s>=thr),(X.y==1)&(X.s<thr),(X.y==0)&(X.s>=thr)],['TP','FN','FP'],'TN')
print(X.grp.value_counts())
cols=['n_events','ptr_frac_web','is_web','pl_android','pl_ios','ua_app','ua_headless','ua_is_lib','ptr_x_max','ptr_x_std','dt_median','dt_lt30','loc_nuniq','loc_ent','pop_mean','page_mean','frac_browse','cookie_age_d','n_items','sdt_log_std','n_event_types','dtexact_uniq_ratio']
pd.set_option('display.width',250)
print(X.groupby('grp')[cols].median().T.round(3))
# FN: platform mix, n_events distribution
print(X[X.y==1].groupby('grp').apply(lambda d: pd.Series({'n':len(d),'web':d.pl_web.mean(),'android':d.pl_android.mean(),'ios':d.pl_ios.mean(),'ev<=5':(d.n_events<=5).mean(),'ptr_none_web':d.ptr_none_web.mean()})))
print(X[X.y==0].groupby('grp').apply(lambda d: pd.Series({'n':len(d),'web':d.pl_web.mean(),'android':d.pl_android.mean(),'ios':d.pl_ios.mean(),'ev<=5':(d.n_events<=5).mean(),'ptr_none_web':d.ptr_none_web.mean()})))
# P@R by platform segment
for seg in ['pl_web','pl_android','pl_ios']:
    k=X[seg]==1; print(seg,k.sum(),int(X.y[k].sum()),'rate',round(X.y[k].mean(),3),'P@R in seg',round(precision_at_recall(X.y[k],X.s[k]),3))
# by event count
for lo,hi in [(1,3),(4,8),(9,20),(21,1000)]:
    k=(X.n_events>=lo)&(X.n_events<=hi); print('nev',lo,hi,k.sum(),'bots',int(X.y[k].sum()),'recall@thr',round(((X.s>=thr)&(X.y==1))[k].sum()/max(1,X.y[k].sum()),3),'FP',int(((X.s>=thr)&(X.y==0))[k].sum()))

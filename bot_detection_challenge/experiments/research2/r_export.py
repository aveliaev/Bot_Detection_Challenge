import sys; sys.path.insert(0,'research2')
from rcv import *
from metrics import operating_point
D=Data(); y=D.y
o=repeated_oof('B0_cat_top120'); f=np.load(OOF2/'B0_cat_top120__fwd7.npy')
pd.DataFrame({'cookie_id':D.ids,'day':D.day,'target':y,'oof_repeated_dayblock':o,'oof_fwd7':f}).to_csv('research2/oof_final_B0.csv',index=False)
mk=o>0; print('op point',operating_point(y[mk],o[mk]))
for sch,parts in [('P0',partitions(D.day)[0]),('fwd7',fwd7(D.day))]:
    s=np.load(OOF2/f'B0_cat_top120__{sch}.npy'); print(sch,[(round(precision_at_recall(y[v],s[v]),3),int(y[v].sum())) for _,v in parts])
print('median rep',np.median([0.8974,0.8975,0.9068,0.899,0.9007]))

import sys; sys.path.insert(0,'research2')
from rcv import *
D=Data(); y=D.y; X=D.F.loc[D.ids].copy(); X['y']=y
box=(X.pl_web==1)&(X.ptr_n>=3)&(X.ptr_x_std<330)
print('web box-pointer cookies',box.sum(),'bot rate',round(X.y[box].mean(),3),' | web ptr>=3 non-box bot rate',round(X.y[(X.pl_web==1)&(X.ptr_n>=3)&~box].mean(),3))
B=X[box]
print(B.groupby(pd.cut(B.sdt_band_30_90,[-.01,0,.1,.2,.3,.5,1])).y.agg(['mean','size']))
print(B.groupby(pd.cut(B.cookie_age_d,[-1,3,7,30,400])).y.agg(['mean','size']))
print(pd.crosstab(pd.cut(B.sdt_band_30_90,[-.01,.1,.25,1]),pd.cut(B.cookie_age_d,[-1,7,400]),values=B.y,aggfunc='mean').round(2))
print(pd.crosstab(pd.cut(B.sdt_band_30_90,[-.01,.1,.25,1]),pd.cut(B.cookie_age_d,[-1,7,400])))
# humans overall: how often do true humans (non-box) have ptr_x_std<330 given n ptr
H=X[(X.pl_web==1)&(X.ptr_n>=3)]
print(H.groupby(pd.cut(H.ptr_n,[2,5,10,20,200])).apply(lambda d: pd.Series({'human_box_rate':(d.ptr_x_std[d.y==0]<330).mean(),'bot_box_rate':(d.ptr_x_std[d.y==1]<330).mean(),'n':len(d)})))

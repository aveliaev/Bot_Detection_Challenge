import sys; sys.path.insert(0,'research2')
from rcv import *
D=Data(); y=D.y
o=repeated_oof('B0_cat_top120'); mk=o>0; yy=y[mk]; s=o[mk]
order=np.argsort(-s); ys=yy[order]
print('n pos',yy.sum(),'n',len(yy))
for k in [50,100,200,300,400,500,540,600,700,800,1000]:
    print(f'top{k:5d} precision={ys[:k].mean():.3f} recall={ys[:k].sum()/yy.sum():.3f}')
# precision in rank bins
for a,b in [(0,100),(100,200),(200,300),(300,400),(400,500),(500,600),(600,700),(700,800),(800,1000)]:
    print(f'rank {a}-{b}: bot share {ys[a:b].mean():.2f}')

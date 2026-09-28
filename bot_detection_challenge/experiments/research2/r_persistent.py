import sys; sys.path.insert(0,'research2')
from rcv import *
D=Data(); y=D.y
models=['B0_cat_top120','B0_lgb_top120','B1_cat_graph','C1_cat_nb','D1_cat_misc2','G1_cat_segment_models','G2_cat_global_plus_mobile']
flag=[];
for m in models:
    for i in range(5):
        o=np.load(OOF2/f'{m}__P{i}.npy'); mk=o>0; thr=np.sort(o[mk])[::-1][600]; flag.append((o>=thr)&mk)
Fl=np.array(flag); mk=np.load(OOF2/'B0_cat_top120__P0.npy')>0
share=Fl.mean(0)
neg=mk&(y==0); pos=mk&(y==1)
print('negatives flagged by share of 35 model-runs:'); print(pd.cut(share[neg],[-.01,0,.2,.5,.8,.99,1]).value_counts().sort_index())
print('positives flagged share:'); print(pd.cut(share[pos],[-.01,0,.2,.5,.8,.99,1]).value_counts().sort_index())
pfp=neg&(share>=0.99); pfn=pos&(share==0)
print('persistent FP',pfp.sum(),'persistent FN',pfn.sum(),'always-caught TP',(pos&(share>=.99)).sum())
# empirical ceiling: if persistent FPs were label noise (true bots), precision at recall .7?
o=repeated_oof('B0_cat_top120'); yy=y.copy(); yy2=yy.copy(); yy2[pfp]=1
print('P@R base',round(precision_at_recall(y[mk],o[mk]),4),' if persistent FP relabelled (hypothetical, analysis only)',round(precision_at_recall(yy2[mk],o[mk]),4))
X=D.F.loc[D.ids]
cols=['pl_web','pl_android','n_events','ptr_none_web','ptr_x_std','ptr_box_frac','sdt_median','sdt_band_30_90','sdt_band_2_20','loc_nuniq','page_mean','cookie_age_d','ua_is_lib','ua_headless']
G=pd.DataFrame({'pFP':X[pfp][cols].median(),'always_TP':X[pos&(share>=.99)][cols].median(),'pFN':X[pfn][cols].median(),'TN':X[neg&(share==0)][cols].median()})
print(G.round(3))
np.save(OOF2/'persist_share.npy',share)

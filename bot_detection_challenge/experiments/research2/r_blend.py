import sys,json; sys.path.insert(0,'research2')
from rcv import *
D=Data(); y=D.y
def blend(ws,sch):
    arrs=[(w,np.load(OOF2/f'{n}__{sch}.npy')) for n,w in ws.items()]; mk=np.all([a>0 for _,a in arrs],0)
    s=np.zeros(len(y)); 
    for w,a in arrs: s[mk]+=w*pd.Series(a[mk]).rank(pct=True).values
    s[~mk]=0; return s/sum(ws.values())
def evaluate(nm,ws):
    per={s:precision_at_recall(y[(o:=blend(ws,s))>0],o[o>0]) for s in ['P0','P1','P2','P3','P4','fwd7']}
    rep=[per[f'P{i}'] for i in range(5)]
    ro=np.mean([blend(ws,f'P{i}') for i in range(5)],0)
    return nm,np.mean(rep),np.std(rep),np.min(rep),per['fwd7'],ro
base={'B0_cat_top120':1}
cands={'B0':base,
 'cat+et+xgb (cycle1 recipe)':{'B0_cat_top120':.6,'K_et':.2,'K_xgb':.2},
 'catnb+et+xgb':{'C1_cat_nb':.6,'K_et':.2,'K_xgb':.2},
 'cat+catnb':{'B0_cat_top120':.5,'C1_cat_nb':.5},
 'cat+catnb+catpool2':{'B0_cat_top120':1/3,'C1_cat_nb':1/3,'C2_cat_pool2':1/3},
 'cat+catnb+G2+pool2':{'B0_cat_top120':.25,'C1_cat_nb':.25,'C2_cat_pool2':.25,'G2_cat_global_plus_mobile':.25},
 'all-cat-variants+et+xgb':{'B0_cat_top120':.15,'C1_cat_nb':.15,'C2_cat_pool2':.15,'G2_cat_global_plus_mobile':.15,'B1_cat_graph':.1,'K_et_nb':.15,'K_xgb_nb':.15},
 'catnb+etnb+xgbnb':{'C1_cat_nb':.6,'K_et_nb':.2,'K_xgb_nb':.2},
}
names=[n for c in cands.values() for n in c]
import os
avail={n for n in names if os.path.exists(OOF2/f'{n}__P0.npy')}
_,_,_,_,_,rb=evaluate('B0',base)
print('corr (spearman) of repeated OOF:')
R=pd.DataFrame({n:repeated_oof(n) for n in sorted(avail)}); mk=(R>0).all(1); print(R[mk].corr(method='spearman').round(3))
for nm,ws in cands.items():
    if not set(ws)<=avail: print('skip',nm); continue
    n,m,s,mn,f,ro=evaluate(nm,ws); pb=paired_bootstrap(y,ro,rb,n=1000)
    print(f'{nm:28s} rep={m:.4f}±{s:.4f} min={mn:.4f} fwd7={f:.4f} | repOOF P@R={precision_at_recall(y[ro>0],ro[ro>0]):.4f} P(>B0)={pb["p_better"]:.2f} Δ={pb["point"]:+.4f} CI95=({pb["ci95"][0]:+.3f},{pb["ci95"][1]:+.3f})')

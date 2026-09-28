import sys,itertools; sys.path.insert(0,'cycle1')
from cv import *
from metrics import operating_point
D=Data(); y=D.y
names=sys.argv[1].split(',')
def load(n,s): 
    o=np.load(OOF_DIR/f'{n}__{s}.npy'); return o
res={}
for s in ['dayblock','fwd7']:
    O=np.vstack([load(n,s) for n in names]); m=(O>0).all(0)
    O=np.vstack([pd.Series(o[m]).rank(pct=True).values for o in O]); yy=y[m]
    print('==',s,'n',m.sum()); print(pd.DataFrame(O.T,columns=names).corr(method='spearman').round(3))
    for i,n in enumerate(names): print(f'  {n:24s} P@R={precision_at_recall(yy,O[i]):.4f} PR={report(yy,O[i])["pr_auc"]:.4f}')
    res[s]=(O,yy)
# weight grid (step .1) chosen on dayblock, reported on fwd7 too
grid=[w for w in itertools.product(np.arange(0,1.01,.1),repeat=len(names)) if abs(sum(w)-1)<1e-9]
rows=[]
for w in grid:
    w=np.array(w); r=[precision_at_recall(res[s][1],w@res[s][0]) for s in ['dayblock','fwd7']]
    pr=[report(res[s][1],w@res[s][0])['pr_auc'] for s in ['dayblock','fwd7']]
    rows.append((tuple(np.round(w,1)),*r,*pr))
R=pd.DataFrame(rows,columns=['w','pr_day','pr_fwd','prauc_day','prauc_fwd']); R['avg']=(R.pr_day+R.pr_fwd)/2
print(R.sort_values('pr_day',ascending=False).head(8).round(4).to_string(index=False))
print(R.sort_values('avg',ascending=False).head(8).round(4).to_string(index=False))
eq=np.ones(len(names))/len(names); print('equal',[round(precision_at_recall(res[s][1],eq@res[s][0]),4) for s in res])

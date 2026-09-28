import sys,warnings; warnings.filterwarnings('ignore'); sys.path.insert(0,'cycle1')
import pandas as pd,numpy as np
from features import load_meta,family_map
from metrics import precision_at_recall
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp
tr,te,meta=load_meta()
F=pd.read_parquet('work/features.parquet')
Ftr=F.loc[tr.cookie_id]; y=tr.target.values; Fte=F.loc[te.cookie_id]
fam=family_map(F.columns)
rows=[]
for c in F.columns:
    x=Ftr[c].fillna(-999).values
    auc=roc_auc_score(y,x); s=x if auc>=.5 else -x
    ks=ks_2samp(Ftr[c].dropna(),Fte[c].dropna()).statistic if Fte[c].notna().any() else np.nan
    rows.append((c,fam[c],max(auc,1-auc),'+' if auc>=.5 else '-',precision_at_recall(y,s),Ftr[c].isna().mean(),ks,
                 Ftr[c][y==0].median(),Ftr[c][y==1].median()))
R=pd.DataFrame(rows,columns=['feat','fam','auc','dir','p@r70','nan','ks_tr_te','med0','med1']).sort_values('auc',ascending=False)
pd.set_option('display.width',250); pd.set_option('display.max_rows',400)
print(R.head(70).round(3).to_string(index=False))
print(R.groupby('fam').auc.max().sort_values())
print('top shift'); print(R.sort_values('ks_tr_te',ascending=False).head(15).round(3).to_string(index=False))
R.to_csv('work/univariate.csv',index=False)

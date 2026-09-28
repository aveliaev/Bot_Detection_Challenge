import sys; sys.path.insert(0,'research2')
from rcv import *
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
D=Data(); y=D.y; F=D.F.loc[D.ids]
o=repeated_oof('B0_cat_top120'); thr=np.sort(o[o>0])[::-1][600]
feat=['pl_web','pl_android','pl_ios','ua_is_lib','ua_headless','ua_app','ptr_none_web','ptr_x_std','ptr_box_frac','ptr_exact_edge_frac',
      'sdt_median','sdt_log_std','sdt_band_2_20','dt_lt2','page_mean','page_max','query_repeat_ratio','loc_nuniq','cat_nuniq','log_cookie_age_d','log_n','frac_contact','frac_browse','n_sessions','hour_mean','frac_ps','frac_fav','frac_srv','ev_per_item']
X=F[feat].copy(); X=X.fillna(X.median())
P=X[y==1]; Z=StandardScaler().fit_transform(P)
for k in range(2,9):
    km=KMeans(k,n_init=10,random_state=0).fit(Z); print(k,round(silhouette_score(Z,km.labels_),3))
k=4; km=KMeans(k,n_init=20,random_state=0).fit(Z); lab=km.labels_
Pd=P.copy(); Pd['cl']=lab; Pd['caught']=(o[y==1]>=thr); Pd['day']=D.day[y==1]
pd.set_option('display.width',250); pd.set_option('display.max_columns',40)
print(Pd.groupby('cl')[feat[:14]].median().round(2).T); print(Pd.groupby('cl')[feat[14:]].median().round(2).T)
print(Pd.groupby('cl').agg(n=('caught','size'),recall=('caught','mean')))
print('cluster share by week', pd.crosstab(Pd.cl,Pd.day>=7,normalize='columns').round(2))

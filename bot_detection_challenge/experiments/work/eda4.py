import sys,re; sys.path.insert(0,'work')
from common_load import *
pd.set_option('display.width',250); pd.set_option('display.max_columns',50); pd.set_option('display.max_rows',300)
tr,te,meta,m=load()
m['pl']=m.platform.str.lower().replace({'desktop':'web','iphone':'ios'})
cp=m.groupby('cookie_id').agg(npl=('pl','nunique'),target=('target','first'))
print('norm platform nunique',cp.npl.value_counts())
# desktop vs web
m['pl2']=m.platform.str.lower()
c=m.groupby('cookie_id').pl2.agg(lambda s:'|'.join(sorted(set(s))))
print(c.value_counts().head(10))
# UA structure
ua=m.drop_duplicates('user_agent')[['user_agent']]
print(len(ua)); 
for u in ua.user_agent.sample(40,random_state=1): print(u)

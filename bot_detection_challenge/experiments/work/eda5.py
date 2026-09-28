import sys,re; sys.path.insert(0,'work')
from common_load import *
pd.set_option('display.width',250); pd.set_option('display.max_columns',50); pd.set_option('display.max_rows',300)
tr,te,meta,m=load()
def fam(u):
    if re.search(r'python|node-fetch|curl|wget|go-http|java/|scrapy|axios|okhttp/(?!4\.11)',u,re.I) and not u.startswith('Avito'): return 'lib:'+re.split(r'[/ ]',u)[0]
    if 'HeadlessChrome' in u: return 'headless'
    if 'bot' in u.lower() or 'spider' in u.lower() or 'crawl' in u.lower(): return 'botword'
    if u.startswith('Avito/'): return 'app_'+('ios' if 'iPhone' in u else 'android')
    if 'YaBrowser' in u: return 'yabro'
    if 'Firefox' in u: return 'firefox'
    if 'Mobile Safari' in u and 'Android' in u: return 'chrome_android'
    if 'Macintosh' in u: return 'chrome_mac'
    if 'Windows' in u: return 'chrome_win'
    if 'X11' in u: return 'linux'
    if 'iPhone' in u: return 'safari_ios'
    return 'other'
m['uaf']=m.user_agent.map(fam)
c=m.groupby('cookie_id').agg(uaf=('uaf',lambda s:'|'.join(sorted(set(s)))),part=('part','first'),target=('target','first'),pl=('platform',lambda s:s.iloc[0].lower().replace('desktop','web').replace('iphone','ios')),nua=('user_agent','nunique'))
print(pd.crosstab([c.uaf],[c.part]).join(c[c.part=='tr'].groupby('uaf').target.mean()).sort_values('tr',ascending=False))
print(pd.crosstab(c.uaf,c.pl).head(40))
# all UAs non-mozilla
print(m[~m.user_agent.str.startswith('Mozilla')].user_agent.value_counts())
# multi UA cookies
mu=m[m.cookie_id.isin(c[c.nua>1].index)]
print(mu.groupby('cookie_id').user_agent.apply(lambda s:list(dict.fromkeys(s))).head(8).to_string())

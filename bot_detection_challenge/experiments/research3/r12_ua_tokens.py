"""UA token temporal target encoding: for each cookie, bot rate of its token among labelled cookies of strictly earlier days.
Report AUC overall / android / web and near-threshold TP vs pFP, and week1->week2 stability."""
import sys, json; sys.path.insert(0, 'research3')
from common import *
from sklearn.metrics import roc_auc_score
pd.set_option('display.width', 250)
ev = events(); m = meta(); o = pd.read_parquet(CACHE / 'groups_opt.parquet'); pfp = json.load(open(CACHE / 'persistent.json'))
c = ev.groupby('cookie_id').agg(ua=('user_agent', 'first'), pl=('pl', 'first')).join(m[['target', 'day']])
u = c.ua.fillna('')
c['device'] = u.str.extract(r'Android \d+; ([^;)]+)')[0]
c['device'] = c.device.fillna(u.str.extract(r'^Avito/[\d.]+ \((?:iOS|Android) [\d.]+; ([^;)]+)\)')[0])
c['iosdev'] = u.str.extract(r'\((iPhone|iPad)[^)]*\)')[0]
c['browser'] = u.str.extract(r'(Chrome|Firefox|YaBrowser|HeadlessChrome|Edg|OPR|Version)/(\d+)').apply(lambda r: f'{r[0]}/{r[1]}' if pd.notna(r[0]) else None, axis=1)
c['os'] = u.str.extract(r'(Windows NT [\d.]+|Mac OS X [\d_]+|Android \d+|iOS [\d_.]+|iPhone OS [\d_]+|X11; Linux)')[0]
c['appv'] = u.str.extract(r'^(Avito/[\d.]+)')[0]
c['full'] = u
c['pl_dev'] = c.pl + '|' + c.device.fillna('-') + '|' + c.appv.fillna('-')
toks = ['device', 'browser', 'os', 'appv', 'full', 'pl_dev']
print('distinct values:', {t: c[t].nunique() for t in toks})
def te(col, a=20, p0=0.081):
    out = pd.Series(np.nan, index=c.index)
    for d in sorted(c.day.unique()):
        h = c[(c.day < min(d, 14)) & c.target.notna()]
        if not len(h): continue
        g = h.groupby(col).target.agg(['sum', 'size'])
        r = (g['sum'] + a * p0) / (g['size'] + a)
        idx = c.index[c.day == d]
        out[idx] = c.loc[idx, col].map(r).fillna(p0).values
    return out
L = c[c.target.notna() & (c.day >= 2)]
grp = o.grp.reindex(L.index)
near = L.index[(L.target == 1) & o['rank'].reindex(L.index).between(300, 604)]
for t in toks:
    s = te(t); c['te_' + t] = s
    r = {'all': roc_auc_score(L.target, s[L.index])}
    for pl in ['android', 'web', 'ios']:
        k = L[L.pl == pl]; r[pl] = roc_auc_score(k.target, s[k.index])
    k = list(near) + pfp; r['TPnear_vs_pFP'] = roc_auc_score([1] * len(near) + [0] * len(pfp), s[k])
    # within android: FN vs hard? AUC among android bots missed by B0 (E) vs android negatives
    print(t, {k_: round(v, 3) for k_, v in r.items()})
c.to_parquet(CACHE / 'ua_tokens.parquet')

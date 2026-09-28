"""Per-cookie 'service representation': normalised histograms of raw fields (activity-invariant).
Blocks: ev (event-type hist), tr (transition hist), dt (log-delay hist), px/py (pointer hist + moments),
cat (category hist), pg (search-page hist), ua (UA tokens), pl (platform), misc (log n, log age, n_loc/n)."""
import sys; sys.path.insert(0, 'research3')
from common import *
EV = ['item_view', 'search_results_view', 'photo_swipe', 'favorite_add', 'seller_page_view', 'contact_phone_show',
      'login', 'contact_chat_open', 'contact_message_sent']


def _hist(df, key, val, prefix, cats=None):
    t = pd.crosstab(df[key], df[val])
    if cats is not None:
        t = t.reindex(columns=cats, fill_value=0)
    t = t.div(t.sum(axis=1), axis=0)
    t.columns = [f'{prefix}_{c}' for c in t.columns]
    return t


def build():
    p = CACHE / 'rep.parquet'
    if p.exists():
        return pd.read_parquet(p)
    ev = events()
    ids = ev.cookie_id.unique()
    B = []
    B.append(_hist(ev, 'cookie_id', 'event_name', 'ev', EV))
    tr = ev[ev.prev_ev != 'START'].assign(b=lambda d: d.prev_ev.str[:6] + '>' + d.event_name.str[:6])
    top = tr.b.value_counts().index[:30]
    B.append(_hist(tr[tr.b.isin(top)], 'cookie_id', 'b', 'tr'))
    d = ev[ev['dt'].notna()].copy()
    d['bin'] = pd.cut(d['dt'], [-1, 1, 2, 3, 5, 8, 12, 20, 30, 45, 60, 90, 180, 600, 1800, 1e9], labels=False)
    B.append(_hist(d, 'cookie_id', 'bin', 'dt'))
    q = ev[ev.pointer_x.notna()].copy()
    q['xb'] = pd.cut(q.pointer_x, [-1, 0, 240, 480, 720, 960, 1200, 1440, 1680, 1919, 1920], labels=False)
    q['yb'] = pd.cut(q.pointer_y, [-1, 0, 180, 360, 540, 720, 900, 1079, 1080], labels=False)
    B.append(_hist(q, 'cookie_id', 'xb', 'px')); B.append(_hist(q, 'cookie_id', 'yb', 'py'))
    g = q[~q.pointer_x.isin([0, 1920]) & ~q.pointer_y.isin([0, 1080])].groupby('cookie_id')
    B.append(pd.DataFrame({'pm_sx': g.pointer_x.std() / 550, 'pm_sy': g.pointer_y.std() / 310,
                           'pm_cx': g.pointer_x.median() / 1920, 'pm_cy': g.pointer_y.median() / 1080}))
    B.append(_hist(ev[ev.item_category.notna()], 'cookie_id', 'item_category', 'cat'))
    s = ev[ev.event_name == 'search_results_view'].copy()
    s['pb'] = pd.cut(s.search_page, [0, 1, 2, 3, 5, 8, 12, 20, 1e9], labels=False)
    B.append(_hist(s, 'cookie_id', 'pb', 'pg'))
    u = ev.groupby('cookie_id').user_agent.first().fillna('')
    toks = {'lib': r'^(?:python-|node-fetch|curl/|Go-http|Scrapy|wget|Java/|axios)', 'headless': 'HeadlessChrome', 'app': '^Avito/',
            'yabro': 'YaBrowser', 'ff': 'Firefox', 'mac': 'Macintosh', 'win': 'Windows', 'x11': 'X11', 'android': 'Android', 'iphone': 'iPhone', 'edge': 'Edg/'}
    B.append(pd.DataFrame({f'ua_{k}': u.str.contains(v, regex=True).astype(float) for k, v in toks.items()}))
    pl = ev.groupby('cookie_id').pl.first()
    B.append(pd.get_dummies(pl, prefix='pl').astype(float))
    gg = ev.groupby('cookie_id')
    n = gg.size()
    B.append(pd.DataFrame({'m_logn': np.log1p(n) / 5, 'm_logage': np.log1p(gg.age_h.min().clip(lower=0) / 24) / 5,
                           'm_locpe': gg.item_location.nunique() / n, 'm_itempe': gg.item_id.nunique() / n,
                           'm_ptrfrac': ev.pointer_x.notna().groupby(ev.cookie_id).mean(),
                           'm_edge': (ev.pointer_x.isin([0, 1920]) | ev.pointer_y.isin([0, 1080])).groupby(ev.cookie_id).mean(),
                           'm_qrep': 1 - s.groupby('cookie_id').search_query.nunique() / s.groupby('cookie_id').size()}))
    R = pd.concat(B, axis=1).reindex(ids).fillna(0.0)
    R.columns = [str(c) for c in R.columns]
    R.to_parquet(p)
    return R


BLOCKS = ['ev', 'tr', 'dt', 'px', 'py', 'pm', 'cat', 'pg', 'ua', 'pl', 'm']


def block_cols(R, blocks):
    return [c for c in R.columns if c.split('_')[0] in blocks]


if __name__ == '__main__':
    R = build(); print(R.shape); print({b: len(block_cols(R, [b])) for b in BLOCKS})

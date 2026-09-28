"""C3.1 anonymized timelines: A (FP), B (TP same rank region), C (easy TP), web platform, similar n events."""
import sys; sys.path.insert(0, 'research3')
from common import *
ev = events(); o = pd.read_parquet(CACHE / 'groups_opt.parquet'); ck = pd.read_parquet(CACHE / 'ck_groups.parquet')
SH = dict(item_view='IV', search_results_view='SRV', photo_swipe='PS', favorite_add='FAV', seller_page_view='SP',
          contact_phone_show='PHONE', login='LOGIN', contact_chat_open='CHAT', contact_message_sent='MSG')
def show(c, label, maxrows=22):
    x = ev[ev.cookie_id == c]
    t0 = x.t.iloc[0]
    items = {v: f'i{k}' for k, v in enumerate(pd.unique(x.item_id.dropna()))}
    print(f'--- {label}  pl={x.pl.iloc[0]} n={len(x)} age_at_first={x.age_h.iloc[0]/24:.1f}d rank={int(o.loc[c,"rank"])} ua={x.user_agent.iloc[0][:60]}')
    for _, r in x.head(maxrows).iterrows():
        ptr = '' if pd.isna(r.pointer_x) else f'({int(r.pointer_x)},{int(r.pointer_y)})'
        q = '' if pd.isna(r.search_query) else f'q="{r.search_query}" p={int(r.search_page)}'
        it = items.get(r.item_id, '')
        print(f'  t=+{r.t-t0:7.0f}s dt={("" if pd.isna(r["dt"]) else int(r["dt"])):>5} {SH[r.event_name]:6s} {it:4s} {str(r.item_category)[:12]:12s} {str(r.item_location)[:10]:10s} {ptr:12s} {q}')
rng = np.random.RandomState(1)
for g, lab in [('A', 'PERSISTENT/OP FP (target=0)'), ('B', 'TP near threshold'), ('C', 'easy TP')]:
    cands = ck[(ck.grp == g) & (ck.pl == 'web') & ck.n.between(15, 60)].index
    for c in rng.choice(cands, 3, replace=False):
        show(c, lab)

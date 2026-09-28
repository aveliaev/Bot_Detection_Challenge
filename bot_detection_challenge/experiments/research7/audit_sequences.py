"""C7.2 sequence / pointer data audit (in-window events, dedup, sorted; cycle-3 cache)."""
import sys; from pathlib import Path
R7 = Path(__file__).resolve().parent; ROOT = R7.parent
sys.path.insert(0, str(ROOT / 'research3'))
import numpy as np, pandas as pd
from common import events, meta
pd.set_option('display.width', 250)
ev = events(); m = meta()
ev['y'] = ev.cookie_id.map(m.target)
c = ev.cookie_id.values; same = np.r_[False, c[1:] == c[:-1]]
hp = ev.pointer_x.notna().values
dt = ev['dt'].values
pair = same & hp & np.r_[False, hp[:-1]]
valid_pair = pair & (np.nan_to_num(dt, nan=1e9) <= 300)          # locked: same cookie, both with pointer, gap <= 300 s
trip = valid_pair & np.r_[False, valid_pair[:-1]]
g = pd.DataFrame({'cookie_id': c, 'hp': hp, 'pair': valid_pair, 'trip': trip, 'vdt': ~np.isnan(dt), 'sess': (~same) | (np.nan_to_num(dt) > 1800)})
A = g.groupby('cookie_id').agg(n_events=('hp', 'size'), n_ptr=('hp', 'sum'), n_ptr_pairs=('pair', 'sum'), n_ptr_triplets=('trip', 'sum'),
                               n_valid_dt=('vdt', 'sum'), n_sessions=('sess', 'sum'))
A = A.join(m[['target']]).join(ev.groupby('cookie_id').pl.first())
A['set'] = np.where(A.target.isna(), 'test', A.target.map({0.0: 'human', 1.0: 'bot'}))
A.to_parquet(R7 / 'cache' / 'audit.parquet')
print('== quantiles per set');
print(A.groupby('set')[['n_events', 'n_ptr', 'n_ptr_pairs', 'n_ptr_triplets', 'n_sessions']].quantile([.1, .25, .5, .75, .9]).unstack(level=1).round(1).T.to_string())
for by in ['set', 'pl']:
    print(f'\n== share of cookies with >= k events, by {by}')
    print(pd.DataFrame({f'>={k}': A.groupby(by).n_events.apply(lambda s: (s >= k).mean()) for k in (3, 5, 10, 20, 40)}).round(3))
    print(f'== share with >= k pointer observations / valid consecutive pointer pairs, by {by}')
    print(pd.DataFrame({**{f'ptr>={k}': A.groupby(by).n_ptr.apply(lambda s: (s >= k).mean()) for k in (2, 3, 5, 10, 20)},
                        **{f'pairs>={k}': A.groupby(by).n_ptr_pairs.apply(lambda s: (s >= k).mean()) for k in (1, 4, 9, 19)}}).round(3))
L = A[A.target.notna()]
print('\n== labelled cookies by platform x class: n, median events, median valid pointer pairs')
print(L.groupby(['pl', 'set']).agg(n=('n_events', 'size'), med_events=('n_events', 'median'), med_pairs=('n_ptr_pairs', 'median'), share_pairs_ge4=('n_ptr_pairs', lambda s: (s >= 4).mean())).round(2))
d = ev.loc[pair, 'dt']
print('\ngap between consecutive pointer events (s): quantiles', d.quantile([.1, .5, .9, .99]).to_dict(), '| share <= 300 s:', round((d <= 300).mean(), 3))

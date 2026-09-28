"""C3.14 descriptive: do hybrid humans (ptr_comp, target 0) share items with bots or with other hybrids?"""
import sys; sys.path.insert(0, 'research3')
from common import *
from cv import Data
from sklearn.metrics import roc_auc_score
D = Data(); ci = D.ci; m = meta()
T = pd.read_parquet(CACHE / 'traits.parquet')
lab = m.target
ci = ci.join(lab, on='cookie_id')
hyb = set(T[(T.ptr_comp == 1)].index[lab.reindex(T[T.ptr_comp == 1].index) == 0])
ci['cls'] = np.where(ci.target == 1, 'bot', np.where(ci.cookie_id.isin(hyb), 'hyb', np.where(ci.target == 0, 'hum', 'test')))
cnt = ci.drop_duplicates(['cookie_id', 'item_id']).groupby(['item_id', 'cls']).size().unstack(fill_value=0)
print('items per class', (cnt > 0).sum().to_dict())
x = ci.drop_duplicates(['cookie_id', 'item_id']).join(cnt, on='item_id')
# leave-one-out
for c in ['bot', 'hyb', 'hum']:
    x[c + '_o'] = x[c] - (x.cls == c)
S = x.groupby('cookie_id')[['bot_o', 'hyb_o', 'hum_o']].mean().join(lab).join(T[['ptr_comp', 'pool']])
S['cls'] = np.where(S.index.isin(hyb), 'hyb', np.where(S.target == 1, np.where(S.ptr_comp == 1, 'bot_comp', 'bot'), np.where(S.target == 0, 'hum', 'test')))
print(S.groupby('cls')[['bot_o', 'hyb_o', 'hum_o']].mean().round(3))
# normalise: hyb share relative to bot share
S['ratio'] = (S.hyb_o + .01) / (S.bot_o + .01)
print(S.groupby('cls').ratio.median().round(3))
k = S[S.cls.isin(['hyb', 'bot_comp'])]
print('AUC hyb vs bot_comp using hyb_o/bot_o ratio (descriptive, in-sample LOO):', roc_auc_score(k.cls == 'hyb', k.ratio))
# expected number of hybrids per bot-view under a single shared pool: hyb_total/bot_total
print('hybrid cookie count', len(hyb), 'bot count', int((lab == 1).sum()))

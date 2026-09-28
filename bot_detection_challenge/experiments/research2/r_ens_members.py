import sys; sys.path.insert(0,'research2')
from rcv import *
from nb import build_tokens, nb_features
import r_stage2
H=Harness(); T=build_tokens(H.D.meta)
H.add_family('nb',lambda D,lab,ids: nb_features(T,lab,ids))
H.run('K_et',model='et',params=dict(min_samples_leaf=3),hypothesis='ensemble member',leak_risk='low')
H.run('K_xgb',model='xgb',hypothesis='ensemble member',leak_risk='low')
H.run('K_et_nb',model='et',params=dict(min_samples_leaf=3),fams=('pool','nb'),hypothesis='ET + NB evidence',leak_risk='medium')
H.run('K_xgb_nb',model='xgb',fams=('pool','nb'),hypothesis='XGB + NB evidence',leak_risk='medium')
r_stage2.H.run('I2_xgb_pairwise_mean',fit_fn=r_stage2.rank_fit,hypothesis='pairwise ranking, mean pair sampling',leak_risk='low')

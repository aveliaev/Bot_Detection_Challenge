import sys; sys.path.insert(0,'research2')
from rcv import *
from nb import build_tokens, nb_features_clean
H=Harness(); T=build_tokens(H.D.meta)
H.add_family('nbc',lambda D,lab,ids: nb_features_clean(T,lab,ids))
H.run('C3_cat_nbclean',fams=('pool','nbc'),hypothesis='NB evidence, lattice-free (adv 0.54)',leak_risk='low: inner-OOF, tests ok, adv 0.54')
H.run('C3_et_nbclean',model='et',params=dict(min_samples_leaf=3),fams=('pool','nbc'),hypothesis='ET + clean NB',leak_risk='low')

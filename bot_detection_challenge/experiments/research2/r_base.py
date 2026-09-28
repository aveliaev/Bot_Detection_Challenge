import sys; sys.path.insert(0,'research2')
from rcv import *
H=Harness()
H.run('B0_cat_top120',hypothesis='baseline',leak_risk='low')
H.run('B0_lgb_top120',model='lgb',hypothesis='baseline fast screener',leak_risk='low')

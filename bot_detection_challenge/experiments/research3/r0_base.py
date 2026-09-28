"""C3.0: reproduce cycle-2 baseline B0 (CatBoost top-120 + pool) on repeated day-block CV + fwd7."""
import sys; sys.path.insert(0, 'research2')
from rcv import *
import rcv; rcv.LOG = Path(__file__).resolve().parent.parent / 'research3' / 'experiments3.jsonl'
H = Harness()
H.run('B0_cat_top120', hypothesis='C3.0 baseline reproduction', leak_risk='low')

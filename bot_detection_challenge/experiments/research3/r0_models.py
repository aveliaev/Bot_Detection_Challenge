"""C3.0: extra model families on the same partitions -> persistence of FPs across models."""
import sys; sys.path.insert(0, 'research2')
from rcv import *
import rcv; rcv.LOG = Path(__file__).resolve().parent.parent / 'research3' / 'experiments3.jsonl'
H = Harness()
for mdl in ['lgb', 'xgb', 'et']:
    H.run(f'B0_{mdl}_top120', model=mdl, hypothesis='C3.0 persistence ensemble member', leak_risk='low')

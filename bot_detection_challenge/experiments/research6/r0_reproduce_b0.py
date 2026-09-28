"""C6.0: fresh re-run of B0 on P0..P4 + fwd7 into research6/oof and bit-level comparison with the cycle-2 OOF."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'research2'))
from rcv import *
import rcv
R6 = Path(__file__).resolve().parent
rcv.OOF2 = R6 / 'oof'; rcv.LOG = R6 / 'experiments6.jsonl'
H = Harness()
H.run('B0_cat_top120', hypothesis='C6.0 baseline reproduction', leak_risk='low')
for s in ['P0', 'P1', 'P2', 'P3', 'P4', 'fwd7']:
    a, b = np.load(R6 / 'oof' / f'B0_cat_top120__{s}.npy'), np.load(HERE / 'oof' / f'B0_cat_top120__{s}.npy')
    print(s, 'identical' if np.array_equal(a, b) else f'DIFF max {np.abs(a - b).max()}')

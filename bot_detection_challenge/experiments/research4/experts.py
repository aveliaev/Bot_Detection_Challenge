"""C4: independent channel experts on the SAME folds as B0 (P0..P4 + fwd7), raw probabilities saved.
Mouse expert is trained and scored only on cookies with mouse data (>=1 pointer event)."""
import sys, json, time, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'research2'))
import numpy as np, pandas as pd
from rcv import Harness, BASE_COLS
from cv import make_model, POOL_COLS
from channels import channels, R4
H = Harness(); D = H.D
CH = channels(list(D.F.columns))
json.dump(CH, open(R4 / 'channels.json', 'w'), indent=1)
print({k: len(v) for k, v in CH.items()}, flush=True)
mouse_ok = (D.F.loc[D.ids, 'ptr_n'].fillna(0) > 0).values
B0C = [c for c in BASE_COLS if c not in POOL_COLS] + [c for c in BASE_COLS if c in POOL_COLS]  # same order as Harness.run
EXPERTS = {'B0': (B0C, ('pool',)),
           'mouse': (CH['mouse'], ()),
           'behavior': (CH['behavior'], ()),
           'item': ([c for c in CH['item'] if c not in POOL_COLS] + [c for c in CH['item'] if c in POOL_COLS], ('pool',)),
           'context': (CH['context'], ())}
for sch in list(H.schemes):
    t0 = time.time()
    out = {e: np.full(len(D.y), np.nan) for e in EXPERTS}
    for k, (tri, vai) in enumerate(H.schemes[sch]):
        for e, (cols, fams) in EXPERTS.items():
            tr, va = tri, vai
            if e == 'mouse':
                tr, va = tri[mouse_ok[tri]], vai[mouse_ok[vai]]
            Xtr, Xva = H.X(cols, fams, sch, k, tr), H.X(cols, fams, sch, k, va)
            m = make_model('cat', None, 0).fit(Xtr, D.y[tr])
            out[e][va] = m.predict_proba(Xva)[:, 1]
    np.savez(R4 / 'oof' / f'experts_{sch}.npz', **out)
    print(sch, 'done', round(time.time() - t0), 's', flush=True)

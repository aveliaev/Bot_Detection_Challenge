"""B0 + auxiliary family on the exact B0 folds (P0..P4 + fwd7), same CatBoost, no tuning.
Usage: python research6/r6_integrate.py NAME file1.parquet[:col1,col2] [file2.parquet ...]
Label-free families (self-supervised, leave-one-day-out) are identical for every fold; label-dependent families must
already be strict OOF per fold (they are passed as per-scheme tables by their own scripts)."""
import sys, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
R6 = Path(__file__).resolve().parent; ROOT = R6.parent
sys.path.insert(0, str(ROOT / 'research2')); sys.path.insert(0, str(ROOT / 'research4'))
from rcv import *
import rcv
rcv.OOF2 = R6 / 'oof'; rcv.LOG = R6 / 'experiments6.jsonl'
from common4 import op_point, PFP, IDS as I4, Y as Y4


def load_family(specs):
    parts = []
    for sp in specs:
        f, _, cols = sp.partition(':')
        t = pd.read_parquet(R6 / 'cache' / f)
        parts.append(t[cols.split(',')] if cols else t)
    F = pd.concat(parts, axis=1)
    F.columns = ['aux_' + c for c in F.columns]
    return F


def report(name):
    y = Y4
    rep = lambda n, d: np.mean([np.load(f'{d}/{n}__P{i}.npy') for i in range(5)], axis=0)
    b0 = rep('B0_cat_top120', R6 / 'oof'); s = rep(name, R6 / 'oof'); mk = b0 > 0
    A, B = op_point(y[mk], b0[mk], I4[mk]), op_point(y[mk], s[mk], I4[mk])
    pos, neg = set(I4[y == 1]), set(I4[y == 0])
    print(f'{name}: avg-OOF P@R {A["P"]:.4f} -> {B["P"]:.4f} | TP {B["TP"]} FP {B["FP"]} FN {B["FN"]} | '
          f'FP removed {len((A["flagged"] - B["flagged"]) & neg)} added {len((B["flagged"] - A["flagged"]) & neg)} | '
          f'TP removed {len((A["flagged"] - B["flagged"]) & pos)} added {len((B["flagged"] - A["flagged"]) & pos)} | '
          f'persistent FP still flagged {len(B["flagged"] & PFP)}/45')
    print('paired bootstrap vs B0:', {k: (round(v, 4) if isinstance(v, float) else v) for k, v in paired_bootstrap(y, np.where(mk, s, 0), np.where(mk, b0, 0), n=1000).items()})
    T = pd.read_parquet(ROOT / 'research3' / 'cache' / 'traits.parquet'); F = pd.read_parquet(ROOT / 'work' / 'features.parquet')
    pl = T.pl.reindex(I4).values; mo = (F.ptr_n.reindex(I4).fillna(0) > 0).values
    grp = np.where(pl == 'web', np.where(mo, 'web+mouse', 'web no mouse'), pl)
    print('platform P@R (inside segment):', {g: (round(precision_at_recall(y[mk & (grp == g)], b0[mk & (grp == g)]), 4),
                                               round(precision_at_recall(y[mk & (grp == g)], s[mk & (grp == g)]), 4)) for g in ['web+mouse', 'web no mouse', 'android', 'ios']})


if __name__ == '__main__':
    name, specs = sys.argv[1], sys.argv[2:]
    AUX = load_family(specs)
    H = Harness()
    H.add_family('aux', lambda D, lab, ids: AUX.reindex(pd.Index(ids, name='cookie_id')))
    H.run(name, fams=('pool', 'aux'), hypothesis=f'C6 B0 + {specs}', leak_risk='label-free LODO aux')
    report(name)

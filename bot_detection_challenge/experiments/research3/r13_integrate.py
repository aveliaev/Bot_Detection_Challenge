"""C3.13 integration candidates, each = B0 + one new family, repeated day-block CV (5 partitions) + fwd7."""
import sys, warnings; sys.path.insert(0, 'research3'); sys.path.insert(0, 'research2'); warnings.filterwarnings('ignore')
from rcv import *
import rcv
rcv.OOF2 = Path(__file__).resolve().parent / 'oof'; rcv.LOG = Path(__file__).resolve().parent / 'experiments3.jsonl'
from llr import llr_features
from manifold import manifold_features
from uate import ua_te_features
which = sys.argv[1:]
H = Harness()
AG = ['mean', 'q75', 'sum', 'fhi']
def llr_fam(neg):
    def f(D, lab, ids):
        F = llr_features(D, lab, ids, fams=['tdt', 'pg', 'm3dt'], neg=neg, prefix='l' + neg[0])
        return F[[c for c in F.columns if c.split('_')[-1] in AG]]
    return f
H.add_family('llrhard', llr_fam('hard'))
H.add_family('llrall', llr_fam('all'))
H.add_family('llrtrait', llr_fam('trait'))
H.add_family('manifold', lambda D, lab, ids: manifold_features(D, lab, ids, which=('knn', 'gmm', 'maha', 'proto')))
H.add_family('uate', lambda D, lab, ids: ua_te_features(D, lab, ids))
EXPS = {'E5_llrhard': ('llrhard', 'C3.5 hard-negative event LLR (tdt,pg,m3dt)'),
        'E5_llrtrait': ('llrtrait', 'C3.5 LLR vs label-free hard humans (>=2 bot traits); fixes 2nd-order leak of E5_llrhard'),
        'E5_llrall': ('llrall', 'C3.5 control: same LLR vs all humans'),
        'E4_manifold': ('manifold', 'C3.4 positive-manifold/prototype scores'),
        'E12_uate': ('uate', 'C3.12 UA token temporal TE (device fingerprint)')}
for name in which:
    fam, hyp = EXPS[name]
    H.run(name, fams=('pool', fam), hypothesis=hyp, leak_risk='medium (fold-aware, inner LODO for train rows)')

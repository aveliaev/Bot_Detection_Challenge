"""Per-platform P@R and the trade made by the best fusions (which persistent FP go down, which TPs go down)."""
import sys, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common4 import *
Z = np.load(R4 / 'oof' / 'fusion_scores.npz')
T = pd.read_parquet(ROOT / 'research3' / 'cache' / 'traits.parquet'); pl = T.pl.reindex(IDS).values
def avg(rule):
    return np.mean([np.where(np.isnan(Z[f'{rule}__{s}']), 0, rankpct(Z[f'{rule}__{s}'], ~np.isnan(Z[f'{rule}__{s}']))) for s in SCHEMES[:5]], axis=0)
b0 = avg('B0'); mk = b0 > 0
base = op_point(Y[mk], b0[mk], IDS[mk])
rows = []
for r in ['B0', 'meta_lr', 'meta_cat2', 'meta_cat3', 'iliou_paper_both[prob]', 'iliou_fit_both', 'mean[prob]', 'count>0.9[prob]']:
    s = avg(r); op = op_point(Y[mk], s[mk], IDS[mk])
    d = dict(rule=r, PR=op['P'], TP=op['TP'], FP=op['FP'])
    for p in ['web', 'android', 'ios']:
        k = mk & (pl == p); d[p] = pr(Y[k], s[k])
    lostTP = len((base['flagged'] - op['flagged']) & set(IDS[Y == 1])); gainTP = len((op['flagged'] - base['flagged']) & set(IDS[Y == 1]))
    remFP = len((base['flagged'] - op['flagged']) & set(IDS[Y == 0])); newFP = len((op['flagged'] - base['flagged']) & set(IDS[Y == 0]))
    d.update(pFP_removed=len(PFP - op['flagged']), FP_removed_vs_B0=remFP, FP_added_vs_B0=newFP, TP_lost_vs_B0=lostTP, TP_gained_vs_B0=gainTP)
    rows.append(d)
pd.set_option('display.width', 250)
print(pd.DataFrame(rows).round(4).to_string(index=False))

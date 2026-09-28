"""Cycle 5 summary: nested local reranking vs frozen B0 on P0..P4 + fwd7; boundary crossings; persistent FP;
fixed-config grid (transparency only, not used for any selection)."""
import sys, pickle, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from local import *
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
res = pickle.load(open(R5 / 'cache_r5.pkl', 'rb'))


def base_scores(sch):
    E = load(sch); mask = valid_mask(E)
    return mask, rankpct(E['B0'], mask)


def assemble(sch, key=None):
    """pooled score of scheme: key=None -> nested choice per fold, else a fixed (cfg, strength)."""
    mask, b0 = base_scores(sch)
    s = np.full(len(Y), np.nan)
    for r in res:
        if r['sch'] != sch: continue
        k = r['chosen'] if key is None else key
        s[r['ap_idx']] = r['fixed'][k]
    # fwd7: rows not covered by k=5/7 do not exist (mask covers days 5..13 only)
    assert np.isnan(s[mask]).sum() == 0
    return mask, b0, s


def crossing(mask, b0, s):
    ids, y = IDS[mask], Y[mask]
    A, B = op_point(y, b0[mask], ids), op_point(y, s[mask], ids)
    pos = set(ids[y == 1]); neg = set(ids[y == 0])
    fa, fb = A['flagged'], B['flagged']
    return dict(PR_B0=A['P'], PR_new=B['P'], TP=B['TP'], FP=B['FP'], FN=B['FN'],
                FP_removed=len((fa - fb) & neg), FP_added=len((fb - fa) & neg), TP_removed=len((fa - fb) & pos), TP_added=len((fb - fa) & pos),
                net_FP=B['FP'] - A['FP'], net_TP=B['TP'] - A['TP'],
                pFP_flagged_B0=len(fa & PFP), pFP_flagged_new=len(fb & PFP))


rows = []
for sch in SCHEMES:
    mask, b0, s = assemble(sch)
    d = crossing(mask, b0, s); d['scheme'] = sch; rows.append(d)
T = pd.DataFrame(rows).set_index('scheme')
rep = T.loc[SCHEMES[:5]]
print('== NESTED local reranking vs frozen B0 (per outer split)')
print(T.round(4).to_string())
print(f"\nrepeated CV: B0 {rep.PR_B0.mean():.4f} -> nested {rep.PR_new.mean():.4f} (delta {rep.PR_new.mean() - rep.PR_B0.mean():+.4f}); "
      f"fwd7: {T.loc['fwd7', 'PR_B0']:.4f} -> {T.loc['fwd7', 'PR_new']:.4f} (delta {T.loc['fwd7', 'PR_new'] - T.loc['fwd7', 'PR_B0']:+.4f}); "
      f"splits improved {(rep.PR_new > rep.PR_B0 + 1e-12).sum()}/5, worse {(rep.PR_new < rep.PR_B0 - 1e-12).sum()}/5")
T.to_csv(R5 / 'nested_results.csv')

print('\n== configs chosen by inner CV (per outer fold)')
ch = pd.DataFrame([dict(scheme=r['sch'], fold=r['k'], chosen=str(r['chosen'][0]), strength=r['chosen'][1],
                        inner_base=round(r['inner_base'], 4), inner_best=round(r['inner_best'], 4)) for r in res])
print(ch.to_string(index=False))
print('\nmethod chosen (count of outer folds):', ch.chosen.str.extract(r"\('(\w+)")[0].value_counts().to_dict())
ch.to_csv(R5 / 'nested_choices.csv', index=False)

# averaged OOF over the 5 partitions -> persistent FP accounting of the nested procedure
avg_b0 = np.mean([base_scores(s)[1] for s in SCHEMES[:5]], axis=0)
avg_new = np.mean([assemble(s)[2] for s in SCHEMES[:5]], axis=0)
mk = ~np.isnan(avg_b0)
print('\n== averaged repeated OOF (days>=2), nested procedure')
print(crossing(mk, avg_b0, avg_new))

# fixed configs: transparency only (post-hoc view of the whole grid; not used to pick anything)
keys = list(res[0]['fixed'].keys())
fx = []
for key in keys:
    v = {sch: crossing(*assemble(sch, key))['PR_new'] for sch in SCHEMES}
    r = np.array([v[s] for s in SCHEMES[:5]]); b = np.array([T.loc[s, 'PR_B0'] for s in SCHEMES[:5]])
    fx.append(dict(method=key[0][0], cfg=str(key[0][1:]), strength=key[1], rep=r.mean(), fwd7=v['fwd7'],
                   d_rep=r.mean() - b.mean(), d_fwd7=v['fwd7'] - T.loc['fwd7', 'PR_B0'], splits_up=int((r > b + 1e-12).sum())))
FX = pd.DataFrame(fx)
FX.to_csv(R5 / 'fixed_grid.csv', index=False)
print(f'\n== fixed-config grid ({len(FX)} configs; post-hoc, NOT used for selection)')
print('configs with d_rep >= +0.01 AND d_fwd7 >= +0.01 AND >=4/5 splits up:', int(((FX.d_rep >= .01) & (FX.d_fwd7 >= .01) & (FX.splits_up >= 4)).sum()))
print('configs with rep >= 0.910 AND fwd7 >= 0.910:', int(((FX.rep >= .91) & (FX.fwd7 >= .91)).sum()))
print('distribution of d_rep:', FX.d_rep.describe().round(4).to_dict())
print('distribution of d_fwd7:', FX.d_fwd7.describe().round(4).to_dict())
print('\nbest 12 by d_rep (optimistic, selected on outer folds!):')
print(FX.sort_values('d_rep', ascending=False).head(12).round(4).to_string(index=False))
print('\nbest per method by d_rep (optimistic):')
print(FX.sort_values('d_rep', ascending=False).groupby('method').head(1).round(4).to_string(index=False))

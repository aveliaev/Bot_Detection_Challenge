"""C4 evaluation of experts and fusion rules on exactly the B0 rows/folds: P0..P4 (repeated day-block) + fwd7.
Outputs research4/fusion_results.csv and research4/oof/fusion_<rule>.npz (scores per scheme)."""
import sys, warnings, json; warnings.filterwarnings('ignore')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fusion import *
from sklearn.metrics import roc_auc_score

FITTED = ['wmean_rank', 'iliou_fit_beh', 'iliou_fit_both', 'meta_lr', 'meta_lr_noB0', 'meta_cat2', 'meta_cat3']


def fitted_scores(kind, Str_r, Str_p, B0tr, ytr, Sva_r, Sva_p, B0va):
    if kind == 'wmean_rank':
        return wmean(Sva_r, best_weighted(Str_r, ytr)), None
    if kind.startswith('iliou_fit'):
        web = 'behavior' if kind.endswith('beh') else 'both'
        p = best_iliou(Str_p, ytr, web); return iliou(Sva_p, *p, web), p
    Xtr, Xva = meta_X(Str_p, B0tr), meta_X(Sva_p, B0va)
    return fit_predict(kind, Xtr, ytr, Xva), None


def sub(S, idx):
    return {k: v[idx] for k, v in S.items()}


def evaluate():
    res, scores, params = {}, {}, {}
    for sch in SCHEMES:
        E = load(sch); mask = valid_mask(E)
        Sp, Sr = scaled(E, mask, 'prob'), scaled(E, mask, 'rank')
        B0r = rankpct(E['B0'], mask)
        R = {'B0': B0r}
        for e in ['mouse', 'behavior', 'item', 'context']:           # single experts, missing mouse -> bottom
            R[f'expert_{e}'] = np.where(mask, np.nan_to_num(Sr[e], nan=-1.0), np.nan)
        for scale, S in (('prob', Sp), ('rank', Sr)):
            for k, v in rules_free(S, B0r).items():
                R[f'{k}[{scale}]'] = v
        R['iliou_paper_beh[prob]'] = iliou(Sp, 0.7, 0.3, 'behavior')
        R['iliou_paper_both[prob]'] = iliou(Sp, 0.7, 0.3, 'both')
        # fitted, cross-fitted
        for kind in FITTED:
            out = np.full(len(Y), np.nan)
            if sch.startswith('P'):
                for k, (tri, vai) in enumerate(folds_of(sch)):
                    trm = mask.copy(); trm[vai] = False
                    tr = np.flatnonzero(trm); va = vai
                    s, p = fitted_scores(kind, sub(Sr, tr), sub(Sp, tr), E['B0'][tr], Y[tr], sub(Sr, va), sub(Sp, va), E['B0'][va])
                    out[va] = s; params.setdefault((kind, sch), []).append(p)
            else:  # fwd7: nested inner OOF from training days only
                for k in (5, 7):
                    I = dict(np.load(R4 / 'oof' / f'experts_fwd7inner_k{k}.npz'))
                    im = ~np.isnan(I['B0']); tr = np.flatnonzero(im)
                    Ir, Ip = scaled(I, im, 'rank'), scaled(I, im, 'prob')
                    va = np.flatnonzero(mask & ((DAY >= k) & (DAY < k + 2) if k == 5 else (DAY >= 7)))
                    s, p = fitted_scores(kind, sub(Ir, tr), sub(Ip, tr), I['B0'][tr], Y[tr], sub(Sr, va), sub(Sp, va), E['B0'][va])
                    out[va] = s; params.setdefault((kind, sch), []).append(p)
            R[kind] = out
        for k, v in R.items():
            m = mask & ~np.isnan(v)
            assert m.sum() == mask.sum(), (sch, k)
            res.setdefault(k, {})[sch] = pr(Y[mask], v[mask])
            scores.setdefault(k, {})[sch] = v
    return res, scores, params


def summarize(res, scores):
    b0 = np.mean([np.where(np.isnan(scores['B0'][s]), 0, rankpct(scores['B0'][s], ~np.isnan(scores['B0'][s]))) for s in SCHEMES[:5]], axis=0)
    mk = b0 > 0
    base = op_point(Y[mk], b0[mk], IDS[mk])
    rows = []
    for k in res:
        rep = [res[k][s] for s in SCHEMES[:5]]
        avg = np.mean([np.where(np.isnan(scores[k][s]), 0, rankpct(scores[k][s], ~np.isnan(scores[k][s]))) for s in SCHEMES[:5]], axis=0)
        op = op_point(Y[mk], avg[mk], IDS[mk])
        bs = paired_bootstrap(Y, np.where(mk, avg, 0), np.where(mk, b0, 0), n=500) if k != 'B0' else {}
        rows.append(dict(rule=k, rep=np.mean(rep), rep_std=np.std(rep), rep_min=np.min(rep), fwd7=res[k]['fwd7'],
                         avgOOF_PR=op['P'], TP=op['TP'], FP=op['FP'], FN=op['FN'], pFP_flagged=len(op['flagged'] & PFP),
                         pFP_removed=45 - len(op['flagged'] & PFP), p_better=bs.get('p_better', np.nan),
                         **{s: res[k][s] for s in SCHEMES[:5]}))
    T = pd.DataFrame(rows).sort_values('rep', ascending=False)
    return T


if __name__ == '__main__':
    res, scores, params = evaluate()
    T = summarize(res, scores)
    pd.set_option('display.width', 250); pd.set_option('display.max_rows', 100)
    print(T.drop(columns=SCHEMES[:5]).round(4).to_string(index=False))
    T.to_csv(R4 / 'fusion_results.csv', index=False)
    np.savez(R4 / 'oof' / 'fusion_scores.npz', **{f'{k}__{s}': v for k, d in scores.items() for s, v in d.items()})
    print('\nfitted parameters (per fold):')
    for (k, s), p in params.items():
        if p[0] is not None: print(' ', k, s, p)

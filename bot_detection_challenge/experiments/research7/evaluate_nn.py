"""C7.12-C7.14 evaluation of the neural sequence models: standalone performance, LSTM vs BiLSTM, window AUC by length,
chunk-distribution diagnostics (persistent FP vs matched TP, conditional band, consistency at same mean), group table,
ablations and order / pointer shuffle controls (P0 + fwd7)."""
import sys, subprocess, shutil, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
R7 = Path(__file__).resolve().parent; ROOT = R7.parent
sys.path.insert(0, str(R7)); sys.path.insert(0, str(ROOT / 'research4'))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from common4 import IDS, Y, PFP, pr
from build_seq_features import aggregate, geom
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
SCH = ['P0', 'P1', 'P2', 'P3', 'P4', 'fwd7']; K = {s: range(6) for s in SCH[:5]}; K['fwd7'] = range(2)
lab = pd.Series(Y, index=IDS)
b0 = {s: pd.Series(np.load(ROOT / 'research2' / 'oof' / f'B0_cat_top120__{s}.npy'), index=IDS) for s in SCH}


def nnfeat(name, s, col='score'):
    t = pd.concat([pd.read_parquet(R7 / 'oof' / f'nn_{name}_{s}_k{k}.parquet') for k in K[s] if (R7 / 'oof' / f'nn_{name}_{s}_k{k}.parquet').exists()])
    t = t[~t.index.duplicated(keep='last')]
    F = aggregate(t[col]); return F[~F.index.duplicated(keep='last')], t


def standalone(name, schemes=SCH):
    rows = []
    for s in schemes:
        F, _ = nnfeat(name, s); y = lab[F.index]
        rows.append(dict(model=name, scheme=s, PR_seq_full=pr(y.values, F.seq_full.values), PR_chunk_mean=pr(y.values, F.chunk_mean.fillna(F.seq_full).values),
                         AUC_seq_full=roc_auc_score(y, F.seq_full), spearman_full_B0=F.seq_full.corr(b0[s][F.index], method='spearman'),
                         PR_B0=pr(y.values, b0[s][F.index].values)))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    S = pd.concat([standalone('lstm'), standalone('bilstm')])
    print('== standalone sequence models (cookie level)'); print(S.round(4).to_string(index=False))
    rep = S[S.scheme != 'fwd7'].groupby('model')[['PR_seq_full', 'PR_chunk_mean', 'AUC_seq_full', 'spearman_full_B0']].agg(['mean', 'std']).round(4)
    print('\nrep mean/std:'); print(rep); print('fwd7:'); print(S[S.scheme == 'fwd7'].set_index('model')[['PR_seq_full', 'PR_chunk_mean']].round(4))
    d_rep = rep.loc['bilstm', ('PR_seq_full', 'mean')] - rep.loc['lstm', ('PR_seq_full', 'mean')]
    d_f = S[(S.scheme == 'fwd7') & (S.model == 'bilstm')].PR_seq_full.item() - S[(S.scheme == 'fwd7') & (S.model == 'lstm')].PR_seq_full.item()
    best = 'bilstm' if (d_rep > 0.01 and d_f > 0.01) else 'lstm'
    print(f'\nLSTM vs BiLSTM: d_rep={d_rep:+.4f}, d_fwd7={d_f:+.4f} -> chosen (locked rule: simpler unless both > +0.01): {best}')
    open(R7 / 'cache' / 'chosen_arch.txt', 'w').write(best)
    # window AUC by length (P0)
    G = geom(); wa = []
    for name in ('lstm', 'bilstm'):
        _, t = nnfeat(name, 'P0'); g = G.loc[t.index]; yy = lab[g.cookie_id].values
        for L in (3, 5, 10, 20, 40, -1):
            m = (g.L == L).values; wa.append(dict(model=name, L='full' if L == -1 else L, AUC=roc_auc_score(yy[m], t.score.values[m])))
    print('\n== window AUC by length (P0)'); print(pd.DataFrame(wa).pivot(index='L', columns='model', values='AUC').round(4))
    # diagnostics on averaged repeated OOF of the chosen arch
    Fs = [nnfeat(best, s)[0] for s in SCH[:5]]
    A = sum(f.reindex(Fs[0].index) for f in Fs) / 5; A.index.name = 'cookie_id'
    A.to_parquet(R7 / 'cache' / f'seq_repavg_{best}.parquet')
    subprocess.run([sys.executable, str(ROOT / 'research6' / 'evaluate_auxiliary.py'), str(R7 / 'cache' / f'seq_repavg_{best}.parquet'), f'nn7'],
                   stdout=open(R7 / 'gate_nn7.log', 'w'), stderr=subprocess.DEVNULL, check=True)
    for f in ('matched', 'conditional'):
        shutil.move(str(ROOT / 'research6' / f'gate_nn7_{f}.csv'), str(R7 / f'gate_nn7_{f}.csv'))
    M = pd.read_csv(R7 / 'gate_nn7_matched.csv'); C = pd.read_csv(R7 / 'gate_nn7_conditional.csv')
    M['best'] = np.maximum(M.AUC_TP_vs_pFP, 1 - M.AUC_TP_vs_pFP); C['band_best'] = np.maximum(C.band_AUC_len_strat, 1 - C.band_AUC_len_strat)
    print(f'\n== persistent FP vs matched TP ({best}), length-matched, top 12')
    print(M[M.match.str.contains('n_events')].sort_values('best', ascending=False).head(12).round(3).to_string(index=False))
    print('\n== conditional band 300-900 (top 10)')
    print(C.sort_values('band_best', ascending=False).head(10)[['feature', 'AUC_target', 'band_AUC_len_strat', 'band_AUC_web+mouse', 'band_AUC_web no mouse',
                                                                'band_AUC_android', 'rho_B0', 'rho_n']].round(3).to_string(index=False))
    o = pd.read_csv(ROOT / 'research2' / 'oof_final_B0.csv').set_index('cookie_id'); o = o[o.oof_repeated_dayblock > 0]
    rk = o.oof_repeated_dayblock.rank(ascending=False)
    grp = pd.Series('easy neg', index=o.index)
    grp[(o.target == 0) & rk.between(605, 1500)] = 'hard neg'; grp[(o.target == 0) & (rk <= 604)] = 'other FP'
    grp[o.index.isin(PFP)] = 'persistent FP'; grp[(o.target == 1) & (rk <= 604)] = 'TP'; grp[(o.target == 1) & (rk > 604)] = 'FN'
    T = A.join(o.oof_repeated_dayblock.rename('B0')).join(grp.rename('grp'))
    print('\n== medians by group'); print(T.groupby('grp')[['B0', 'seq_full', 'chunk_mean', 'chunk_std', 'chunk_q10', 'chunk_q90', 'chunk_min', 'chunk_max']].median().round(3).to_string())
    F7 = pd.read_parquet(ROOT / 'work' / 'features.parquet'); Tr = pd.read_parquet(ROOT / 'research3' / 'cache' / 'traits.parquet')
    T['g2'] = np.where(Tr.pl.reindex(T.index) == 'web', np.where(F7.ptr_n.reindex(T.index).fillna(0) > 0, 'wm', 'wn'), Tr.pl.reindex(T.index)); T['rank'] = rk
    fp = T[T.grp == 'persistent FP']; tp = T[o.target.reindex(T.index) == 1]; pairs = []
    for c, r in fp.iterrows():
        cand = tp[tp.g2 == r.g2]; d = (cand['rank'] - r['rank']).abs() / 100 + (cand.chunk_mean - r.chunk_mean).abs() * 20
        pairs += [(c, j) for j in d.nsmallest(3).index]
    print('\n== consistency at the same mean (B0 rank + chunk_mean + platform x mouse)')
    for f in ('chunk_mean', 'chunk_std', 'chunk_q10', 'chunk_min', 'chunk_range', 'len5_std', 'position_std_L5'):
        x = pd.concat([T.loc[[a for a, _ in pairs], f], T.loc[[b for _, b in pairs], f]]).values; y = np.r_[np.zeros(len(pairs)), np.ones(len(pairs))]; m = ~np.isnan(x)
        print(f'  {f:16s} AUC={roc_auc_score(y[m], x[m]):.3f}  median pFP={np.nanmedian(x[:len(pairs)]):.3f}  median TP={np.nanmedian(x[len(pairs):]):.3f}')
    # ablations + controls (P0, fwd7)
    print('\n== ablations and controls (cookie-level P@R of seq_full / chunk_mean; window AUC L=5)')
    rows = []
    for name in ['abl_full', 'abl_event', 'abl_event_timing', 'abl_event_timing_ptr', 'abl_pointer_only']:
        for s in ('P0', 'fwd7'):
            if not (R7 / 'oof' / f'nn_{name}_{s}_k0.parquet').exists(): continue
            F, t = nnfeat(name, s); y = lab[F.index]; g = G.loc[t.index]; yy = lab[g.cookie_id].values; m5 = (g.L == 5).values
            r = dict(model=name, scheme=s, n_cookies=len(F), PR_full=pr(y.values, F.seq_full.values), AUC_full=roc_auc_score(y, F.seq_full),
                     win5_AUC=roc_auc_score(yy[m5], t.score.values[m5]))
            for cn in ('order_shuffled', 'pointer_shuffled'):
                Fc = aggregate(t[cn]); Fc = Fc[~Fc.index.duplicated(keep='last')]
                r[f'PR_full_{cn}'] = pr(lab[Fc.index].values, Fc.seq_full.values); r[f'win5_AUC_{cn}'] = roc_auc_score(yy[m5], t[cn].values[m5])
                r[f'corr_{cn}'] = np.corrcoef(t.score, t[cn])[0, 1]
            rows.append(r)
    print(pd.DataFrame(rows).round(4).to_string(index=False))

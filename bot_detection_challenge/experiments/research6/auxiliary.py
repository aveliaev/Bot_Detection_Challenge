"""C6.2-C6.5 self-supervised auxiliary tasks (no labels anywhere in this file).

Provenance (identical for every task):
  training rows : transitions / prefixes of LABELLED-TRAIN cookies of all train days EXCEPT day d
  fitting       : LightGBM, fixed seed; quantile bins of Δt from the training rows only
  prediction    : cookies of day d  (days 0..13);  test cookies (days 14..20) by a model fitted on all 14 train days
  => no cookie is ever scored by a model fitted on its own events; labels are never read.
Tasks: A next event type (multiclass), B next Δt (log1p regression + 8-bin quantile classifier),
       C remaining event count after a prefix, D new items in the suffix after a prefix (log1p regressions).
Cookie-level OOF outputs (research6/cache/aux_cookie.parquet): surprise / error aggregates.
Also writes the task-quality table incl. trivial baselines and negative controls (research6/aux_quality.csv)."""
from __future__ import annotations
import sys, time, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
R6 = Path(__file__).resolve().parent
sys.path.insert(0, str(R6))
import numpy as np, pandas as pd
import lightgbm as lgb
from make_prefix_dataset import base_events, transitions, prefixes, FEAT_T, CAT_T, FEAT_P, R6

SEED = 0
NB = 8
TRAIN_DAYS = list(range(14))
CTRL_DAYS = [3, 8, 12]


def lgbm(kind, **kw):
    p = dict(n_estimators=300, learning_rate=0.08, num_leaves=63, min_child_samples=50, subsample=0.8, subsample_freq=1,
             colsample_bytree=0.8, verbose=-1, n_jobs=4, random_state=SEED)
    p.update(kw)
    return (lgb.LGBMClassifier if kind == 'clf' else lgb.LGBMRegressor)(**p)


def nll(P, y):
    return -np.log(np.clip(P[np.arange(len(y)), y], 1e-6, 1))


def markov_tables(Ttr):
    g = np.bincount(Ttr.next_ev, minlength=9) + 1.0
    m1 = pd.crosstab(Ttr.cur, Ttr.next_ev).reindex(index=range(9), columns=range(9), fill_value=0) + 1.0
    k2 = Ttr.prev1.astype(str) + '_' + Ttr.cur.astype(str)
    m2 = pd.crosstab(k2, Ttr.next_ev).reindex(columns=range(9), fill_value=0)
    return g / g.sum(), (m1.div(m1.sum(1), axis=0)).values, m2


def markov_pred(tabs, T):
    g, m1, m2 = tabs
    P1 = m1[T.cur.values]
    k2 = T.prev1.astype(str) + '_' + T.cur.astype(str)
    C2 = m2.reindex(k2).fillna(0).values
    P2 = (C2 + 5 * P1) / (C2.sum(1, keepdims=True) + 5)      # back-off to order 1
    return np.tile(g, (len(T), 1)), P1, P2


def shuffled(ev, what, seed):
    """negative controls: permute event types (keep timestamps) or permute Δt (keep event order) within each cookie."""
    rng = np.random.RandomState(seed)
    e = ev.copy()
    if what == 'events':
        idx = e.groupby('cookie_id', sort=False).apply(lambda d: pd.Series(rng.permutation(d.index.values.copy()))).values
        for col in ['event_name', 'e', 'item_id', 'item_category', 'item_location']:
            e[col] = ev.loc[idx, col].values
        e['e'] = e['e'].astype(int)
    else:
        dt = e.groupby('cookie_id').t.diff()
        d2 = dt.groupby(e.cookie_id).transform(lambda s: pd.Series(np.r_[s.values[:1], rng.permutation(s.values[1:].copy())], index=s.index))
        t0 = e.groupby('cookie_id').t.transform('min')
        e['t'] = t0 + d2.fillna(0).groupby(e.cookie_id).cumsum()
    return e


def run():
    t0 = time.time()
    ev = base_events()
    lab = set(pd.read_csv(R6.parent / 'data' / 'train.csv').cookie_id)
    T = transitions(ev); P = prefixes(ev)
    T['is_tr'] = T.cookie_id.isin(lab); P['is_tr'] = P.cookie_id.isin(lab)
    ctrl = {w: transitions(shuffled(ev[ev.day.isin(CTRL_DAYS)], w, 1)) for w in ('events', 'dt')}
    Tout = {k: np.full(len(T), np.nan) for k in ['ev_nll', 'ev_top1', 'dt_nll', 'dt_abs']}
    Pout = {k: np.full(len(P), np.nan) for k in ['cnt_pred', 'div_pred']}
    Q = []
    for d in TRAIN_DAYS + ['test']:
        trm = T.is_tr & (T.day != d) if d != 'test' else T.is_tr
        vam = (T.day == d) if d != 'test' else ~T.is_tr
        Ttr = T[trm].sample(n=min(120000, int(trm.sum())), random_state=SEED)
        Tva = T[vam]
        # --- A next event
        A = lgbm('clf').fit(Ttr[FEAT_T], Ttr.next_ev, categorical_feature=CAT_T)
        PA = A.predict_proba(Tva[FEAT_T])
        Tout['ev_nll'][vam.values] = nll(PA, Tva.next_ev.values)
        Tout['ev_top1'][vam.values] = (PA.argmax(1) == Tva.next_ev.values)
        # --- B next Δt: quantile bins from TRAINING rows only
        edges = np.unique(np.quantile(Ttr.next_ldt, np.linspace(0, 1, NB + 1)[1:-1]))
        btr, bva = np.digitize(Ttr.next_ldt, edges), np.digitize(Tva.next_ldt, edges)
        Bc = lgbm('clf').fit(Ttr[FEAT_T], btr, categorical_feature=CAT_T)
        PB = Bc.predict_proba(Tva[FEAT_T])
        Tout['dt_nll'][vam.values] = nll(PB, bva)
        Br = lgbm('reg').fit(Ttr[FEAT_T], Ttr.next_ldt, categorical_feature=CAT_T)
        Tout['dt_abs'][vam.values] = np.abs(Br.predict(Tva[FEAT_T]) - Tva.next_ldt.values)
        # --- C/D prefixes
        pm = P.is_tr & (P.day != d) if d != 'test' else P.is_tr
        pv = (P.day == d) if d != 'test' else ~P.is_tr
        C = lgbm('reg', n_estimators=200).fit(P[pm][FEAT_P], P[pm].rem_cnt)
        Dm = lgbm('reg', n_estimators=200).fit(P[pm][FEAT_P], P[pm].rem_new_items)
        Pout['cnt_pred'][pv.values] = C.predict(P[pv][FEAT_P]); Pout['div_pred'][pv.values] = Dm.predict(P[pv][FEAT_P])
        if d != 'test':
            # trivial baselines + negative controls, evaluated on the same held-out day
            g, P1, P2 = markov_pred(markov_tables(Ttr), Tva)
            y = Tva.next_ev.values
            med = np.median(Ttr.next_ldt); medc = Ttr.groupby('cur').next_ldt.median()
            bf = np.bincount(btr, minlength=len(edges) + 1) + 1.0; bf = bf / bf.sum()
            kb = lambda X: np.minimum(X.k, 40)
            cm = P[pm].groupby(kb(P[pm])).rem_cnt.mean(); dm = P[pm].groupby(kb(P[pm])).rem_new_items.mean()
            q = dict(day=d, n_trans=len(Tva), A_nll=nll(PA, y).mean(), A_acc=(PA.argmax(1) == y).mean(),
                     A_top3=np.mean([y[i] in np.argsort(-PA[i])[:3] for i in range(len(y))]),
                     A_nll_global=nll(g, y).mean(), A_nll_markov1=nll(P1, y).mean(), A_nll_markov2=nll(P2, y).mean(),
                     A_acc_markov1=(P1.argmax(1) == y).mean(),
                     B_nll=nll(PB, bva).mean(), B_nll_freq=-np.log(bf[bva]).mean(), B_acc=(PB.argmax(1) == bva).mean(),
                     B_mae=np.abs(Br.predict(Tva[FEAT_T]) - Tva.next_ldt).mean(), B_mae_median=np.abs(med - Tva.next_ldt).mean(),
                     B_mae_per_event=np.abs(Tva.cur.map(medc).fillna(med) - Tva.next_ldt).mean(),
                     C_mae=np.abs(Pout['cnt_pred'][pv.values] - P[pv].rem_cnt).mean(),
                     C_mae_by_k=np.abs(kb(P[pv]).map(cm).fillna(P[pm].rem_cnt.mean()) - P[pv].rem_cnt).mean(),
                     C_spearman=pd.Series(Pout['cnt_pred'][pv.values]).corr(P[pv].rem_cnt.reset_index(drop=True), method='spearman'),
                     D_mae=np.abs(Pout['div_pred'][pv.values] - P[pv].rem_new_items).mean(),
                     D_mae_by_k=np.abs(kb(P[pv]).map(dm).fillna(P[pm].rem_new_items.mean()) - P[pv].rem_new_items).mean(),
                     D_spearman=pd.Series(Pout['div_pred'][pv.values]).corr(P[pv].rem_new_items.reset_index(drop=True), method='spearman'))
            if d in CTRL_DAYS:
                for w, Tc in ctrl.items():
                    Tc = Tc[Tc.day == d]
                    if w == 'events':
                        q['A_nll_shuffled_events'] = nll(A.predict_proba(Tc[FEAT_T]), Tc.next_ev.values).mean()
                        q['B_mae_shuffled_events'] = np.abs(Br.predict(Tc[FEAT_T]) - Tc.next_ldt).mean()
                    else:
                        q['A_nll_shuffled_dt'] = nll(A.predict_proba(Tc[FEAT_T]), Tc.next_ev.values).mean()
                        q['B_mae_shuffled_dt'] = np.abs(Br.predict(Tc[FEAT_T]) - Tc.next_ldt).mean()
            Q.append(q)
        print('day', d, 'done', round(time.time() - t0), 's', flush=True)
    Qd = pd.DataFrame(Q); Qd.to_csv(R6 / 'aux_quality.csv', index=False)
    # cookie aggregates
    for k, v in Tout.items(): T[k] = v
    for k, v in Pout.items(): P[k] = v
    g = T.groupby('cookie_id')
    A = pd.DataFrame({'ev_nll_mean': g.ev_nll.mean(), 'ev_nll_std': g.ev_nll.std(), 'ev_nll_max': g.ev_nll.max(),
                      'ev_top1_acc': g.ev_top1.mean(),
                      'dt_nll_mean': g.dt_nll.mean(), 'dt_nll_std': g.dt_nll.std(), 'dt_nll_max': g.dt_nll.max(),
                      'dt_abs_mean': g.dt_abs.mean()})
    P['cnt_res'] = P.rem_cnt - P.cnt_pred; P['div_res'] = P.rem_new_items - P.div_pred
    gp = P.groupby('cookie_id')
    A = A.join(pd.DataFrame({'fut_cnt_res': gp.cnt_res.mean(), 'fut_cnt_abs': gp.cnt_res.apply(lambda s: s.abs().mean()),
                             'fut_div_res': gp.div_res.mean(), 'fut_div_abs': gp.div_res.apply(lambda s: s.abs().mean())}), how='left')
    A.index.name = 'cookie_id'
    A.to_parquet(R6 / 'cache' / 'aux_cookie.parquet')
    T[['cookie_id', 'day', 'pos', 'cur', 'next_ev', 'ev_nll', 'dt_nll', 'dt_abs']].to_parquet(R6 / 'cache' / 'aux_transitions.parquet')
    P.to_parquet(R6 / 'cache' / 'aux_prefixes.parquet')
    pd.set_option('display.width', 250)
    print(Qd.mean(numeric_only=True).round(4).to_string())
    print('cookie table', A.shape, 'total', round(time.time() - t0), 's')


if __name__ == '__main__':
    run()

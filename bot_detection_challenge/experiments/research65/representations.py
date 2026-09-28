"""Cycle 6.5 compact channel representations built ONLY from existing cycle-1/2 features (work/features.parquet) and the
fold-specific item pool (history.pool_features, labels of the fold's training part, strictly earlier days).
No target, no B0 score, no expert probabilities, no UA (UA TE showed train/test shift AUC 1.0 in cycle 3)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

MOUSE = ['ptr_x_std', 'ptr_y_std', 'ptr_box_frac', 'ptr_exact_edge_frac', 'ptr_dist_mean', 'ptr_dist_med',
         'ptr_x_center_dev', 'ptr_y_center_dev', 'ptr_x_gt1300_frac', 'ptr_frac_web', 'ptrxy_uniq_ratio', 'ptr_zero_move']
# behaviour target dimensions and their groups; sign convention: + = slower / longer pauses, + = more activity
BEH = {'b_ldt_median': 'timing', 'b_ldt_logmean': 'timing', 'b_ldt_logstd': 'timing', 'b_ldt_q90': 'timing',
       'b_band_30_90': 'timing', 'b_fast_2_20': 'timing',
       'b_log_n': 'activity', 'b_log_dur': 'activity', 'b_n_sess': 'activity', 'b_log_rate': 'activity',
       'b_frac_srv': 'eventmix', 'b_frac_iv': 'eventmix', 'b_frac_engage': 'eventmix'}
TIMING_SIGN = {'b_ldt_median': 1, 'b_ldt_logmean': 1, 'b_ldt_logstd': 0, 'b_ldt_q90': 1, 'b_band_30_90': 1, 'b_fast_2_20': -1}
ITEM = {'i_log_items': 'diversity', 'i_cat_n': 'diversity', 'i_loc_n': 'diversity', 'i_cat_ent': 'diversity', 'i_loc_ent': 'diversity',
        'i_repeat': 'navigation', 'i_loc_switch': 'navigation', 'i_cat_switch': 'navigation',
        'i_pop_logmean': 'pool', 'i_pool_ge1_frac': 'pool', 'i_pool_llr_mean': 'pool'}
CTX = ['c_web', 'c_android', 'c_ios', 'c_mouse', 'c_log_age']

_F = None


def base():
    global _F
    if _F is None:
        _F = pd.read_parquet(ROOT / 'work' / 'features.parquet')
        from history import popularity_features, cookie_items  # noqa
        ci = pd.read_parquet(ROOT / 'work' / 'cookie_items.parquet')
        _F = _F.join(popularity_features(ci))
    return _F


def build(pool: pd.DataFrame | None, ids) -> dict[str, pd.DataFrame]:
    """channel matrices for cookies `ids`; `pool` = fold-specific pool frame (index cookie_id)."""
    F = base().reindex(ids)
    C = pd.DataFrame(index=F.index)
    C['c_web'] = F.pl_web; C['c_android'] = F.pl_android; C['c_ios'] = F.pl_ios
    C['c_mouse'] = (F.ptr_n.fillna(0) > 0).astype(float)
    C['c_log_age'] = np.log1p(F.cookie_age_d.clip(lower=0))
    B = pd.DataFrame(index=F.index)
    B['b_ldt_median'] = np.log1p(F.sdt_median); B['b_ldt_logmean'] = F.sdt_log_mean; B['b_ldt_logstd'] = F.sdt_log_std
    B['b_ldt_q90'] = np.log1p(F.sdt_q90); B['b_band_30_90'] = F.sdt_band_30_90; B['b_fast_2_20'] = F.sdt_band_2_20
    B['b_log_n'] = F.log_n; B['b_log_dur'] = np.log1p(F.duration_s); B['b_n_sess'] = np.log1p(F.n_sessions)
    B['b_log_rate'] = np.log1p(F.ev_per_hour_active)
    B['b_frac_srv'] = F.frac_srv; B['b_frac_iv'] = F.frac_iv; B['b_frac_engage'] = F.frac_engage
    M = F[MOUSE].copy()
    M.loc[C.c_mouse == 0] = np.nan                      # mouse representation only where mouse is present
    I = pd.DataFrame(index=F.index)
    I['i_log_items'] = np.log1p(F.n_items.fillna(0)); I['i_cat_n'] = F.cat_nuniq; I['i_loc_n'] = F.loc_nuniq
    I['i_cat_ent'] = F.cat_ent; I['i_loc_ent'] = F.loc_ent
    I['i_repeat'] = F.item_repeat_ratio; I['i_loc_switch'] = F.loc_switch_frac; I['i_cat_switch'] = F.cat_switch_frac
    I['i_pop_logmean'] = F.pop_logmean
    P = pool.reindex(ids) if pool is not None else pd.DataFrame(index=F.index, columns=['pool_nb_ge1_frac', 'pool_llr_mean'])
    I['i_pool_ge1_frac'] = P['pool_nb_ge1_frac'].values; I['i_pool_llr_mean'] = P['pool_llr_mean'].values
    return {'mouse': M.astype(float), 'behavior': B.astype(float), 'item': I.astype(float), 'context': C.astype(float)}

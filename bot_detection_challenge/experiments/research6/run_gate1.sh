#!/bin/sh
cd "$(dirname "$0")/.."
python research6/r6_integrate.py E6_aux_all aux_cookie.parquet > research6/int_E6_aux_all.log 2>&1
python research6/r6_integrate.py E6_aux_time aux_cookie.parquet:dt_nll_mean,dt_nll_std,dt_nll_max,dt_abs_mean > research6/int_E6_aux_time.log 2>&1
python research6/r6_integrate.py E6_aux_event aux_cookie.parquet:ev_nll_mean,ev_nll_std,ev_nll_max,ev_top1_acc > research6/int_E6_aux_event.log 2>&1
python research6/r6_integrate.py E6_aux_future aux_cookie.parquet:fut_cnt_res,fut_cnt_abs,fut_div_res,fut_div_abs > research6/int_E6_aux_future.log 2>&1

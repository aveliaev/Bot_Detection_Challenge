# Cycle 7 — locked experimental plan (written after the data audit, BEFORE any model result)

Source of the ISIT ideas: the task text of cycle 7 (the ISIT 2024 "Bits and Bots" event report could not be read: no PDF
attached to the session, ieeexplore.ieee.org / kaggle.com / researchgate blocked by the environment's network policy).
Transferred ideas: Bombay — RF on local windows with pointer kinematics; Kanpur — BiLSTM over event tokens with
x, y, Δt, dx, dy, distance, velocity, acceleration, angle; Maryland — chunks + random-length sub-segments, class-weighted
objective; NYCU — confidence stability (only as chunk-score dispersion; calibration is irrelevant for ranking).

## Audit facts that constrain the plan (`audit_sequences.log`)
- pointer only on web (android/iOS 0 %); ONE pointer point per event; median gap between consecutive pointer events 43 s
  -> sparse click positions, not a trajectory. No separate mouse LSTM; kinematics are per-event features where valid.
- events per cookie: bots median 20, humans 12; >= 10 events: 80 % / 58 %; >= 40 events: 23 % / 9 %.

## Locked definitions
| item | value |
|---|---|
| splits | B0 folds exactly: P0..P4 (6 folds each) + fwd7 (k=5,7). Windows are built AFTER the split, per cookie; a cookie's windows never cross folds |
| window lengths | L in {3, 5, 10, 20, 40} + full; a length is used only if L <= n_events |
| training windows (cheap stage) | per training cookie 6 random contiguous windows: L uniform over the allowed set (incl. full), start uniform; seed 0; weight 1/K per window (every cookie total weight 1) |
| inference windows (all stages) | per allowed L: starts at 0 %, 25 %, 50 %, 75 % of (n-L) and end-aligned, duplicates removed (<= 5 per L) + the full window |
| valid pointer pair | consecutive events of the cookie, both with pointer, Δt <= 300 s; kinematics undefined otherwise (no forward fill) |
| window features (cheap) | length, log duration, 9 event-type shares, #types, mean/std/max log Δt inside the window, share Δt > 300 s, item/category/location change rates, within-window item revisit rate, new-item share, search-page mean, pointer availability share, #valid pairs, mean/std/max distance, mean/std velocity (px/s), mean abs acceleration, mean abs angle change, path length, net displacement, straightness (path / max(net, 1)), platform one-hot. Raw coordinates (no normalisation) |
| cheap models | RandomForest(300 trees, min_samples_leaf=5, max_features='sqrt', seed 0) and CatBoost(500 it, depth 6, lr 0.05, seed 0), both with the 1/K weights; no tuning |
| cookie aggregation | over the deterministic windows: mean, std, min, max, q10, q25, median, q75, q90, range, frac > .5/.7/.9; seq_full = score of the full window; len{3,5,10,20,40}_mean; len{5,10,20}_std; position: early_minus_late and position_std for L=5 and L=10 |
| strict OOF for integration | validation cookies: window model fit on the outer training cookies; training cookies: inner 4-fold by day groups inside the outer training part (CatBoost window model only) |
| integration set (<= 14 features) | seq_full, chunk_mean, chunk_std, chunk_q10, chunk_median, chunk_q90, chunk_min, chunk_max, len5_mean, len10_mean, len20_mean, len5_std, position_std_L5, frac_gt07 |
| controls | (1) GLOBAL vs LOCAL: seq_full alone vs chunk distribution; (2) order: validation events shuffled within cookie (timestamps kept in place, event attributes permuted), windows/features recomputed, scored by the same model; (3) label permutation of training cookies on P0 fold 0 |
| diagnostics | matched TP (3 per persistent FP: B0 rank, platform x mouse, event-count; B0-only variant) with bootstrap CI; conditional band ranks 300-900 with length stratification; consistency-at-same-mean (match on B0 + chunk_mean) |
| cheap gate (pass if ANY) | A: window-model cookie P@R meaningful and Spearman with B0 < 0.8; B: any chunk-distribution feature pFP-vs-TP AUC >= 0.65 (length-matched); C: B0 + chunk features >= +0.005 rep without fwd7 loss; D: strong local discrimination stable across weeks |
| neural stage (only if gate passes) | LSTM and BiLSTM: event embedding 12, hidden 48, 1 layer, dropout 0.25, BCE with cookie-balanced weights, Adam 1e-3, random-length chunks each epoch, early stopping on an inner temporal split of the training part, 3 seeds for a promising candidate |
| blends (only if standalone seq is strong) | rank(B0) + λ·rank(seq), λ in {0.1, 0.2, 0.3} |
| success | Level 2: >= +0.01 rep AND >= +0.01 fwd7 AND >= 4/5 splits; final acceptance: rep >= 0.910 AND fwd7 >= 0.910 AND >= 4/5 |

Anything added after the first outer result is marked POST-HOC in RESEARCH_LOG.md.

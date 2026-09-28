# Cycle 7 — ISIT winner transfer: local sequence discriminative modelling

Plan: `PLAN_LOCKED.md` (locked after the data audit, before any model result).

## Chronological log
| step | result | decision |
|---|---|---|
| C7.0 | ISIT 2024 report not readable in this environment (no PDF attached; ieeexplore / kaggle / researchgate blocked). Winners' methods taken from the cycle-7 task text | stated in plan and report |
| C7.1 | B0 OOF arrays bit-identical to cycle 2 in all 6 schemes (0.9003 / 0.9023) | baseline OK |
| C7.2 | pointer only on web, one point per event, median gap between consecutive pointer events 43 s; events: bots median 20, humans 12 | no dedicated mouse LSTM; kinematics as per-event tokens |
| C7.4 | 96 000 random training windows (6 / cookie), 213 214 deterministic inference windows; order-shuffled copy | — |
| C7.5 | label-permutation control: window-mean AUC 0.358 (< 0.5: sparse-region artefact of trees on 8 % random labels, no positive leakage) | sanity OK |
| C7.6 | cheap CatBoost windows: cookie P@R seq_full 0.665 / chunk_mean 0.653 (rep), fwd7 0.618 / 0.571; RF 0.616 / 0.606; Spearman with B0 0.68; window AUC 0.85 (L=3) … 0.97 (L=40) | local signal exists but far below B0 |
| C7.7 | persistent FP vs matched TP (length-matched): all chunk features 0.50–0.60, CIs cover 0.5; consistency at same mean: std 0.545, q10 0.486, min 0.511 | **core hypothesis refuted** (TPs are not bot-like in every chunk either: chunk_min median 0.107 vs 0.085) |
| C7.7 | order control: shuffled order -> window AUC 0.885 -> 0.887, score corr 0.90 | signal is bag-of-events, not order |
| C7.8 | B0 + 14 locked chunk features: rep 0.8993 (−0.001), fwd7 0.8838 (−0.019), FP −7/+10 | gate C fails |
| C7.8 | CHEAP GATE: A formally passes (P@R 0.65, ρ 0.68 < 0.8); B, C, D fail | neural branch opened per locked rule |
| C7.9 (pre-result detail) | torch 2.5.1 (CPU) installed from PyPI; epoch = 8 random chunks per training cookie (1 chunk/epoch gave ~28 steps and an unconverged inner curve); max 25 epochs, patience 3 on the inner temporal hold-out. Chosen from the INNER learning curve of P0 fold 0 only, before any outer result | locked before outer evaluation |
| C7.10-12 | LSTM / BiLSTM on 32 folds: rep 0.592 / 0.606, fwd7 0.563 / 0.532, Spearman with B0 0.66 / 0.67; locked rule -> LSTM (BiLSTM +0.014 rep but −0.031 fwd7) | weaker than the cheap window CatBoost |
| C7.13 | LSTM chunk features, pFP vs matched TP (length-matched): best 0.64–0.65 in the REVERSE direction (pFP more bot-like: len3_mean 0.35 vs 0.25); consistency at same mean 0.46–0.55 | hypothesis refuted again |
| C7.14 (partial) | ablations on P0 folds 0-2 (inner AUC): event 0.80, +timing 0.84–0.86, +pointer 0.91–0.92, full 0.92, pointer-only (mouse subset) 0.93–0.94 | pointer carries the local signal |
| audit | test_leakage7: 6/6 ok; 0 duplicate sequences; adversarial 0.506 | — |
| STOP | stopped on user request before LSTM integration / seeds / NN shuffle controls | verdict C: ISIT winner methods do not transfer — KEEP B0 |

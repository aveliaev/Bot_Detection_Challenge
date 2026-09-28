# Cycle 6.5 — cross-modal conditional residuals (final falsification experiment)

Hypothesis: at equal B0, a true positive and a persistent FP differ in how *unexpected one channel is given the others*
(behaviour − E[behaviour | mouse, item, context]), not in how bot-like each channel is (that was cycle 4).
Mappings are label-free Ridge regressions (alpha = 1, no search) between compact representations of existing features;
residuals are strict OOF (validation: mapping fit on the fold's training rows; training rows: inner leave-one-day-out).

| mapping | hypothesis | source → target | same-row baseline | ΔR² (P0 / fwd7) | shuffle control ΔMAE | pFP vs matched TP (best feature; + n_events matching) | band 300–900 AUC (length-stratified) | integration rep / fwd7 (Δ) | decision |
|---|---|---|---|---|---|---|---|---|---|
| CB | baseline | context → behaviour | constant | +0.005 / +0.004 | — | — | — | — | baseline |
| **MB** | bot-like mouse with unexpected behaviour | mouse+ctx → behaviour (mouse rows) | CB on mouse rows | **+0.011 / +0.009** | +0.008 | MB_L1 0.665 (CI 0.56–0.78) → **0.561** with n_events matching (length-confounded, ρ with n −0.56) | 0.55 | not integrated | **fails mapping gate** (mouse carries ~1 % of behaviour variance) |
| IB | timing unexpected given item strategy | item+ctx → behaviour | CB | +0.214 / +0.206 (activity R² 0.53, timing 0.08) | +0.041 | IB_L1 0.606 → 0.544 | 0.556 | 0.8979 / 0.8940 (−0.002 / −0.008) | fail |
| BI | item strategy unexpected given behaviour | behaviour+ctx → item | CI | +0.288 / +0.286 | +0.042 | BI_pool 0.373 (pFP **more** pool than expected; 0.61 with n) | **0.659** (weeks 0.63 / 0.67), adversarial 0.605 | 0.8958 / 0.8966 (−0.005 / −0.006) | fail (signal real but not new; wrong direction for pFP) |
| MIB | behaviour unexpected given mouse AND item | mouse+item+ctx → behaviour (mouse rows) | IB on mouse rows | **+0.0001 / −0.0006** over IB (+0.21 over context) | +0.046 | MIB_L1 0.640 → 0.562 | 0.537 | 0.9015 / 0.8899 (+0.001 / −0.012) | fail (mouse adds nothing over item) |
| all | IB + BI + MIB | — | — | — | — | — | — | 0.8953 / 0.8834 (−0.005 / −0.019) | fail |

Signed residuals (hypothesis "pauses longer than expected for their mouse/item pattern"): direction present but weak —
MB_timing / MIB_timing / MB_dtmed: pFP medians −0.17 / −0.04 / −0.38 vs matched TP −0.42 / −0.28 / −0.57, AUC 0.57–0.58
(0.54–0.56 with event-count matching). Far below 0.70.

Leakage tests (`test_leakage65.py`): 7/7 ok. Adversarial: IB 0.509, MIB 0.497, MB 0.500, BI 0.605 (pool-aging shift).

**Verdict: NO USEFUL CROSS-MODAL SIGNAL — B0 IS FINAL.**

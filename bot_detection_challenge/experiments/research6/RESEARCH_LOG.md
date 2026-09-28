# Research cycle 6 — auxiliary / self-supervised behavioural learning

## 0. Written BEFORE the first experiment

### 0.1 What cycles 1–5 already tested (do not repeat under another name)
| area | what exactly | outcome |
|---|---|---|
| aggregates | ~310 cookie-level features: activity, sessions, Δt quantiles/bands/entropy (`dtbin_ent`, `dtexact_*`, `ddt_*`), event counts, bigram/trigram entropies and shares, search, items, pointer, UA, time of day, last-N | B0 = CatBoost top-120 + item pool: rep 0.9003 / fwd7 0.9023 / hidden 0.8953 |
| sequence as text | TF-IDF over event sequences + LR (cycle 1) | 0.28 alone, nothing in blend |
| supervised per-event evidence | cycle 2 NB-LLR of Δt, transitions, pointer cells, pages; cycle 3 LLR motifs (transition\|Δt, 3-gram\|Δt, page\|Δt, hashed signatures) vs all / hard humans | redundant with B0, adversarial 0.75–0.99, integration −0.004…−0.008 |
| item graph / pool variants, UA TE, manifold/prototypes, clustering, pseudo-services, hierarchical, multiclass | cycles 2–3 | all ≤ B0 |
| channel experts + decision fusion (Iliou et al.) | cycle 4 | all ≤ B0 |
| local reranking around the operating point (649 configs, nested) | cycle 5 | 0.8933 / 0.8986, B0 final |
| persistent FP | 45 negatives flagged by all 20 runs; same pointer generator and item-pool usage as bots, more human rhythm; profile mouse HIGH / item HIGH / behaviour LOW, shared by many TPs | irreducible so far |

All of these are **supervised** functions of a cookie's own aggregates or of per-pattern label statistics.

### 0.2 What cycle 6 does differently
1. **No labels in the auxiliary layer.** Models are fitted on *other cookies' sequences* to predict the *process*
   (next event, next Δt, remaining activity, remaining new items). A cookie's feature is how well a population model of
   the process predicts **this** cookie — a learned cross-cookie quantity, not an aggregate of the cookie itself
   and not a label statistic.
2. **Strict provenance.** Auxiliary models are fitted leave-one-day-out over the labelled train days: a cookie of day d
   is scored by models fitted on other days' cookies only (never on itself). Labels are not used at all in this layer,
   so the same feature table is valid for every outer fold. For test: models fitted on all train days.
3. **Prefix dynamics** (phase B): the same target model trained on features of the first 25/50/75/100% of events,
   strict OOF on the B0 folds; the object of study is the *trajectory* of the score, which no previous cycle looked at.
4. Diagnostics are **conditional on B0** (inside B0 deciles / near the operating point) and on length, because an
   auxiliary signal is only useful if it is new information.

### 0.3 What will count as a new signal (fixed now)
- Auxiliary task quality: clearly better than its trivial baselines (global frequency / Markov / medians) **and** a
  negative control (shuffled order / shuffled Δt) must degrade it — otherwise the model learned no process.
- Diagnostic: AUC(persistent FP vs matched TP, matched on B0 / platform / mouse availability) ≥ 0.65 with a bootstrap
  CI excluding 0.5, **or** conditional AUC in the B0 band around the operating point ≥ 0.60 that survives
  event-count matching.
- Decision metric: B0 + family on the same P0–P4 + fwd7 rows. Level 2 (≥ +0.01 rep **and** ≥ +0.01 fwd7, ≥ 4/5 splits)
  is required to replace B0; +0.003 … +0.007 is "interesting, not enough"; < +0.003 is noise.
- Gates as specified: cheap gate after the four cheap tasks, second gate after prefix dynamics. No signal → stop,
  no world models / cross-modal / multi-task.

## 1. Results (chronological)
| ID | step | result | decision |
|---|---|---|---|
| C6.0 | B0 re-run (`r0_reproduce_b0.py`) | rep 0.9003 ± 0.0035, fwd7 0.9023; OOF arrays bit-identical to cycle 2 | baseline OK |
| C6.1 | datasets | 267 833 transitions / 15 641 cookies; 33 063 prefixes / 11 021 cookies (n ≥ 8); 9 in-window event types (captcha only post-window) | — |
| C6.2 | A next event (LGB, LODO) | NLL 1.666 vs global 1.626 / Markov-1 1.626 / Markov-2 1.628; shuffled order 1.665 | **no sequential structure**: task fails quality condition |
| C6.3 | B next Δt | bin-NLL 1.971 vs 2.078; MAE 1.349 vs 1.412 (median); shuffled Δt 1.343 | gain = cookie Δt level, not sequence |
| C6.4 | C remaining count / D new items | MAE 0.478 vs 0.525, 0.520 vs 0.567 (length-conditional baseline) | learnable, strongly length-driven (ρ with n up to 0.81) |
| C6.5 | 12 cookie surprise features, label-free LODO | adversarial train/test 0.50–0.51 | — |
| C6.6 | cheap gate | pFP vs matched TP: best 0.641 (ev_nll_mean, CI 0.54–0.74); band AUC dt_nll_std 0.606 (length-stratified), stable 0.616 / 0.591 by week; B0 + aux: all 0.9012/0.8891, time 0.9024/0.8968, event 0.8963/0.9026, future 0.9023/0.8984; 45/45 pFP flagged everywhere | Level 0; continue to phase B per protocol |
| C6.7 | prefix target models (strict OOF, suffix-free features) | P@R rep 0.593 / 0.818 / 0.883 / 0.9003 at 25/50/75/100 %; fwd7 0.589 / 0.743 / 0.886 / 0.9023 | — |
| C6.8 | score trajectories | pFP 0.69→0.77→0.85→0.90 ≈ TP 0.73→0.83→0.89→0.91, first P>0.5 at 0.34 vs 0.35; dynamics vs matched TP AUC 0.43–0.55 | pFP are bot-like from the first quarter |
| C6.9 | B0 + dynamics (stacked OOF; nested fwd7) | rep 0.8866 (−0.014), fwd7 0.8861, P(better) 0.06 | **second gate: STOP** |
| audit | `test_leakage6.py` | 8/8 ok (history-only inputs, suffix isolation, aux isolation, P100 == B0 path, nested fwd7, window filter) | — |

Phases C–H (world models, cross-modal residuals, motor predictability, latent states, multi-task) were **not built** — both gates failed.
Verdict: **NO ADDITIONAL INFORMATION — KEEP B0** (see REPORT_CYCLE6.md).

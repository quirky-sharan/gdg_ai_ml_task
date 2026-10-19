# AUDIT: WAFER-GUARD release candidate (`bnypi.ipynb`)

**Verdict: NO-GO.** The reported F1 = 0.99 is an evaluation artifact. When the same recipe is evaluated honestly it scores **F1 = 0.00**.
The repaired detector, trained only on the past and scored once on the 313 most recent lots at their natural defect rate, reaches
**ROC-AUC 0.70, AP 0.23 (4.3× the 0.054 no-skill rate) and F1 0.17** at its validation-chosen threshold. That is far below the
0.90 gate. Every number below is an output saved in `wafer_guard_fixed.ipynb` (also in `results/metrics.json`).

## 1 · Findings, ranked by severity

| # | Sev. | Location | Defect → mechanism | Evidence (measured) | Repair |
|---|---|---|---|---|---|
| 1 | **Critical** | §2 `SMOTE` before `train_test_split` | Synthetic defects are created from *all* real defects, then split. The test set is 50 % defective and mostly fake, and the children of each test defect sit in train. | 274 of 293 test defects are synthetic. Real test defects lie a median **0.99** z-units from a training row, versus **17.8** for real passes. The same RC recipe evaluated honestly falls from F1 **0.992 → 0.000** and AP **1.000 → 0.137** | Split first; no synthetic data; class imbalance via `pos_weight` (training window only) |
| 2 | **Critical** | §2 random split; `imputer`/`scaler.fit` on all rows | Lots are time-ordered and drift (defect rate falls from 15 % to 2 %). The model trains on lots *after* the ones it is graded on, and future statistics enter preprocessing. | **100 %** of real test lots were scored by a model that saw later lots. **68** sensors shift > 0.5 SD in the last 20 % of lots | Chronological 80/20 split (equal timestamps never straddle it). Every statistic is fitted on the training window. Epochs and threshold come from past-only rolling validation |
| 3 | **High** | §3 loader `unsqueeze(1)` + §4 `flatten` + §5 loss | Target `[64,1,590]` minus output `[64,590]` silently broadcasts to `[64,64,590]`. Each reconstruction is compared with every wafer in the batch. | ∂L/∂x̂ = 2/(B·F)·(x̂ − batch-mean) exactly. A **perfect** reconstruction scores 2.30, versus 1.15 for outputting the batch mean. RC reconstruction loss goes 0.681 → 0.677 in 40 epochs. Re-injected into the repaired model, test reconstruction MSE is **1.425, identical to predicting the mean**, versus **1.10** fixed | Flat `[B,F]` tensors, masked MSE, exact-shape assertions, a gradient unit test, and a model that rejects 3-D input |
| 4 | **High** | §2 imputer/scaler on `sensor_0/1` | PS-L sentinel `0.0` and firmware codes `100/200/300` are treated as physical readings. | `sensor_0` mean/std **2732/879** vs genuine **3014/74**, so real variation is squashed **11.8×** (8.4× for `sensor_1`). Mean-imputation writes **2732.1** into missing `sensor_0`, lower than *every* genuine reading. The codes become ordinal z-values (−3.23/−3.09/−2.94) | Legacy-aware preprocessor (see Q1c) |
| 5 | Medium | §6 `probs >= 0.5` | The threshold was never tuned, and a model trained at 50 % prevalence is mis-scaled at 6.6 %. | The RC recipe evaluated honestly flags **zero** lots at 0.5 | Threshold that maximises F1 on out-of-time validation, then frozen |
| 6 | Medium | whole notebook / §7 gate | No validation data, model selection or uncertainty. The gate is F1 on a balanced synthetic set, so it cannot fail. | A 17-defect test window gives F1 a 95 % CI of [0.00, 0.38] | Rolling-origin validation, bootstrap CIs, and a gate at natural prevalence |
| 7 | Low | §7 artifacts | `preprocessor.joblib` is fitted on all rows, has no PS-L logic, and no threshold is saved. Production would re-create Bug 4. | n/a | The fitted preprocessor, threshold, epoch budget, seeds and training cut-off are saved together |

## 2 · Required audit questions

**Q1. What were the three silent bugs, and how were they fixed?**

* **(a) Data leakage (findings 1 + 2).** SMOTE ran before the split, the split was time-shuffled, and preprocessing was fitted on all rows. *Fix:* sort by
  `timestamp`; the last 20 % (313 lots) is a test window touched once. Every fitted object (channel filter, quantile scaler) sees only training lots.
  The epoch budget (38) and decision threshold (0.142) come from 4 expanding-window folds *inside* the development period (22 validation defects).
  No synthetic samples are used anywhere. Imbalance is handled with `pos_weight = n_pass/n_fail` of the training window.
* **(b) Corrupted loss gradients (finding 3).** A `[B,1,F]` vs `[B,F]` broadcast turned the reconstruction term into "predict the batch mean". *Fix:* tensors
  stay `[B,F]` end to end. `sae_loss` asserts `x̂.shape == x.shape == mask.shape` and `logits.shape == y.shape == [B]`, and computes a masked MSE over
  observed cells only. A unit test checks ∂L/∂x̂ᵢ = 2(x̂ᵢ − xᵢ)·maskᵢ / #observed (no cross-sample terms). The decoder now learns: test reconstruction MSE
  is 1.10, against 1.43 for the mean predictor.
* **(c) Invariant blindness (finding 4).** The pipeline ignored the PS-L tool-matching invariant (`sensor_0 ≡ 0.0` ⇔ `sensor_1 ∈ {100,200,300}`, verified
  on all 146 PS-L lots). *Fix:* on PS-L rows both channels are removed from the continuous block. They are set to NaN before any statistic is fitted, so they are
  never scaled and never imputed with a fabricated reading. They are input-masked to 0 and excluded from the reconstruction target. The firmware code is
  one-hot encoded (`code_100/200/300`, plus `code_unknown` for unseen firmware) together with an `is_PSL` flag. Genuine readings on modern stations keep
  their full variance (standardised std 1.02).

**Q2. What are the true metrics of the repaired model at the natural defect prevalence?**
Test window: the 313 most recent lots (2008-10-02 → 10-17), with 17 defective (**5.43 %**, the period's real rate; nothing resampled). Threshold 0.142.
The 5-seed ensemble was trained on the 1,254 earlier lots.

| Metric | Value | 95 % bootstrap CI |
|---|---|---|
| ROC-AUC | **0.698** | 0.560 – 0.833 |
| Average precision (PR-AUC) | **0.233** | 0.078 – 0.436 (no-skill = 0.054) |
| Precision | 0.286 | 0.000 – 0.667 |
| Recall | 0.118 (2 of 17) | 0.000 – 0.300 |
| **F1** | **0.167** | 0.000 – 0.381 |
| MCC | 0.154 | −0.034 – 0.385 |
| Confusion | TP 2 · FP 5 · FN 15 · TN 291 | false-alarm rate 1.7 %, alarm rate 2.2 % |
| Re-weighted to the data set's 6.64 % prevalence | precision 0.331 · F1 0.174 · AP 0.255 | |

Reference models on the same test window and the same causal features: logistic regression AP 0.187 / AUC 0.669, random forest AP 0.100 / AUC 0.634.
The RC recipe evaluated honestly gets AP 0.137 / AUC 0.713 / F1 0.000.

## 3 · Why these metrics and this threshold

* **AP (PR-AUC)** is the primary metric. It is threshold-free, focuses on the rare class, and its baseline equals the prevalence, so a lift is easy to read.
  ROC-AUC is reported but is insensitive to prevalence. **F1** is reported because it is the gate metric, and MCC as a balanced single number.
  With 17 test defects every point estimate needs its CI.
* **Threshold:** maximise F1 on the pooled out-of-time predictions of the 4 rolling folds at the selected epoch, then freeze it before the test window is scored.
  Folds are pooled because 2-defect folds make per-fold F1 curves degenerate. This mirrors `TunedThresholdClassifierCV` with refit.
* **Prevalence:** the test window's own rate (5.43 %) is the honest "natural" rate for that period. Metrics re-weighted to the full-data 6.64 % are given alongside.

## 4 · Residual risks

1. **Weak, drifting signal.** Out-of-time validation AUC ranges from 0.49 to 0.71 across folds. The test window happens to be one of the easier periods.
   Expect large swings from one month to the next.
2. **The threshold does not transfer well.** The alarm rate was 4.6 % on validation but 2.2 % on test, because score scales shift between refits. As a result only
   2 of 17 defects were caught. Thresholds must be re-calibrated on recent labelled lots. Descriptively, the test ranking would catch
   4/17 defects at a 5 % alarm budget, 5/17 at 10 % and 8/17 at 20 %. Those cut-offs are read off the test scores, so they are not deployable.
3. **PS-L blind spot.** None of the 146 PS-L lots is defective, about 10 fewer than chance would predict. The model learns PS-L ⇒ pass (mean score 0.000 vs 0.008),
   so it cannot detect a PS-L defect and its PS-L recall cannot be measured. PS-L lots need a separate (rule-based or manual) screen until PS-L failures are labelled.
4. **Model choice is not settled.** On the test window the SAE ranks best (AP 0.233 vs LR 0.187 vs RF 0.100). During development a random forest looked
   stronger (`experiments/dev_cv_results.txt`). Rankings between models are unstable at about 100 failures.
5. Bootstrap CIs assume independent lots, but there are 33 duplicated timestamps and lot-to-lot correlation, so the true uncertainty is wider.

## 5 · Recommendation

**Stop Friday's release.** The model cannot meet F1 ≥ 0.90. Its honest F1 is 0.17 (CI up to 0.38), and the existing gate measured leakage.
Before any future go: (i) re-define the gate on out-of-time, natural-prevalence metrics with CIs, for example recall at an alarm budget agreed with the fab;
(ii) run the repaired model in shadow mode with drift monitoring and scheduled threshold re-calibration; (iii) collect more labelled failures,
including PS-L.

*Reproduce:* `pip install -r requirements.txt && python tools/run_notebook.py` (CPU, about 4–25 min depending on load; seeded and deterministic, and two full runs produced identical metrics). `secom_legacy_mask_EVALUATOR_ONLY.csv` was
neither read nor used.

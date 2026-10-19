# Model-selection experiments (development period only)

These scripts record how the repaired configuration was chosen. All of them read only the first 80 % of lots
(the development period). The test window (last 20 %) is never loaded for any decision.

| File | What it is |
|---|---|
| `wg.py` | Experiment library: legacy-aware preprocessor, shape-safe WaferSAE, masked loss (same logic as the notebook). |
| `explore.py`, `grid.py` | First pass on a single chronological 60/20 hold-out. It was abandoned because the validation block holds only 11 defects. → `single_holdout_results.txt` |
| `devcv.py` | Rolling-origin CV with 4 expanding windows inside the development period (22 validation defects). → `dev_cv_results.txt` |

Run from the repository root, for example `python experiments/devcv.py q1` (each config takes about 5 min on CPU).

**Outcome.** Rank-based (quantile → N(0,1)) scaling fitted on the training window roughly doubles validation AP compared with
z-scoring (about 0.10 vs 0.055). Every other change tried (capacity, dropout/weight decay, learning rate, loss weighting,
class weighting, missing-indicator channels, a stricter missing-rate filter) stayed within seed and fold noise. The
release-candidate architecture and hyper-parameters were therefore kept. Dropping the PS-L categorical features lowered validation AP slightly
(0.08 vs 0.09). The categorical encoding is kept because the business logic requires it (see `AUDIT.md`, residual risks).

Note: these runs used 60 epochs (cosine T_max = 60) and 3 seeds. The notebook re-runs the selection itself with the
release-candidate budget of 40 epochs and 5 seeds, and its saved outputs are the numbers of record.

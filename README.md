# WAFER-GUARD: audit and repair of the Fab-7 excursion detector

The release candidate (`original/bnypi.ipynb`) reported **F1 = 0.99** and passed its F1 ≥ 0.90 release gate. The score was an artifact
of three silent bugs: data leakage, a broadcasting loss and PS-L legacy-station blindness. This repository holds the audited,
repaired pipeline and an honest re-evaluation at the natural defect prevalence.

| File | |
|---|---|
| **`wafer_guard_fixed.ipynb`** | Repaired notebook, executed, with all outputs saved. Part A reproduces and dissects the release candidate; Part B is the repaired pipeline, honest test evaluation and ablations. |
| **`AUDIT.md`** | Findings ranked by severity, the repairs, answers to the required audit questions, measured metrics, residual risks and the go/no-go call. |
| `results/metrics.json` | Machine-readable copy of every number reported (written by the notebook). |
| `original/bnypi.ipynb` | The release candidate as received, unmodified. |
| `experiments/` | Development-period-only model-selection runs (rolling-origin CV) and their logs. |
| `tools/` | `build_notebook.py` generates the notebook; `run_notebook.py` executes it headless. |
| `secom_telemetry_modified.csv` | Input data, as supplied. |

## Run it
```bash
pip install -r requirements.txt
python tools/run_notebook.py          # or open wafer_guard_fixed.ipynb and "Run all" (CPU, ~4–25 min)
```

## Headline (test window = 313 most recent lots, natural prevalence, untouched)
| | Release candidate (as reported) | Release-candidate recipe, honest protocol | **Repaired WaferSAE** |
|---|---|---|---|
| Test set | 586 rows, 50 % defective, 274 of 293 defects synthetic | 313 most recent lots, 5.43 % defective | 313 most recent lots, 5.43 % defective |
| F1 | 0.992 | 0.000 | **0.167** [0.00–0.38] |
| ROC-AUC | 1.000 | 0.713 | **0.698** [0.56–0.83] |
| Average precision (no-skill 0.054) | 1.000 | 0.137 | **0.233** [0.08–0.44] |

**Release gate (F1 ≥ 0.90): FAILED. Recommendation: NO-GO.** Brackets are 95 % bootstrap CIs.

See `AUDIT.md` for the full write-up.

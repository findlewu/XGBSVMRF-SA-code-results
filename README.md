# XGBSVMRF-SA: code and aggregate results

This folder contains analysis code and selected aggregate numerical results for the manuscript **“Hyperspectral Grading of Rice Blast Severity (0–5) Using Regional Spectral-Angle Features and Heterogeneous Stacking.”** It is a limited public-release package, not a raw-data release.

## Contents

- `code/`: Python scripts for the grouped model comparison and supporting analyses.
- `results/primary_grouped_fivefold/`: fold-level results, model summaries, pooled class metrics, and run metadata for the final no-resampling comparison.
- `results/figure_source_data/`: aggregate numerical inputs for the figures and regional analyses.
- `results/learning_curve/`: aggregate learning-curve results.
- `requirements.txt`: package versions used for the analysis environment.

Individual spectra, acquisition records, group identifiers, and per-spectrum predictions are **not included** because permission to publish the source data has not yet been confirmed. Consequently, the scripts document the analysis procedure, but the model-training runs cannot be reproduced from this package alone. The included numerical outputs can be inspected without the source records.

## Analysis environment and input convention

The final runs used Python 3.10.11, NumPy 2.2.6, pandas 2.3.3, SciPy 1.15.3, scikit-learn 1.7.2, matplotlib 3.10.7, and XGBoost 3.1.2 with CUDA. The code expects source tables under `data/raw_spectra/` relative to this README. That directory is intentionally absent from the release.

If the source data become available through a separately authorized route, install the dependencies with `pip install -r requirements.txt` and run `python code/run_final_common_grouped_fivefold_gpu.py` from the package root. The principal evaluation uses five outer pot-grouped folds with seed `20260929`, three inner grouped folds for stacking, no resampling, and no grade-0/1 threshold adjustment. XGBoost is configured for CUDA, so reproducing the final run requires a compatible GPU installation.

Results are descriptive summaries of the recorded cohort and grouped folds, not independent external validation. The `results/figure_source_data/` directory also contains supporting analyses with their own comparison settings; their values should not be pooled indiscriminately with the principal comparison.

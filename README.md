# Gaia BAV ring temperature project

This project is a PyCharm-friendly refactor of the two original notebooks that model
Gaia basic-angle LOS variation from adapter ring temperatures.

## What this project gives you

- one reusable codebase for both FoVs
- config-driven experiments instead of hard-coded notebook paths
- a clean split between:
  - data loading
  - feature engineering
  - model training
  - evaluation
  - plotting
- saved outputs for metrics, predictions, figures, and trained models
- side-by-side random-split and time-blocked validation runs
- a better starting point for deeper scientific analysis and paper writing



## What gets saved

For each experiment, the pipeline writes into the configured output directory:

- `enriched_dataset.csv`
- `validation_comparison.csv` and `validation_comparison.json`
- `random_split/` with metrics, predictions, model, SHAP, and plots
- `time_blocked/` with metrics, predictions, model, SHAP, and plots
- `plots_global/*.png` for figures based on the full dataset

## Core design choices

### 1. Config-driven experiments
The FoV-specific differences are now in TOML files instead of duplicated notebook cells.

### 2. Reusable feature engineering
The notebook loop that computes ring-temperature means around each LOS sample is now a dedicated function in `features.py`.

### 3. Reproducible runs
All outputs are tied to a named experiment and saved to one output directory.

### 4. Built-in validation comparison
Each run now evaluates the model twice:
- `random_split`: reproduces the notebook-style random train/test split
- `time_blocked`: trains on the earlier OBMT block and tests on the final contiguous block

The time-blocked run also supports a configurable `gap_size` in rows to reduce leakage from rolling-window features near the train/test boundary.

### 5. Easy extension path
This structure makes it straightforward to add:
- rolling or walk-forward validation
- baseline linear models
- lagged temperature features
- additional thermal sensors
- paper-ready figure scripts

## Files worth reading first

- `docs/notebook_audit.md`
- `src/gaia_bav_ring_temp/pipeline.py`
- `configs/fov1_full.toml`
- `configs/fov2_full.toml`

## Important scientific note

The original notebooks use random train/test splits. This project now also runs a time-blocked validation by default so you can directly compare the optimistic random-split score against a more realistic future-block holdout.


Note: the feature averaging now explicitly preserves the original notebook window rule so migrated results match the notebook as closely as possible.


## Multi-model comparison

The pipeline can compare multiple models on the same validation splits. By default the config enables:

- `random_forest`
- `linear`
- `ridge`
- `mlp`

Each run writes `model_validation_comparison.csv` so the models can be compared directly for both random and time-blocked validation.

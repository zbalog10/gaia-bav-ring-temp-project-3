# Notebook audit and migration notes

## What the original notebooks currently do

Both notebooks follow the same high-level workflow:

1. Load adapter ring temperature data from `ring_temp.csv`.
2. Load FoV-specific LOS variation data from a detrended `sfunc7000` CSV.
3. Convert LOS variation from radians to mas using the constant  
   `2.0626480624709636e8`.
4. Build three averaged temperature features:
   - `NEI00054_avg`
   - `NEI00230_avg`
   - `NEI00321_avg`
5. Train a random forest regressor to predict LOS variation in mas.
6. Produce diagnostic plots:
   - predictions vs truth
   - residuals vs predicted
   - residual histograms
   - residuals vs time
   - feature importance
   - residuals vs features
   - correlation heatmap
7. In FoV1, also compute SHAP values and SHAP-vs-time plots.
8. Repeat the model for:
   - full time range
   - zoomed time range: `16490.09 < obmtRev < 16571.17`

## Main structural problems found

- Absolute local paths are hard-coded throughout.
- Execution depends on notebook state across cells.
- FoV1 and FoV2 duplicate most logic instead of reusing one pipeline.
- Data prep, model training, evaluation, and plotting are mixed together.
- Output locations are embedded directly in plotting cells.
- Experiment parameters are not centralized.
- The current time-window averaging is implemented as a Python loop over rows.

## How the new project maps onto the notebooks

- `dataset.py` replaces data loading and target scaling.
- `features.py` replaces the per-row averaging loop with a reusable function.
- `modeling.py` handles train/test split and model fitting.
- `evaluation.py` handles metrics, predictions, and SHAP.
- `plotting.py` contains figure-generation functions.
- `pipeline.py` runs a complete experiment from config.
- `configs/*.toml` separate FoV-specific settings from code.

## Recommended next extension steps

1. Add a linear-model baseline and compare against random forest.
2. Add blocked time splits so train/test separation respects temporal structure.
3. Add lagged and derivative temperature features.
4. Add experiment tracking for paper-quality tables and figure metadata.
5. Add a formal comparison between FoV1 and FoV2 feature importance / SHAP structure.

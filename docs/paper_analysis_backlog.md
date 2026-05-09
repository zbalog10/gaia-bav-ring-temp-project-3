# Suggested next scientific steps after the refactor

This is the natural analysis roadmap once the notebooks are running as a reusable project.

## 1. Establish the baseline cleanly
- Reproduce the notebook random forest results for FoV1 and FoV2.
- Save metrics and figures in a deterministic directory structure.
- Freeze the exact dataset and config used for each result.

## 2. Check whether the current split is optimistic
The notebooks use random train/test splits. Because the data are time ordered, this can leak neighboring states into both train and test.
Recommended extensions:
- contiguous train/test splits
- blocked cross-validation
- leave-one-SAA-block-out experiments

## 3. Test feature engineering choices
The current features are local averages over the three ring sensors. Compare against:
- instantaneous interpolated temperatures
- rolling means with fixed physical timescales
- temperature differences between sensors
- first derivatives / rates of change
- lagged temperatures

## 4. Quantify where the model works and fails
Break performance down by:
- FoV
- SAA interval / operational phase
- time
- LOS amplitude
- thermal state

## 5. Compare model families
At minimum:
- linear regression
- ridge / lasso
- random forest
- gradient boosting

This will show whether the ring-temperature relationship is mostly linear or genuinely nonlinear.

## 6. Tie results to the paper text
You will likely want:
- one methods subsection describing data alignment and model setup
- one results subsection comparing FoV1 and FoV2
- one robustness subsection on time-split validation
- one interpretation subsection discussing thermal sensitivity and possible physical meaning

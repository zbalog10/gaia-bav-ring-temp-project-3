from __future__ import annotations

import numpy as np
import pandas as pd


SUPPORTED_TEMPERATURE_DETREND_METHODS = ("none", "linear")


def _linear_detrend_series(
    obmt: pd.Series,
    values: pd.Series,
    *,
    preserve_mean: bool = True,
) -> tuple[pd.Series, pd.Series]:
    """Remove a linear trend from one feature as a function of OBMT.

    Returns
    -------
    detrended : pd.Series
        Feature with the fitted linear trend removed.
    trend : pd.Series
        The fitted linear trend evaluated at all valid rows.
    """
    obmt_num = pd.to_numeric(obmt, errors="coerce")
    values_num = pd.to_numeric(values, errors="coerce")

    detrended = pd.Series(np.nan, index=values.index, dtype=float)
    trend = pd.Series(np.nan, index=values.index, dtype=float)

    valid = obmt_num.notna() & values_num.notna()
    if valid.sum() < 2:
        detrended.loc[valid] = values_num.loc[valid]
        return detrended, trend

    x = obmt_num.loc[valid].to_numpy(dtype=float)
    y = values_num.loc[valid].to_numpy(dtype=float)

    x0 = float(np.mean(x))
    slope, intercept = np.polyfit(x - x0, y, deg=1)
    fitted = slope * (x - x0) + intercept

    if preserve_mean:
        y_ref = float(np.mean(y))
        detrended.loc[valid] = y - fitted + y_ref
    else:
        detrended.loc[valid] = y - fitted

    trend.loc[valid] = fitted
    return detrended, trend


def detrend_temperature_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    obmt_col: str,
    *,
    method: str = "linear",
    keep_raw_columns: bool = True,
    preserve_mean: bool = True,
) -> pd.DataFrame:
    """Detrend the temperature feature columns used by the model.

    The detrended values overwrite the original feature columns so the rest of
    the pipeline can stay unchanged. Optionally, the original values are kept
    as `<feature>_raw`, and the fitted trend is saved as `<feature>_trend`.
    """
    if method not in SUPPORTED_TEMPERATURE_DETREND_METHODS:
        raise ValueError(
            f"Unsupported detrending method: {method}. "
            f"Supported methods: {SUPPORTED_TEMPERATURE_DETREND_METHODS}"
        )

    result = df.copy()

    if method == "none":
        return result

    if obmt_col not in result.columns:
        raise ValueError(f"OBMT column '{obmt_col}' not found in DataFrame.")

    obmt = result[obmt_col]

    for col in feature_cols:
        if col not in result.columns:
            raise ValueError(f"Feature column '{col}' not found in DataFrame.")

        original = pd.to_numeric(result[col], errors="coerce")
        detrended, trend = _linear_detrend_series(
            obmt,
            original,
            preserve_mean=preserve_mean,
        )

        if keep_raw_columns:
            result[f"{col}_raw"] = original

        result[f"{col}_trend"] = trend
        result[col] = detrended

    return result
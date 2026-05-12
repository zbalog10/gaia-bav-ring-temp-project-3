from __future__ import annotations

import numpy as np
import pandas as pd


SUPPORTED_TEMPERATURE_DETREND_METHODS = ("none", "linear", "hybrid_rolling")


def _linear_detrend_series(
    obmt: pd.Series,
    values: pd.Series,
    *,
    preserve_mean: bool = True,
) -> tuple[pd.Series, pd.Series]:
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


def _estimate_period_fft(
    obmt: np.ndarray,
    values: np.ndarray,
    *,
    initial_window: int = 50,
) -> float | None:
    """Estimate dominant period from the post-switch signal using FFT.

    This mirrors the notebook logic:
    - remove a coarse trend with a rolling median,
    - FFT the residual,
    - take the strongest non-DC frequency.
    """
    if len(values) < max(initial_window, 10):
        return None

    series = pd.Series(values)
    coarse = series.rolling(window=initial_window, center=True, min_periods=1).median()
    periodic_part = series - coarse

    y = periodic_part.dropna().to_numpy(dtype=float)
    if len(y) < 10:
        return None

    dt = np.median(np.diff(obmt))
    if not np.isfinite(dt) or dt <= 0:
        return None

    fft_result = np.abs(np.fft.fft(y))
    frequencies = np.fft.fftfreq(len(y), d=dt)

    if len(fft_result) < 2:
        return None

    # ignore DC component
    positive = np.where(frequencies > 0)[0]
    if len(positive) == 0:
        return None

    idx = positive[np.argmax(fft_result[positive])]
    freq = frequencies[idx]
    if not np.isfinite(freq) or freq <= 0:
        return None

    return float(1.0 / freq)


def _window_size_from_period(
    obmt: np.ndarray,
    period: float,
    *,
    anchor_obmt: float | None = None,
    min_window: int = 5,
) -> int:
    """Convert period in OBMT to a rolling-window size in samples.

    The notebook counted how many points fall into one period near a chosen
    time interval. We mimic that here.
    """
    if len(obmt) == 0 or not np.isfinite(period) or period <= 0:
        return min_window

    if anchor_obmt is None:
        anchor_obmt = float(np.median(obmt))

    count = int(np.sum((obmt > anchor_obmt) & (obmt < anchor_obmt + period)))
    if count < min_window:
        dt = np.median(np.diff(obmt)) if len(obmt) > 1 else np.nan
        if np.isfinite(dt) and dt > 0:
            count = int(round(period / dt))

    return max(int(count), min_window)


def _hybrid_rolling_detrend_series(
    obmt: pd.Series,
    values: pd.Series,
    *,
    switch_obmt: float,
    fft_initial_window: int = 50,
    anchor_obmt: float | None = None,
    preserve_mean: bool = True,
    forced_window_size: int | None = None,
) -> tuple[pd.Series, pd.Series, dict[str, float | int | None]]:
    """Notebook-style detrending:
    - rolling median before switch
    - rolling mean after switch
    - window estimated from FFT on post-switch part
    """
    obmt_num = pd.to_numeric(obmt, errors="coerce")
    values_num = pd.to_numeric(values, errors="coerce")

    detrended = pd.Series(np.nan, index=values.index, dtype=float)
    trend = pd.Series(np.nan, index=values.index, dtype=float)

    valid = obmt_num.notna() & values_num.notna()
    if valid.sum() < 3:
        detrended.loc[valid] = values_num.loc[valid]
        return detrended, trend, {
            "switch_obmt": switch_obmt,
            "estimated_period": None,
            "window_size": None,
        }

    work = pd.DataFrame(
        {
            "obmt": obmt_num.loc[valid],
            "value": values_num.loc[valid],
        }
    ).sort_values("obmt")

    after = work[work["obmt"] >= switch_obmt].copy()
    estimated_period = None
    if len(after) >= 10:
        estimated_period = _estimate_period_fft(
            after["obmt"].to_numpy(dtype=float),
            after["value"].to_numpy(dtype=float),
            initial_window=fft_initial_window,
        )

    if forced_window_size is not None:
        window_size = int(forced_window_size)
    else:
        window_size = _window_size_from_period(
            work["obmt"].to_numpy(dtype=float),
            estimated_period if estimated_period is not None else np.nan,
            anchor_obmt=anchor_obmt,
            min_window=5,
        )

    rolling_median = work["value"].rolling(
        window=window_size,
        center=True,
        min_periods=1,
    ).median()

    rolling_mean = work["value"].rolling(
        window=window_size,
        center=True,
        min_periods=1,
    ).mean()

    fitted = pd.Series(index=work.index, dtype=float)
    before_mask = work["obmt"] < switch_obmt
    after_mask = ~before_mask

    fitted.loc[before_mask] = rolling_median.loc[before_mask]
    fitted.loc[after_mask] = rolling_mean.loc[after_mask]

    y = work["value"].to_numpy(dtype=float)
    t = fitted.to_numpy(dtype=float)

    if preserve_mean:
        y_ref = float(np.mean(y))
        d = y - t + y_ref
    else:
        d = y - t

    detrended.loc[work.index] = d
    trend.loc[work.index] = t

    return detrended, trend, {
        "switch_obmt": float(switch_obmt),
        "estimated_period": None if estimated_period is None else float(estimated_period),
        "window_size": int(window_size),
    }


def detrend_temperature_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    obmt_col: str,
    *,
    method: str = "linear",
    keep_raw_columns: bool = True,
    preserve_mean: bool = True,
    switch_obmt: float | None = None,
    fft_initial_window: int = 50,
    anchor_obmt: float | None = None,
    forced_window_size: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Detrend the temperature feature columns used by the model.

    Returns
    -------
    result_df : pd.DataFrame
        DataFrame with detrended feature columns in place.
    detrend_summary : pd.DataFrame
        Per-feature summary of the detrending settings/results.
    """
    if method not in SUPPORTED_TEMPERATURE_DETREND_METHODS:
        raise ValueError(
            f"Unsupported detrending method: {method}. "
            f"Supported methods: {SUPPORTED_TEMPERATURE_DETREND_METHODS}"
        )

    result = df.copy()
    summary_rows: list[dict[str, float | int | str | None]] = []

    if method == "none":
        return result, pd.DataFrame()

    if obmt_col not in result.columns:
        raise ValueError(f"OBMT column '{obmt_col}' not found in DataFrame.")

    obmt = result[obmt_col]

    for col in feature_cols:
        if col not in result.columns:
            raise ValueError(f"Feature column '{col}' not found in DataFrame.")

        original = pd.to_numeric(result[col], errors="coerce")

        if method == "linear":
            detrended, trend = _linear_detrend_series(
                obmt,
                original,
                preserve_mean=preserve_mean,
            )
            meta = {
                "feature": col,
                "method": method,
                "switch_obmt": None,
                "estimated_period": None,
                "window_size": None,
            }

        elif method == "hybrid_rolling":
            if switch_obmt is None:
                raise ValueError(
                    "switch_obmt must be provided for method='hybrid_rolling'."
                )

            detrended, trend, meta_extra = _hybrid_rolling_detrend_series(
                obmt,
                original,
                switch_obmt=switch_obmt,
                fft_initial_window=fft_initial_window,
                anchor_obmt=anchor_obmt,
                preserve_mean=preserve_mean,
                forced_window_size=forced_window_size,
            )
            meta = {
                "feature": col,
                "method": method,
                **meta_extra,
            }

        else:
            raise ValueError(f"Unhandled method: {method}")

        if keep_raw_columns:
            result[f"{col}_raw"] = original

        result[f"{col}_trend"] = trend
        result[col] = detrended

        raw_valid = pd.to_numeric(original, errors="coerce").dropna()
        detr_valid = pd.to_numeric(detrended, errors="coerce").dropna()

        meta.update(
            {
                "raw_mean": None if raw_valid.empty else float(raw_valid.mean()),
                "raw_std": None if raw_valid.empty else float(raw_valid.std()),
                "detrended_mean": None if detr_valid.empty else float(detr_valid.mean()),
                "detrended_std": None if detr_valid.empty else float(detr_valid.std()),
            }
        )
        summary_rows.append(meta)

    summary_df = pd.DataFrame(summary_rows)
    return result, summary_df


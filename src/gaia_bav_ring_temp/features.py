"""Feature engineering routines."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_window_bounds(target_times: pd.Series | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute notebook-compatible symmetric windows around each target time.

    The original notebooks used a slightly asymmetric rule when stepping through the
    LOS samples:
    - for every row except the last, ``d = t[i+1] - t[i]``
    - for the last row, ``d = t[i] - t[i-1]``
    - the averaging window is then ``[t - d/2, t + d/2]``

    This is *not* the same as using midpoints to the previous and next sample. The
    notebook rule is preserved here so the migrated project reproduces the original
    features and predictions.
    """
    t = np.asarray(target_times, dtype=float)
    if t.ndim != 1:
        raise ValueError("target_times must be 1D")
    if len(t) == 0:
        return np.array([]), np.array([])
    if len(t) == 1:
        return t.copy(), t.copy()

    d = np.empty_like(t)
    d[:-1] = t[1:] - t[:-1]
    d[-1] = t[-1] - t[-2]

    left = t - d / 2.0
    right = t + d / 2.0
    return left, right


def average_columns_in_windows(
    source_df: pd.DataFrame,
    source_time_col: str,
    target_times: pd.Series | np.ndarray,
    value_cols: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    """Average source columns over time windows centered on target_times.

    Uses searchsorted and cumulative sums, so it scales much better than looping
    over every target row.
    """
    if source_time_col not in source_df.columns:
        raise KeyError(f"Missing time column: {source_time_col}")

    df = source_df.sort_values(source_time_col).reset_index(drop=True)
    source_times = df[source_time_col].to_numpy(dtype=float)

    left, right = compute_window_bounds(target_times)
    idx_left = np.searchsorted(source_times, left, side="left")
    idx_right = np.searchsorted(source_times, right, side="right")

    out: dict[str, np.ndarray] = {}

    for col in value_cols:
        values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        valid = ~np.isnan(values)

        cum_sum = np.concatenate([[0.0], np.cumsum(np.where(valid, values, 0.0))])
        cum_count = np.concatenate([[0], np.cumsum(valid.astype(int))])

        sums = cum_sum[idx_right] - cum_sum[idx_left]
        counts = cum_count[idx_right] - cum_count[idx_left]

        out[f"{col}_avg"] = np.divide(
            sums,
            counts,
            out=np.full_like(sums, np.nan, dtype=float),
            where=counts > 0,
        )

    return pd.DataFrame(out)


def build_feature_table(
    ring_df: pd.DataFrame,
    los_df: pd.DataFrame,
    ring_time_col: str,
    los_time_col: str,
    sensor_cols: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    """Return LOS dataframe enriched with averaged ring temperature features."""
    if los_time_col not in los_df.columns:
        raise KeyError(f"Missing LOS time column: {los_time_col}")

    enriched = los_df.sort_values(los_time_col).reset_index(drop=True).copy()
    averages = average_columns_in_windows(
        source_df=ring_df,
        source_time_col=ring_time_col,
        target_times=enriched[los_time_col],
        value_cols=sensor_cols,
    )
    return pd.concat([enriched, averages], axis=1)

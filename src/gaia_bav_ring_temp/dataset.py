"""Dataset preparation."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .features import build_feature_table
from .io import load_csv
from .preprocessing import detrend_temperature_features
from .settings import TemperatureDetrendConfig


def prepare_dataset(
    ring_temp_csv: str,
    los_csv: str,
    sensors: list[str] | tuple[str, ...],
    ring_obmt_col: str,
    los_obmt_col: str,
    target_col: str,
    target_scale: float,
    trend_col: str | None = None,
    obmt_range: tuple[float, float] | None = None,
    temperature_detrend: TemperatureDetrendConfig | None = None,
) -> pd.DataFrame:
    ring_df = load_csv(ring_temp_csv)
    los_df = load_csv(los_csv)

    enriched = build_feature_table(
        ring_df=ring_df,
        los_df=los_df,
        ring_time_col=ring_obmt_col,
        los_time_col=los_obmt_col,
        sensor_cols=sensors,
    )

    feature_cols = [f"{sensor}_avg" for sensor in sensors]

    detrend_summary = pd.DataFrame()
    if temperature_detrend is not None and temperature_detrend.enabled:
        enriched, detrend_summary = detrend_temperature_features(
            enriched,
            feature_cols=feature_cols,
            obmt_col=los_obmt_col,
            method=temperature_detrend.method,
            keep_raw_columns=temperature_detrend.keep_raw_columns,
            preserve_mean=temperature_detrend.preserve_mean,
            switch_obmt=temperature_detrend.switch_obmt,
            fft_initial_window=temperature_detrend.fft_initial_window,
            anchor_obmt=temperature_detrend.anchor_obmt,
            forced_window_size=temperature_detrend.forced_window_size,
        )

    enriched["y_mas"] = pd.to_numeric(enriched[target_col], errors="coerce") * target_scale
    if trend_col and trend_col in enriched.columns:
        enriched["trend_scaled"] = pd.to_numeric(enriched[trend_col], errors="coerce") * target_scale

    if obmt_range is not None:
        lo, hi = obmt_range
        enriched = enriched[(enriched[los_obmt_col] > lo) & (enriched[los_obmt_col] < hi)].copy()

    return enriched.reset_index(drop=True), detrend_summary


def select_model_columns(
    enriched_df: pd.DataFrame,
    sensors: list[str] | tuple[str, ...],
    target_col: str = "y_mas",
) -> pd.DataFrame:
    feature_cols = [f"{sensor}_avg" for sensor in sensors]
    return enriched_df.dropna(subset=feature_cols + [target_col]).copy()

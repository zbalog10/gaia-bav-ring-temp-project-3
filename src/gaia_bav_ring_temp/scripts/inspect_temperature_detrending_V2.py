from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gaia_bav_ring_temp.dataset import prepare_dataset
from gaia_bav_ring_temp.io import ensure_dir, save_dataframe, save_json
from gaia_bav_ring_temp.settings import TemperatureDetrendConfig, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run temperature detrending and plot raw vs trend vs detrended features."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the experiment TOML config.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="linear",
        choices=["linear", "hybrid_rolling"],
        help="Detrending method to use.",
    )
    parser.add_argument(
        "--switch-obmt",
        type=float,
        default=None,
        help="OBMT switch point for hybrid_rolling detrending.",
    )
    parser.add_argument(
        "--fft-initial-window",
        type=int,
        default=50,
        help="Initial rolling-median window used before FFT period estimation.",
    )
    parser.add_argument(
        "--anchor-obmt",
        type=float,
        default=None,
        help="Reference OBMT used when converting the estimated period to a sample window size.",
    )
    parser.add_argument(
        "--forced-window-size",
        type=int,
        default=None,
        help="If set, override the FFT-derived rolling window size.",
    )
    parser.add_argument(
        "--out-subdir",
        type=str,
        default="temperature_detrending_inspection",
        help="Subdirectory under the experiment output directory.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot interactively in addition to saving it.",
    )
    return parser.parse_args()


def compute_feature_summary(
    df: pd.DataFrame,
    feature_cols: list[str],
    obmt_col: str,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []

    for feature_col in feature_cols:
        raw_col = f"{feature_col}_raw"
        trend_col = f"{feature_col}_trend"

        work = df[[obmt_col, raw_col, trend_col, feature_col]].dropna().copy()
        if work.empty:
            continue

        x = pd.to_numeric(work[obmt_col], errors="coerce").to_numpy(dtype=float)
        y_raw = pd.to_numeric(work[raw_col], errors="coerce").to_numpy(dtype=float)
        y_detrended = pd.to_numeric(work[feature_col], errors="coerce").to_numpy(dtype=float)

        valid = np.isfinite(x) & np.isfinite(y_raw) & np.isfinite(y_detrended)
        x = x[valid]
        y_raw = y_raw[valid]
        y_detrended = y_detrended[valid]

        if len(x) < 2:
            slope_raw = np.nan
            slope_detrended = np.nan
        else:
            x0 = np.mean(x)
            slope_raw, _ = np.polyfit(x - x0, y_raw, deg=1)
            slope_detrended, _ = np.polyfit(x - x0, y_detrended, deg=1)

        rows.append(
            {
                "feature": feature_col,
                "n_rows": int(len(x)),
                "raw_mean": float(np.mean(y_raw)) if len(y_raw) else np.nan,
                "raw_std": float(np.std(y_raw)) if len(y_raw) else np.nan,
                "detrended_mean": float(np.mean(y_detrended)) if len(y_detrended) else np.nan,
                "detrended_std": float(np.std(y_detrended)) if len(y_detrended) else np.nan,
                "raw_slope_per_obmt": float(slope_raw) if np.isfinite(slope_raw) else np.nan,
                "detrended_slope_per_obmt": float(slope_detrended) if np.isfinite(slope_detrended) else np.nan,
            }
        )

    return pd.DataFrame(rows)


def plot_detrending_comparison(
    df: pd.DataFrame,
    feature_cols: list[str],
    obmt_col: str,
    out_path: Path,
    show: bool = False,
) -> None:
    n = len(feature_cols)
    fig, axes = plt.subplots(
        n,
        1,
        figsize=(13, 3.8 * n),
        sharex=True,
        constrained_layout=True,
    )

    if n == 1:
        axes = [axes]

    for ax, feature_col in zip(axes, feature_cols):
        raw_col = f"{feature_col}_raw"
        trend_col = f"{feature_col}_trend"

        work = (
            df[[obmt_col, raw_col, trend_col, feature_col]]
            .dropna()
            .sort_values(obmt_col)
            .copy()
        )

        if work.empty:
            ax.set_title(f"{feature_col} (no valid data)")
            ax.grid(True, alpha=0.3)
            continue

        ax.plot(
            work[obmt_col],
            work[raw_col],
            label=f"{feature_col} raw averaged temperature",
            linewidth=1.0,
            alpha=0.85,
        )
        ax.plot(
            work[obmt_col],
            work[trend_col],
            label=f"{feature_col} fitted trend",
            linewidth=2.0,
        )
        ax.plot(
            work[obmt_col],
            work[feature_col],
            label=f"{feature_col} detrended",
            linewidth=1.0,
            alpha=0.9,
        )

        ax.set_title(feature_col)
        ax.set_ylabel("Temperature")
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel(obmt_col)
    fig.suptitle("Raw averaged temperatures, fitted trend, and detrended features", fontsize=14)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)


def main() -> None:
    args = parse_args()

    config = load_config(args.config)

    detrend_cfg = TemperatureDetrendConfig(
        enabled=True,
        method=args.method,
        keep_raw_columns=True,
        preserve_mean=True,
        switch_obmt=args.switch_obmt,
        fft_initial_window=args.fft_initial_window,
        anchor_obmt=args.anchor_obmt,
        forced_window_size=args.forced_window_size,
    )

    enriched, detrend_summary = prepare_dataset(
        ring_temp_csv=config.ring_temp_csv,
        los_csv=config.los_csv,
        sensors=config.sensors,
        ring_obmt_col=config.ring_obmt_col,
        los_obmt_col=config.los_obmt_col,
        target_col=config.target_col,
        target_scale=config.target_scale,
        trend_col=config.trend_col,
        obmt_range=config.obmt_range,
        temperature_detrend=detrend_cfg,
    )

    feature_cols = [f"{sensor}_avg" for sensor in config.sensors]

    out_dir = ensure_dir(Path(config.output_dir) / args.out_subdir)

    save_dataframe(enriched, out_dir / "enriched_dataset_with_detrending.csv", index=False)

    if detrend_summary is not None and not detrend_summary.empty:
        save_dataframe(detrend_summary, out_dir / "detrending_summary.csv", index=False)
    else:
        summary_df = compute_feature_summary(
            enriched,
            feature_cols=feature_cols,
            obmt_col=config.los_obmt_col,
        )
        save_dataframe(summary_df, out_dir / "detrending_summary.csv", index=False)

    plot_path = out_dir / "temperature_detrending_comparison.png"
    plot_detrending_comparison(
        enriched,
        feature_cols=feature_cols,
        obmt_col=config.los_obmt_col,
        out_path=plot_path,
        show=args.show,
    )

    summary_json = {
        "config": args.config,
        "method": args.method,
        "switch_obmt": args.switch_obmt,
        "fft_initial_window": args.fft_initial_window,
        "anchor_obmt": args.anchor_obmt,
        "forced_window_size": args.forced_window_size,
        "output_dir": str(out_dir),
        "plot_file": str(plot_path),
        "features": feature_cols,
        "n_rows": int(len(enriched)),
    }
    save_json(summary_json, out_dir / "run_summary.json")

    print("Finished temperature detrending inspection.")
    print(f"Output directory: {out_dir}")
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
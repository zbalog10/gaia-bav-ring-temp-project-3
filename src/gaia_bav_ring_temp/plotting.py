"""Plotting functions for diagnostic figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .constants import Layer


def _finalize(fig, path: str | Path | None) -> None:
    fig.tight_layout()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_predictions_vs_truth(pred: pd.DataFrame, path: str | Path | None = None) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(pred["y_true"], pred["y_pred"], s=10, alpha=0.7)
    lo = min(pred["y_true"].min(), pred["y_pred"].min())
    hi = max(pred["y_true"].max(), pred["y_pred"].max())
    ax.plot([lo, hi], [lo, hi], "r--")
    ax.set_xlabel(r"True $\delta\eta$ [mas]")
    ax.set_ylabel(r"Predicted $\delta\eta$ [mas]")
    ax.set_title("Model predictions vs true values")
    ax.grid(True)
    _finalize(fig, path)


def plot_residuals_vs_predicted(pred: pd.DataFrame, path: str | Path | None = None) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(pred["y_pred"], pred["residual"], alpha=0.5)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_xlabel(r"Predicted $\delta\eta$ [mas]")
    ax.set_ylabel("Residual [mas]")
    ax.set_title("Residuals vs predicted values")
    ax.grid(True)
    _finalize(fig, path)


def plot_residual_histogram(
    pred: pd.DataFrame,
    path: str | Path | None = None,
    bins: int = 100,
    xlim: tuple[float, float] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(pred["residual"], bins=bins, edgecolor="black", alpha=0.5)
    ax.set_xlabel("Residual [mas]")
    ax.set_ylabel("Frequency")
    ax.set_title("Residual distribution")
    if xlim:
        ax.set_xlim(*xlim)
    ax.grid(True)
    _finalize(fig, path)


def plot_residuals_vs_time(pred: pd.DataFrame, path: str | Path | None = None) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(pred["obmt"], pred["residual"], alpha=0.5)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_xlabel("OBMT [rev]")
    ax.set_ylabel("Residual [mas]")
    ax.set_title("Residuals over time")
    ax.grid(True)
    _finalize(fig, path)


def plot_feature_importance(feature_importance: pd.DataFrame, path: str | Path | None) -> None:
    if feature_importance.empty:
        return

    value_col = "importance"
    abs_col = "abs_importance" if "abs_importance" in feature_importance.columns else value_col
    label = "importance"
    if "importance_type" in feature_importance.columns and not feature_importance.empty:
        label = str(feature_importance["importance_type"].iloc[0]).replace("_", " ")

    ordered = feature_importance.sort_values(abs_col, ascending=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(ordered["feature"], ordered[value_col])
    ax.set_xlabel(label)
    ax.set_title(f"Feature effects ({label})")
    ax.grid(True, axis="x", alpha=0.3)
    _finalize(fig, path)


def plot_feature_correlation_heatmap(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    path: str | Path | None = None,
) -> None:
    cols = feature_cols + [target_col]
    corr = df[cols].corr().to_numpy()

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, interpolation="nearest", aspect="auto")
    ax.set_xticks(np.arange(len(cols)), labels=cols, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(cols)), labels=cols)

    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax)
    ax.set_title("Feature correlation matrix")
    _finalize(fig, path)


def plot_residuals_vs_features(
    X_test: pd.DataFrame,
    pred: pd.DataFrame,
    path: str | Path | None = None,
) -> None:
    ncols = len(X_test.columns)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))
    if ncols == 1:
        axes = [axes]

    for ax, feature in zip(axes, X_test.columns):
        ax.scatter(X_test[feature], pred["residual"], alpha=0.5)
        ax.axhline(0, color="red", linestyle="--")
        ax.set_xlabel(feature)
        ax.set_ylabel("Residual [mas]")
        ax.grid(True)

    fig.suptitle("Residuals vs sensor averages")
    _finalize(fig, path)


def plot_shap_vs_time(shap_df: pd.DataFrame, path: str | Path | None = None) -> None:
    feature_cols = [col for col in shap_df.columns if col != "obmt"]
    fig, ax = plt.subplots(figsize=(12, 6))
    for feature in feature_cols:
        ax.plot(shap_df["obmt"], shap_df[feature], label=feature, alpha=0.8)
    ax.set_xlabel("OBMT")
    ax.set_ylabel("SHAP value [mas]")
    ax.set_title("SHAP values over time")
    ax.legend()
    ax.grid(True)
    _finalize(fig, path)


def plot_shap_vs_feature(
    X_test: pd.DataFrame,
    shap_df: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for feature in X_test.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(X_test.reset_index(drop=True)[feature], shap_df[feature], alpha=0.5, edgecolors="k")
        ax.set_xlabel(f"{feature} value")
        ax.set_ylabel("SHAP value [mas]")
        ax.set_title(f"SHAP vs {feature}")
        ax.grid(True)
        _finalize(fig, out / f"shap_vs_{feature}.png")


def plot_layered_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    obmt_col: str,
    layers: list[Layer] | tuple[Layer, ...],
    x_label: str,
    y_label: str,
    title: str,
    path: str | Path | None = None,
    y_limits: tuple[float, float] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for layer in layers:
        mask = (df[obmt_col] > layer.start) & (df[obmt_col] < layer.end)
        ax.plot(
            df.loc[mask, x_col],
            df.loc[mask, y_col],
            linestyle="none",
            marker="o",
            markersize=2,
            color=layer.color,
            label=layer.label,
        )
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.set_title(title)
    if y_limits:
        ax.set_ylim(*y_limits)
    handles, labels = ax.get_legend_handles_labels()
    if any(labels):
        ax.legend(fontsize=10, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.grid(True)
    _finalize(fig, path)

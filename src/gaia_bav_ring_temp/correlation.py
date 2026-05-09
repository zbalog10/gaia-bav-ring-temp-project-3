"""Correlation and low-dimensional structure analysis for ring-temperature features.

This module is intentionally model-agnostic. It focuses on understanding how the
engineered temperature features relate to each other before interpreting feature
importance or writing paper claims about individual sensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .io import ensure_dir, save_dataframe, save_json


@dataclass(slots=True)
class PcaResult:
    """Principal-component decomposition for a feature matrix."""

    explained_variance_ratio: pd.DataFrame
    loadings: pd.DataFrame
    scores: pd.DataFrame


@dataclass(slots=True)
class CorrelationAnalysisResult:
    """Bundle of key correlation-analysis outputs."""

    raw_correlation: pd.DataFrame
    common_mode_removed_correlation: pd.DataFrame
    detrended_correlation: pd.DataFrame
    differenced_correlation: pd.DataFrame
    common_mode: pd.Series
    top_raw_pairs: pd.DataFrame
    vif: pd.DataFrame
    pca: PcaResult


_TRANSFORM_OPTIONS = {"raw", "common_mode_removed", "detrended", "first_difference"}


def infer_feature_columns(
    df: pd.DataFrame,
    suffix: str = "_avg",
    exclude: tuple[str, ...] = ("y_mas", "y_detrended_mas"),
) -> list[str]:
    """Infer engineered feature columns from a dataframe.

    By default, this picks columns ending in ``_avg`` and excludes the target
    columns that are not true inputs.
    """
    return [col for col in df.columns if col.endswith(suffix) and col not in exclude]


def compute_common_mode(feature_df: pd.DataFrame) -> pd.Series:
    """Return the row-wise mean temperature across all selected features."""
    return feature_df.mean(axis=1)


def remove_common_mode(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Subtract the row-wise common mode from every feature."""
    common_mode = compute_common_mode(feature_df)
    return feature_df.sub(common_mode, axis=0)


def detrend_features(feature_df: pd.DataFrame, obmt: pd.Series | np.ndarray) -> pd.DataFrame:
    """Remove a linear trend versus OBMT from each feature.

    The intent here is diagnostic, not to create the final modeling features.
    Linear detrending is sufficient for a first-pass audit of whether the strong
    correlations are driven mostly by mission-long drift.
    """
    x = pd.to_numeric(pd.Series(obmt), errors="coerce").to_numpy(dtype=float)
    out: dict[str, np.ndarray] = {}

    for col in feature_df.columns:
        y = pd.to_numeric(feature_df[col], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        resid = np.full_like(y, np.nan, dtype=float)
        if valid.sum() < 2:
            out[col] = resid
            continue
        slope, intercept = np.polyfit(x[valid], y[valid], deg=1)
        resid[valid] = y[valid] - (slope * x[valid] + intercept)
        out[col] = resid

    return pd.DataFrame(out, index=feature_df.index)


def difference_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Return first differences along time order."""
    return feature_df.diff()


def prepare_feature_matrix(
    df: pd.DataFrame,
    feature_cols: list[str],
    transform: str = "raw",
    obmt_col: str | None = None,
) -> pd.DataFrame:
    """Prepare a transformed feature matrix for correlation analysis.

    Parameters
    ----------
    df
        Full enriched dataset.
    feature_cols
        Names of engineered temperature features.
    transform
        One of: ``raw``, ``common_mode_removed``, ``detrended``,
        ``first_difference``.
    obmt_col
        Required for detrending and recommended for differencing so the rows are
        transformed in chronological order.
    """
    if transform not in _TRANSFORM_OPTIONS:
        raise ValueError(f"Unknown transform {transform!r}. Expected one of {_TRANSFORM_OPTIONS}.")

    work = df.copy()
    if obmt_col is not None:
        work = work.sort_values(obmt_col).reset_index(drop=True)

    feature_df = work[feature_cols].apply(pd.to_numeric, errors="coerce")

    if transform == "raw":
        return feature_df
    if transform == "common_mode_removed":
        return remove_common_mode(feature_df)
    if transform == "detrended":
        if obmt_col is None:
            raise ValueError("obmt_col is required for detrended transform.")
        return detrend_features(feature_df, work[obmt_col])
    if transform == "first_difference":
        return difference_features(feature_df)

    raise AssertionError("Unreachable")


def correlation_matrix(feature_df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """Compute the feature-feature correlation matrix."""
    return feature_df.corr(method=method)


def top_correlated_pairs(corr: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Return the strongest off-diagonal absolute-correlation pairs."""
    rows: list[dict[str, float | str]] = []
    cols = list(corr.columns)
    for i, col_i in enumerate(cols):
        for j in range(i + 1, len(cols)):
            col_j = cols[j]
            value = float(corr.iloc[i, j])
            rows.append(
                {
                    "feature_1": col_i,
                    "feature_2": col_j,
                    "correlation": value,
                    "abs_correlation": abs(value),
                }
            )
    pairs = pd.DataFrame(rows)
    if pairs.empty:
        return pairs
    return pairs.sort_values("abs_correlation", ascending=False).head(top_n).reset_index(drop=True)


def compute_vif(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Compute variance inflation factors for each feature.

    This implementation avoids an additional statsmodels dependency. For each
    feature, it regresses that feature on all remaining features and converts the
    resulting :math:`R^2` into ``VIF = 1 / (1 - R^2)``.
    """
    clean = feature_df.dropna(axis=0, how="any")
    X = clean.to_numpy(dtype=float)
    cols = list(clean.columns)

    if X.size == 0:
        return pd.DataFrame(columns=["feature", "vif", "r2_auxiliary"])

    out: list[dict[str, float | str]] = []
    n_features = X.shape[1]
    for i, feature in enumerate(cols):
        y = X[:, i]
        if n_features == 1:
            out.append({"feature": feature, "vif": 1.0, "r2_auxiliary": 0.0})
            continue

        X_other = np.delete(X, i, axis=1)
        design = np.column_stack([np.ones(len(X_other)), X_other])
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        y_hat = design @ beta

        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))

        if np.isclose(ss_tot, 0.0):
            r2 = 1.0
        else:
            r2 = 1.0 - ss_res / ss_tot
            r2 = min(max(r2, 0.0), 1.0)

        vif = np.inf if r2 >= 1.0 - 1e-12 else 1.0 / (1.0 - r2)
        out.append({"feature": feature, "vif": float(vif), "r2_auxiliary": float(r2)})

    vif_df = pd.DataFrame(out)
    return vif_df.sort_values("vif", ascending=False, na_position="last").reset_index(drop=True)


def compute_pca(feature_df: pd.DataFrame, n_components: int | None = None) -> PcaResult:
    """Fit PCA on standardized features and return explained variance, loadings, and scores."""
    clean = feature_df.dropna(axis=0, how="any")
    if clean.empty:
        empty = pd.DataFrame()
        return PcaResult(explained_variance_ratio=empty, loadings=empty, scores=empty)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(clean.to_numpy(dtype=float))

    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X_scaled)

    component_names = [f"PC{i+1}" for i in range(pca.n_components_)]
    evr = pd.DataFrame(
        {
            "component": component_names,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    loadings = pd.DataFrame(pca.components_.T, index=clean.columns, columns=component_names)
    scores_df = pd.DataFrame(scores, index=clean.index, columns=component_names)

    return PcaResult(
        explained_variance_ratio=evr,
        loadings=loadings,
        scores=scores_df,
    )


def _plot_heatmap(corr: pd.DataFrame, title: str, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.to_numpy(), interpolation="nearest", aspect="auto")
    ax.set_xticks(np.arange(len(corr.columns)), labels=corr.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(corr.index)), labels=corr.index)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_pca_variance(evr: pd.DataFrame, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(1, len(evr) + 1)
    ax.bar(x, evr["explained_variance_ratio"].to_numpy(), alpha=0.8)
    ax.plot(x, evr["cumulative_explained_variance_ratio"].to_numpy(), marker="o")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("PCA explained variance")
    ax.set_xticks(x)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_correlation_analysis(
    df: pd.DataFrame,
    feature_cols: list[str],
    obmt_col: str,
    out_dir: str | Path | None = None,
    top_n_pairs: int = 10,
) -> CorrelationAnalysisResult:
    """Run a first-pass correlation audit and optionally save outputs.

    The saved outputs are intentionally paper-friendly and easy to inspect:
    correlation matrices for several transforms, strongest raw pairs, VIF, and
    PCA summaries.
    """
    raw_features = prepare_feature_matrix(df, feature_cols, transform="raw", obmt_col=obmt_col)
    common_mode = compute_common_mode(raw_features)
    common_mode_removed = remove_common_mode(raw_features)
    detrended = prepare_feature_matrix(df, feature_cols, transform="detrended", obmt_col=obmt_col)
    differenced = prepare_feature_matrix(df, feature_cols, transform="first_difference", obmt_col=obmt_col)

    raw_corr = correlation_matrix(raw_features)
    common_removed_corr = correlation_matrix(common_mode_removed)
    detrended_corr = correlation_matrix(detrended)
    differenced_corr = correlation_matrix(differenced)
    pairs = top_correlated_pairs(raw_corr, top_n=top_n_pairs)
    vif_df = compute_vif(raw_features)
    pca = compute_pca(raw_features)

    result = CorrelationAnalysisResult(
        raw_correlation=raw_corr,
        common_mode_removed_correlation=common_removed_corr,
        detrended_correlation=detrended_corr,
        differenced_correlation=differenced_corr,
        common_mode=common_mode.rename("common_mode"),
        top_raw_pairs=pairs,
        vif=vif_df,
        pca=pca,
    )

    if out_dir is not None:
        out = ensure_dir(out_dir)

        save_dataframe(raw_corr, out / "correlation_raw.csv", index=True)
        save_dataframe(common_removed_corr, out / "correlation_common_mode_removed.csv", index=True)
        save_dataframe(detrended_corr, out / "correlation_detrended.csv", index=True)
        save_dataframe(differenced_corr, out / "correlation_first_difference.csv", index=True)
        save_dataframe(result.common_mode.to_frame(), out / "common_mode.csv", index=True)
        save_dataframe(pairs, out / "top_raw_correlated_pairs.csv", index=False)
        save_dataframe(vif_df, out / "vif.csv", index=False)
        save_dataframe(pca.explained_variance_ratio, out / "pca_explained_variance.csv", index=False)
        save_dataframe(pca.loadings, out / "pca_loadings.csv", index=True)
        save_dataframe(pca.scores, out / "pca_scores.csv", index=True)

        summary = {
            "n_rows": int(len(df)),
            "n_feature_rows_complete": int(raw_features.dropna(axis=0, how="any").shape[0]),
            "n_features": int(len(feature_cols)),
            "feature_columns": feature_cols,
            "top_absolute_raw_correlation": None if pairs.empty else float(pairs.iloc[0]["abs_correlation"]),
            "pc1_explained_variance_ratio": None
            if pca.explained_variance_ratio.empty
            else float(pca.explained_variance_ratio.iloc[0]["explained_variance_ratio"]),
            "pcs_for_90_percent": None,
        }
        if not pca.explained_variance_ratio.empty:
            cumulative = pca.explained_variance_ratio["cumulative_explained_variance_ratio"].to_numpy()
            summary["pcs_for_90_percent"] = int(np.searchsorted(cumulative, 0.90) + 1)
        save_json(summary, out / "correlation_summary.json")

        _plot_heatmap(raw_corr, "Raw feature correlation", out / "correlation_raw.png")
        _plot_heatmap(
            common_removed_corr,
            "Feature correlation after common-mode removal",
            out / "correlation_common_mode_removed.png",
        )
        _plot_heatmap(detrended_corr, "Feature correlation after detrending", out / "correlation_detrended.png")
        _plot_heatmap(
            differenced_corr,
            "Feature correlation of first differences",
            out / "correlation_first_difference.png",
        )
        if not pca.explained_variance_ratio.empty:
            _plot_pca_variance(pca.explained_variance_ratio, out / "pca_explained_variance.png")

    return result

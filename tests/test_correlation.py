from __future__ import annotations

import math

import numpy as np
import pandas as pd

from gaia_bav_ring_temp.correlation import (
    compute_common_mode,
    compute_pca,
    compute_vif,
    infer_feature_columns,
    prepare_feature_matrix,
    run_correlation_analysis,
    top_correlated_pairs,
)


def make_demo_df() -> pd.DataFrame:
    x = np.arange(6, dtype=float)
    return pd.DataFrame(
        {
            "obmtRev": x,
            "A_avg": x,
            "B_avg": 2.0 * x,
            "C_avg": np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
            "y_mas": 3.0 * x,
        }
    )


def test_infer_feature_columns_uses_avg_suffix() -> None:
    df = make_demo_df()
    assert infer_feature_columns(df) == ["A_avg", "B_avg", "C_avg"]


def test_common_mode_removal_preserves_zero_row_mean() -> None:
    df = make_demo_df()
    features = prepare_feature_matrix(df, ["A_avg", "B_avg", "C_avg"], transform="common_mode_removed")
    row_means = features.mean(axis=1)
    assert np.allclose(row_means.to_numpy(), 0.0)


def test_top_correlated_pairs_returns_ab_pair_first() -> None:
    df = make_demo_df()
    raw = prepare_feature_matrix(df, ["A_avg", "B_avg", "C_avg"], transform="raw")
    pairs = top_correlated_pairs(raw.corr(), top_n=1)
    assert pairs.iloc[0]["feature_1"] == "A_avg"
    assert pairs.iloc[0]["feature_2"] == "B_avg"
    assert math.isclose(float(pairs.iloc[0]["abs_correlation"]), 1.0)


def test_vif_detects_perfect_collinearity() -> None:
    df = make_demo_df()
    vif = compute_vif(df[["A_avg", "B_avg", "C_avg"]])
    ab = vif[vif["feature"].isin(["A_avg", "B_avg"])]
    assert np.isinf(ab["vif"]).all()


def test_pca_returns_explained_variance_table() -> None:
    df = make_demo_df()
    pca = compute_pca(df[["A_avg", "B_avg", "C_avg"]])
    assert not pca.explained_variance_ratio.empty
    assert list(pca.explained_variance_ratio.columns) == [
        "component",
        "explained_variance_ratio",
        "cumulative_explained_variance_ratio",
    ]


def test_run_correlation_analysis_produces_key_outputs() -> None:
    df = make_demo_df()
    result = run_correlation_analysis(df, ["A_avg", "B_avg", "C_avg"], obmt_col="obmtRev")
    assert result.raw_correlation.shape == (3, 3)
    assert result.common_mode.shape[0] == len(df)
    assert not result.vif.empty
    assert not result.pca.explained_variance_ratio.empty
    assert "A_avg" in result.raw_correlation.columns
    assert "B_avg" in result.common_mode_removed_correlation.columns
    assert compute_common_mode(df[["A_avg", "B_avg", "C_avg"]]).shape[0] == len(df)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

@dataclass(slots=True)
class ThermalResponseFit:
    model_name: str
    taus: dict[str, float]
    intercept: float
    coefficients: pd.DataFrame
    metrics: dict[str, float]
    predictions: pd.DataFrame
    diagnostics: pd.DataFrame


def exponential_filter_irregular(
    obmt: np.ndarray,
    values: np.ndarray,
    tau: float,
) -> np.ndarray:
    """Causal first-order exponential response for irregular time sampling.

    The continuous model is equivalent to

        dz/dt = (T - z) / tau

    with solution

        z_k = alpha_k z_{k-1} + (1 - alpha_k) T_k

    where

        alpha_k = exp(-(t_k - t_{k-1}) / tau).

    Parameters
    ----------
    obmt
        Time coordinate, in OBMT revolutions.
    values
        Temperature feature values.
    tau
        Exponential response time scale, also in OBMT revolutions.

    Returns
    -------
    np.ndarray
        Filtered temperature response.
    """
    if tau <= 0 or not np.isfinite(tau):
        raise ValueError(f"tau must be positive and finite, got {tau}")

    t = np.asarray(obmt, dtype=float)
    x = np.asarray(values, dtype=float)

    if len(t) != len(x):
        raise ValueError("obmt and values must have the same length.")
    if len(t) == 0:
        return np.array([], dtype=float)

    z = np.empty_like(x, dtype=float)

    # Start from the first value. Since the temperature features are detrended
    # in most experiments, this is usually close to a zero-response initial state.
    z[0] = x[0]

    for k in range(1, len(x)):
        dt = t[k] - t[k - 1]
        if not np.isfinite(dt) or dt < 0:
            raise ValueError("OBMT values must be finite and sorted increasingly.")

        alpha = np.exp(-dt / tau)
        z[k] = alpha * z[k - 1] + (1.0 - alpha) * x[k]

    return z


def make_tau_grid(
    tau_min: float = 0.02,
    tau_max: float = 20.0,
    n_grid: int = 1000,
) -> np.ndarray:
    """Create a logarithmic tau grid in OBMT revolutions."""
    if tau_min <= 0 or tau_max <= tau_min:
        raise ValueError("Require 0 < tau_min < tau_max.")
    if n_grid < 2:
        raise ValueError("n_grid must be at least 2.")

    return np.logspace(np.log10(tau_min), np.log10(tau_max), n_grid)


def build_filtered_matrix(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    obmt_col: str,
    taus: dict[str, float] | float,
) -> pd.DataFrame:
    """Build exponentially filtered temperature features.

    Parameters
    ----------
    df
        Input dataframe sorted by OBMT.
    feature_cols
        Temperature feature columns.
    obmt_col
        OBMT column.
    taus
        Either a single common tau or a dictionary mapping feature column to tau.

    Returns
    -------
    pd.DataFrame
        Filtered features, one column per input feature.
    """
    obmt = df[obmt_col].to_numpy(dtype=float)

    out = pd.DataFrame(index=df.index)

    for col in feature_cols:
        if isinstance(taus, dict):
            tau = float(taus[col])
        else:
            tau = float(taus)

        out[f"{col}_exp_tau"] = exponential_filter_irregular(
            obmt,
            df[col].to_numpy(dtype=float),
            tau=tau,
        )

    return out


def _fit_linear_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    alpha: float = 0.0,
):
    """Fit either ordinary linear regression or ridge regression."""
    if alpha > 0:
        model = Ridge(alpha=alpha)
    else:
        model = LinearRegression()

    model.fit(X_train, y_train)
    return model


def evaluate_prediction(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "n_test": int(len(y_true)),
    }


def fit_for_fixed_taus(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    obmt_col: str,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    taus: dict[str, float] | float,
    model_name: str,
    alpha: float = 0.0,
) -> ThermalResponseFit:
    """Fit the linear response amplitudes for fixed exponential time constants."""
    X = build_filtered_matrix(df, feature_cols, obmt_col, taus)
    y = df[target_col]

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    model = _fit_linear_model(X_train, y_train, alpha=alpha)

    y_pred = model.predict(X_test)
    metrics = evaluate_prediction(y_test, y_pred)

    pred_df = pd.DataFrame(
        {
            "obmt": df.iloc[test_idx][obmt_col].to_numpy(dtype=float),
            "y_true": y_test.to_numpy(dtype=float),
            "y_pred": y_pred,
            "residual": y_test.to_numpy(dtype=float) - y_pred,
        }
    ).sort_values("obmt").reset_index(drop=True)

    coef_df = pd.DataFrame(
        {
            "feature": list(X.columns),
            "coefficient": np.ravel(model.coef_),
        }
    )

    if isinstance(taus, dict):
        tau_dict = {col: float(taus[col]) for col in feature_cols}
    else:
        tau_dict = {col: float(taus) for col in feature_cols}

    diagnostics = pd.DataFrame()

    return ThermalResponseFit(
        model_name=model_name,
        taus=tau_dict,
        intercept=float(model.intercept_),
        coefficients=coef_df,
        metrics=metrics,
        predictions=pred_df,
        diagnostics=diagnostics,
    )


def fit_instantaneous_linear(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    obmt_col: str,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    alpha: float = 0.0,
) -> ThermalResponseFit:
    """Fit the instantaneous linear baseline without exponential memory."""
    X = df[list(feature_cols)]
    y = df[target_col]

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    model = _fit_linear_model(X_train, y_train, alpha=alpha)
    y_pred = model.predict(X_test)

    metrics = evaluate_prediction(y_test, y_pred)

    pred_df = pd.DataFrame(
        {
            "obmt": df.iloc[test_idx][obmt_col].to_numpy(dtype=float),
            "y_true": y_test.to_numpy(dtype=float),
            "y_pred": y_pred,
            "residual": y_test.to_numpy(dtype=float) - y_pred,
        }
    ).sort_values("obmt").reset_index(drop=True)

    coef_df = pd.DataFrame(
        {
            "feature": list(feature_cols),
            "coefficient": np.ravel(model.coef_),
        }
    )

    return ThermalResponseFit(
        model_name="instantaneous_linear",
        taus={col: 0.0 for col in feature_cols},
        intercept=float(model.intercept_),
        coefficients=coef_df,
        metrics=metrics,
        predictions=pred_df,
        diagnostics=pd.DataFrame(),
    )


def fit_common_tau_grid(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    obmt_col: str,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    tau_grid: np.ndarray,
    alpha: float = 0.0,
) -> ThermalResponseFit:
    """Fit one common tau by grid search on the training set."""
    rows: list[dict[str, float]] = []

    y_train = df.iloc[train_idx][target_col]

    best_tau = None
    best_train_mse = np.inf

    for tau in tau_grid:
        X = build_filtered_matrix(df, feature_cols, obmt_col, float(tau))
        X_train = X.iloc[train_idx]

        model = _fit_linear_model(X_train, y_train, alpha=alpha)
        train_pred = model.predict(X_train)
        train_mse = mean_squared_error(y_train, train_pred)
        train_rmse = np.sqrt(train_mse)
        train_r2 = r2_score(y_train, train_pred)

        rows.append(
            {
                "tau": float(tau),
                "train_mse": float(train_mse),
                "train_rmse": float(train_rmse),
                "train_r2": float(train_r2),
            }
        )

        if train_mse < best_train_mse:
            best_train_mse = train_mse
            best_tau = float(tau)

    result = fit_for_fixed_taus(
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        obmt_col=obmt_col,
        train_idx=train_idx,
        test_idx=test_idx,
        taus=float(best_tau),
        model_name="exponential_common_tau",
        alpha=alpha,
    )
    result.diagnostics = pd.DataFrame(rows)
    return result


def fit_separate_taus_coordinate_grid(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    obmt_col: str,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    tau_grid: np.ndarray,
    initial_tau: float,
    max_iter: int = 6,
    alpha: float = 0.0,
    rel_tol: float = 1.0e-5,
) -> ThermalResponseFit:
    """Fit separate tau values using simple coordinate grid search.

    This avoids adding scipy as a dependency. Each iteration scans the tau grid
    for one feature at a time while holding the other tau values fixed.
    """
    current = {col: float(initial_tau) for col in feature_cols}

    y_train = df.iloc[train_idx][target_col]

    def train_mse_for(taus: dict[str, float]) -> float:
        X = build_filtered_matrix(df, feature_cols, obmt_col, taus)
        X_train = X.iloc[train_idx]
        model = _fit_linear_model(X_train, y_train, alpha=alpha)
        pred = model.predict(X_train)
        return float(mean_squared_error(y_train, pred))

    history: list[dict[str, float | str | int]] = []

    previous_best = train_mse_for(current)

    for iteration in range(max_iter):
        improved = False

        for col in feature_cols:
            best_tau_for_col = current[col]
            best_mse_for_col = previous_best

            for tau in tau_grid:
                trial = dict(current)
                trial[col] = float(tau)
                mse = train_mse_for(trial)

                history.append(
                    {
                        "iteration": iteration,
                        "feature": col,
                        "trial_tau": float(tau),
                        "train_mse": float(mse),
                    }
                )

                if mse < best_mse_for_col:
                    best_mse_for_col = mse
                    best_tau_for_col = float(tau)

            if best_mse_for_col < previous_best * (1.0 - rel_tol):
                improved = True

            current[col] = best_tau_for_col
            previous_best = best_mse_for_col

        if not improved:
            break

    result = fit_for_fixed_taus(
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        obmt_col=obmt_col,
        train_idx=train_idx,
        test_idx=test_idx,
        taus=current,
        model_name="exponential_separate_tau",
        alpha=alpha,
    )
    result.diagnostics = pd.DataFrame(history)
    return result


def fit_summary_row(result: ThermalResponseFit) -> dict[str, float | str]:
    row: dict[str, float | str] = {
        "model": result.model_name,
        "intercept": result.intercept,
        **result.metrics,
    }

    for feature, tau in result.taus.items():
        row[f"tau_{feature}"] = tau

    for _, coef_row in result.coefficients.iterrows():
        row[f"coef_{coef_row['feature']}"] = float(coef_row["coefficient"])

    return row
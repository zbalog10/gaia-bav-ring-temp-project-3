"""Model evaluation and optional SHAP helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline


@dataclass(slots=True)
class EvalResult:
    metrics: dict[str, float]
    predictions: pd.DataFrame
    feature_importance: pd.DataFrame
    shap_values: pd.DataFrame | None = None


def _unwrap_model(model):
    if isinstance(model, Pipeline):
        return model.named_steps["model"]
    return model


def _extract_feature_effects(model, feature_names: list[str]) -> pd.DataFrame:
    base_model = _unwrap_model(model)

    if hasattr(base_model, "feature_importances_"):
        values = np.asarray(base_model.feature_importances_, dtype=float)
        result = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": values,
                "abs_importance": np.abs(values),
                "importance_type": "native_importance",
            }
        )
        return result.sort_values("abs_importance", ascending=False).reset_index(drop=True)

    if hasattr(base_model, "coef_"):
        values = np.ravel(np.asarray(base_model.coef_, dtype=float))
        result = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": values,
                "abs_importance": np.abs(values),
                "importance_type": "coefficient",
            }
        )
        return result.sort_values("abs_importance", ascending=False).reset_index(drop=True)

    return pd.DataFrame(
        columns=["feature", "importance", "abs_importance", "importance_type"]
    )


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    obmt_test: pd.Series,
) -> EvalResult:
    y_pred = model.predict(X_test)
    residuals = y_test.to_numpy() - y_pred

    metrics = {
        "mse": float(mean_squared_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
        "n_test": int(len(y_test)),
    }

    predictions = pd.DataFrame(
        {
            "obmt": obmt_test.to_numpy(),
            "y_true": y_test.to_numpy(),
            "y_pred": y_pred,
            "residual": residuals,
        }
    ).sort_values("obmt")

    feature_importance = _extract_feature_effects(model, list(X_test.columns))

    return EvalResult(
        metrics=metrics,
        predictions=predictions.reset_index(drop=True),
        feature_importance=feature_importance.reset_index(drop=True),
        shap_values=None,
    )


def compute_shap_values(model, X_test: pd.DataFrame, obmt_test: pd.Series) -> pd.DataFrame:
    import shap  # type: ignore

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap_df = pd.DataFrame(shap_values, columns=X_test.columns)
    shap_df["obmt"] = obmt_test.reset_index(drop=True)
    return shap_df.sort_values("obmt").reset_index(drop=True)

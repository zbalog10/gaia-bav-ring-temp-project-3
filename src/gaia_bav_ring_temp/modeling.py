"""Model training helpers."""

from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .settings import ExperimentConfig
from .validation import SplitData

SUPPORTED_MODELS = ("random_forest", "linear", "ridge", "mlp")


def get_enabled_models(config: ExperimentConfig) -> list[str]:
    models = list(config.models.enabled)
    unknown = [m for m in models if m not in SUPPORTED_MODELS]
    if unknown:
        raise ValueError(
            f"Unsupported model(s): {unknown}. Supported models are: {SUPPORTED_MODELS}"
        )
    return models


def build_model(model_name: str, config: ExperimentConfig):
    if model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=config.rf.n_estimators,
            random_state=config.rf.model_random_state,
            n_jobs=config.rf.n_jobs,
        )

    if model_name == "linear":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        )

    if model_name == "ridge":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=config.ridge.alpha)),
            ]
        )

    if model_name == "mlp":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=config.mlp.hidden_layer_sizes,
                        activation=config.mlp.activation,
                        alpha=config.mlp.alpha,
                        learning_rate_init=config.mlp.learning_rate_init,
                        max_iter=config.mlp.max_iter,
                        early_stopping=config.mlp.early_stopping,
                        validation_fraction=config.mlp.validation_fraction,
                        n_iter_no_change=config.mlp.n_iter_no_change,
                        random_state=config.mlp.random_state,
                    ),
                ),
            ]
        )

    raise ValueError(f"Unknown model name: {model_name}")


def train_model(split: SplitData, model_name: str, config: ExperimentConfig):
    model = build_model(model_name, config)
    model.fit(split.X_train, split.y_train)
    return model

from __future__ import annotations

import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from gaia_bav_ring_temp.dataset import prepare_dataset, select_model_columns
from gaia_bav_ring_temp.io import ensure_dir, save_dataframe, save_json
from gaia_bav_ring_temp.settings import load_config
from gaia_bav_ring_temp.validation import make_random_split


SUPPORTED_MODELS = ("linear", "ridge", "random_forest", "mlp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare raw-feature regressors with PCA-feature regressors."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the experiment TOML config.",
    )
    parser.add_argument(
        "--components",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="List of PCA component counts to test. Default: 1 2 3",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["linear", "ridge", "random_forest", "mlp"],
        help="Models to test. Choices: linear ridge random_forest mlp",
    )
    parser.add_argument(
        "--out-subdir",
        type=str,
        default="pca_model_comparison",
        help="Subdirectory under the experiment output directory.",
    )
    return parser.parse_args()


def build_raw_model(model_name: str, config):
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

    if model_name == "random_forest":
        # RF does not require scaling, but keeping a Pipeline interface makes
        # the raw/PCA experiments easier to compare.
        return Pipeline(
            steps=[
                ("model", RandomForestRegressor(
                    n_estimators=config.rf.n_estimators,
                    random_state=config.rf.model_random_state,
                    n_jobs=config.rf.n_jobs,
                )),
            ]
        )

    if model_name == "mlp":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", MLPRegressor(
                    hidden_layer_sizes=config.mlp.hidden_layer_sizes,
                    activation=config.mlp.activation,
                    alpha=config.mlp.alpha,
                    learning_rate_init=config.mlp.learning_rate_init,
                    max_iter=config.mlp.max_iter,
                    early_stopping=config.mlp.early_stopping,
                    validation_fraction=config.mlp.validation_fraction,
                    n_iter_no_change=config.mlp.n_iter_no_change,
                    random_state=config.mlp.random_state,
                )),
            ]
        )

    raise ValueError(f"Unsupported model name: {model_name}")


def build_pca_model(model_name: str, config, n_components: int):
    if model_name == "linear":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=n_components)),
                ("model", LinearRegression()),
            ]
        )

    if model_name == "ridge":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=n_components)),
                ("model", Ridge(alpha=config.ridge.alpha)),
            ]
        )

    if model_name == "random_forest":
        # PCA is applied to standardized inputs, then the PCs are used by RF.
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=n_components)),
                ("model", RandomForestRegressor(
                    n_estimators=config.rf.n_estimators,
                    random_state=config.rf.model_random_state,
                    n_jobs=config.rf.n_jobs,
                )),
            ]
        )

    if model_name == "mlp":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=n_components)),
                ("model", MLPRegressor(
                    hidden_layer_sizes=config.mlp.hidden_layer_sizes,
                    activation=config.mlp.activation,
                    alpha=config.mlp.alpha,
                    learning_rate_init=config.mlp.learning_rate_init,
                    max_iter=config.mlp.max_iter,
                    early_stopping=config.mlp.early_stopping,
                    validation_fraction=config.mlp.validation_fraction,
                    n_iter_no_change=config.mlp.n_iter_no_change,
                    random_state=config.mlp.random_state,
                )),
            ]
        )

    raise ValueError(f"Unsupported model name: {model_name}")


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "mse": float(mse),
        "rmse": rmse,
        "mae": float(mae),
        "r2": float(r2),
        "n_test": int(len(y_true)),
    }


def extract_pca_info(model: Pipeline) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[None, None]:
    if "pca" not in model.named_steps:
        return None, None

    pca = model.named_steps["pca"]

    explained = pd.DataFrame(
        {
            "component": np.arange(1, len(pca.explained_variance_ratio_) + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
        }
    )

    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i}" for i in range(1, pca.n_components_ + 1)],
    )

    return explained, loadings


def main() -> None:
    args = parse_args()

    unknown_models = [m for m in args.models if m not in SUPPORTED_MODELS]
    if unknown_models:
        raise ValueError(
            f"Unsupported models requested: {unknown_models}. "
            f"Supported models are: {SUPPORTED_MODELS}"
        )

    config = load_config(args.config)

    # Build the same dataset used by the main project.
    enriched = prepare_dataset(
        ring_temp_csv=config.ring_temp_csv,
        los_csv=config.los_csv,
        sensors=config.sensors,
        ring_obmt_col=config.ring_obmt_col,
        los_obmt_col=config.los_obmt_col,
        target_col=config.target_col,
        target_scale=config.target_scale,
        trend_col=config.trend_col,
        obmt_range=config.obmt_range,
        temperature_detrend=config.temperature_detrend,
    )

    model_df = select_model_columns(
        enriched,
        sensors=config.sensors,
        target_col="y_mas",
    )

    split = make_random_split(
        model_df,
        feature_cols=[f"{sensor}_avg" for sensor in config.sensors],
        target_col="y_mas",
        obmt_col=config.los_obmt_col,
        test_size=config.rf.test_size,
        random_state=config.rf.split_random_state,
    )

    out_dir = ensure_dir(Path(config.output_dir) / args.out_subdir)

    summary_rows: list[dict[str, object]] = []

    # Save the exact test set used, so the comparison is fully reproducible.
    test_reference = pd.DataFrame(
        {
            "obmt": split.obmt_test,
            "y_true": split.y_test,
        }
    ).reset_index(drop=True)
    save_dataframe(test_reference, out_dir / "test_reference.csv", index=False)

    # --- RAW FEATURE MODELS -------------------------------------------------
    raw_dir = ensure_dir(out_dir / "raw")

    for model_name in args.models:
        model = build_raw_model(model_name, config)
        model.fit(split.X_train, split.y_train)
        y_pred = model.predict(split.X_test)

        metrics = evaluate_predictions(split.y_test, y_pred)

        row = {
            "feature_space": "raw",
            "model": model_name,
            "n_components": len(split.X_train.columns),
            **metrics,
        }
        summary_rows.append(row)

        pred_df = pd.DataFrame(
            {
                "obmt": split.obmt_test,
                "y_true": split.y_test,
                "y_pred": y_pred,
                "residual": split.y_test.to_numpy() - y_pred,
            }
        ).sort_values("obmt").reset_index(drop=True)

        model_out = ensure_dir(raw_dir / model_name)
        save_dataframe(pred_df, model_out / "predictions.csv", index=False)
        save_json(row, model_out / "metrics.json")

    # --- PCA FEATURE MODELS -------------------------------------------------
    pca_root = ensure_dir(out_dir / "pca")

    max_allowed = len(split.X_train.columns)
    component_list = sorted(set(int(c) for c in args.components if 1 <= int(c) <= max_allowed))
    if not component_list:
        raise ValueError(
            f"No valid PCA component counts. With {max_allowed} input features, "
            f"components must be between 1 and {max_allowed}."
        )

    for n_components in component_list:
        comp_dir = ensure_dir(pca_root / f"pc{n_components}")

        for model_name in args.models:
            model = build_pca_model(model_name, config, n_components=n_components)
            model.fit(split.X_train, split.y_train)
            y_pred = model.predict(split.X_test)

            metrics = evaluate_predictions(split.y_test, y_pred)

            row = {
                "feature_space": "pca",
                "model": model_name,
                "n_components": n_components,
                **metrics,
            }
            summary_rows.append(row)

            pred_df = pd.DataFrame(
                {
                    "obmt": split.obmt_test,
                    "y_true": split.y_test,
                    "y_pred": y_pred,
                    "residual": split.y_test.to_numpy() - y_pred,
                }
            ).sort_values("obmt").reset_index(drop=True)

            model_out = ensure_dir(comp_dir / model_name)
            save_dataframe(pred_df, model_out / "predictions.csv", index=False)
            save_json(row, model_out / "metrics.json")

            explained, loadings = extract_pca_info(model)
            if explained is not None:
                save_dataframe(explained, model_out / "pca_explained_variance.csv", index=False)

                loadings.index = split.X_train.columns
                save_dataframe(loadings, model_out / "pca_loadings.csv", index=True)

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["feature_space", "n_components", "model"]
    ).reset_index(drop=True)

    save_dataframe(summary_df, out_dir / "pca_vs_raw_comparison.csv", index=False)
    save_json(summary_rows, out_dir / "pca_vs_raw_comparison.json")

    # Small convenience summary: best model per feature-space setup by RMSE
    best_rows = (
        summary_df.sort_values(["feature_space", "n_components", "rmse"])
        .groupby(["feature_space", "n_components"], as_index=False)
        .first()
    )
    save_dataframe(best_rows, out_dir / "best_models_by_setup.csv", index=False)

    print("Finished PCA vs raw comparison.")
    print(f"Output directory: {out_dir}")
    print()
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
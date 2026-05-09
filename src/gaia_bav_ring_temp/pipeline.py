"""End-to-end experiment pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .constants import DEFAULT_LAYERS
from .dataset import prepare_dataset, select_model_columns
from .evaluation import compute_shap_values, evaluate_model
from .io import ensure_dir, save_dataframe, save_json, save_model
from .modeling import get_enabled_models, train_model
from .plotting import (
    plot_feature_correlation_heatmap,
    plot_feature_importance,
    plot_layered_scatter,
    plot_predictions_vs_truth,
    plot_residual_histogram,
    plot_residuals_vs_features,
    plot_residuals_vs_predicted,
    plot_residuals_vs_time,
    plot_shap_vs_feature,
    plot_shap_vs_time,
)
from .settings import ExperimentConfig
from .validation import ValidationRun, build_validation_runs


def _save_validation_outputs(
    *,
    run: ValidationRun,
    model_name: str,
    model,
    eval_result,
    out_dir: Path,
    make_plots: bool,
    compute_shap: bool,
    save_model_flag: bool,
) -> dict[str, object]:
    run_dir = ensure_dir(out_dir)
    plots_dir = ensure_dir(run_dir / "plots")

    metrics_with_split = dict(eval_result.metrics)
    metrics_with_split["validation_mode"] = run.summary.mode
    metrics_with_split["model"] = model_name
    metrics_with_split["n_train"] = run.summary.n_train
    metrics_with_split["gap_size"] = run.summary.gap_size
    metrics_with_split["train_obmt_min"] = run.summary.train_obmt_min
    metrics_with_split["train_obmt_max"] = run.summary.train_obmt_max
    metrics_with_split["test_obmt_min"] = run.summary.test_obmt_min
    metrics_with_split["test_obmt_max"] = run.summary.test_obmt_max

    save_json(run.summary.to_dict(), run_dir / "split_summary.json")
    save_json(metrics_with_split, run_dir / "metrics.json")
    save_dataframe(eval_result.predictions, run_dir / "predictions.csv")
    if not eval_result.feature_importance.empty:
        save_dataframe(eval_result.feature_importance, run_dir / "feature_importance.csv")

    if save_model_flag:
        save_model(model, run_dir / "model.joblib")

    shap_df = None
    if compute_shap and model_name == "random_forest":
        shap_df = compute_shap_values(model=model, X_test=run.split.X_test, obmt_test=run.split.obmt_test)
        save_dataframe(shap_df, run_dir / "shap_values.csv")
        if make_plots:
            plot_shap_vs_time(shap_df, plots_dir / "shap_vs_time.png")
            plot_shap_vs_feature(run.split.X_test, shap_df, plots_dir)

    if make_plots:
        plot_predictions_vs_truth(eval_result.predictions, plots_dir / "predictions_vs_truth.png")
        plot_residuals_vs_predicted(eval_result.predictions, plots_dir / "residuals_vs_predicted.png")
        plot_residual_histogram(eval_result.predictions, plots_dir / "residual_histogram.png")
        plot_residuals_vs_time(eval_result.predictions, plots_dir / "residuals_vs_time.png")
        if not eval_result.feature_importance.empty:
            plot_feature_importance(eval_result.feature_importance, plots_dir / "feature_importance.png")
        plot_residuals_vs_features(
            run.split.X_test,
            eval_result.predictions,
            plots_dir / "residuals_vs_features.png",
        )

    return {
        "run": run,
        "model": model,
        "model_name": model_name,
        "evaluation": eval_result,
        "shap": shap_df,
        "output_dir": run_dir,
    }


def run_experiment(config: ExperimentConfig) -> dict[str, object]:
    out_dir = ensure_dir(config.output_dir)
    global_plots_dir = ensure_dir(out_dir / "plots_global")

    enriched = prepare_dataset(
        ring_temp_csv=str(config.ring_temp_csv),
        los_csv=str(config.los_csv),
        sensors=list(config.sensors),
        ring_obmt_col=config.ring_obmt_col,
        los_obmt_col=config.los_obmt_col,
        target_col=config.target_col,
        target_scale=config.target_scale,
        trend_col=config.trend_col,
        obmt_range=config.obmt_range,
        temperature_detrend=config.temperature_detrend,
    )

    if config.save_enriched_dataset:
        save_dataframe(enriched, out_dir / "enriched_dataset.csv")

    model_df = select_model_columns(enriched, sensors=config.sensors, target_col="y_mas")
    feature_cols = [f"{sensor}_avg" for sensor in config.sensors]

    runs = build_validation_runs(
        df=model_df,
        feature_cols=feature_cols,
        target_col="y_mas",
        obmt_col=config.los_obmt_col,
        random_test_size=config.rf.test_size,
        random_state=config.rf.split_random_state,
        time_blocked_test_size=config.time_blocked.test_size,
        time_blocked_gap_size=config.time_blocked.gap_size,
    )
    enabled_models = get_enabled_models(config)

    run_outputs: dict[str, dict[str, dict[str, object]]] = {}
    comparison_rows: list[dict[str, object]] = []

    for run in runs:
        run_outputs[run.name] = {}

        for model_name in enabled_models:
            model = train_model(run.split, model_name, config)
            eval_result = evaluate_model(
                model=model,
                X_test=run.split.X_test,
                y_test=run.split.y_test,
                obmt_test=run.split.obmt_test,
            )

            run_dir = out_dir / run.name / model_name
            run_outputs[run.name][model_name] = _save_validation_outputs(
                run=run,
                model_name=model_name,
                model=model,
                eval_result=eval_result,
                out_dir=run_dir,
                make_plots=config.make_plots,
                compute_shap=config.compute_shap,
                save_model_flag=config.save_model,
            )

            comparison_rows.append(
                {
                    "validation": run.name,
                    "model": model_name,
                    **run.summary.to_dict(),
                    **eval_result.metrics,
                }
            )

    comparison_df = pd.DataFrame(comparison_rows)
    save_dataframe(comparison_df, out_dir / "model_validation_comparison.csv")
    save_json(comparison_rows, out_dir / "model_validation_comparison.json")

    if config.make_plots:
        plot_feature_correlation_heatmap(
            model_df,
            feature_cols=feature_cols,
            target_col="y_mas",
            path=global_plots_dir / "feature_correlation.png",
        )

        first_feature = feature_cols[0]
        plot_layered_scatter(
            df=enriched,
            x_col=first_feature,
            y_col="y_mas",
            obmt_col=config.los_obmt_col,
            layers=DEFAULT_LAYERS,
            x_label=first_feature,
            y_label=r"$\delta\eta$ [mas]",
            title=f"{config.name}: {first_feature} vs LOS variation",
            path=global_plots_dir / f"{first_feature}_vs_los_layered.png",
        )

    return {
        "config": config,
        "enriched_df": enriched,
        "model_df": model_df,
        "runs": run_outputs,
        "comparison": comparison_df,
    }

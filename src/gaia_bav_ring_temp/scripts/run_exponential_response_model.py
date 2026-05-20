from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from gaia_bav_ring_temp.dataset import prepare_dataset, select_model_columns
from gaia_bav_ring_temp.io import ensure_dir, save_dataframe, save_json
from gaia_bav_ring_temp.settings import load_config
from gaia_bav_ring_temp.thermal_response import (
    ThermalResponseFit,
    fit_common_tau_grid,
    fit_instantaneous_linear,
    fit_separate_taus_coordinate_grid,
    fit_summary_row,
    make_tau_grid,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit physically motivated exponential thermal-response models."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment TOML config.",
    )
    parser.add_argument(
        "--out-subdir",
        type=str,
        default="thermal_response_model",
        help="Subdirectory below config.output_dir.",
    )
    parser.add_argument(
        "--tau-min",
        type=float,
        default=0.02,
        help="Minimum tau in OBMT revolutions.",
    )
    parser.add_argument(
        "--tau-max",
        type=float,
        default=20.0,
        help="Maximum tau in OBMT revolutions.",
    )
    parser.add_argument(
        "--n-tau-grid",
        type=int,
        default=100,
        help="Number of tau grid points.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="Ridge alpha for the linear amplitude fit. Use 0 for ordinary least squares.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=6,
        help="Maximum coordinate-search iterations for separate tau model.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["instantaneous", "common_tau", "separate_tau", "random_forest"],
        choices=["instantaneous", "common_tau", "separate_tau", "random_forest"],
        help="Which models to run.",
    )
    return parser.parse_args()


def _prepare_enriched_dataset(config):
    prepared = prepare_dataset(
        ring_temp_csv=config.ring_temp_csv,
        los_csv=config.los_csv,
        sensors=config.sensors,
        ring_obmt_col=config.ring_obmt_col,
        los_obmt_col=config.los_obmt_col,
        target_col=config.target_col,
        target_scale=config.target_scale,
        trend_col=config.trend_col,
        obmt_range=config.obmt_range,
        temperature_detrend=getattr(config, "temperature_detrend", None),
    )

    # Compatibility with both project versions:
    # old prepare_dataset -> DataFrame
    # new prepare_dataset -> (DataFrame, detrend_summary)
    if isinstance(prepared, tuple):
        enriched = prepared[0]
        detrend_summary = prepared[1]
    else:
        enriched = prepared
        detrend_summary = pd.DataFrame()

    return enriched, detrend_summary


def _get_random_split_indices(model_df: pd.DataFrame, config) -> tuple[np.ndarray, np.ndarray]:
    test_size = getattr(config.rf, "test_size", 0.3)

    split_seed = getattr(
        config.rf,
        "split_random_state",
        getattr(config.rf, "random_state", 42),
    )

    indices = np.arange(len(model_df))

    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=split_seed,
        shuffle=True,
    )

    return np.asarray(train_idx), np.asarray(test_idx)


def _save_fit_outputs(result: ThermalResponseFit, out_dir: Path) -> None:
    model_dir = ensure_dir(out_dir / result.model_name)

    save_dataframe(result.predictions, model_dir / "predictions.csv", index=False)
    save_dataframe(result.coefficients, model_dir / "coefficients.csv", index=False)
    save_json(result.metrics, model_dir / "metrics.json")
    save_json(result.taus, model_dir / "taus.json")

    if not result.diagnostics.empty:
        save_dataframe(result.diagnostics, model_dir / "diagnostics.csv", index=False)

def _plot_predictions(results: list[ThermalResponseFit], out_path: Path) -> None:
    if not results:
        return

    n = len(results)
    ncols = 2
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(10.5, 4.8 * nrows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    axes = np.asarray(axes).reshape(-1)

    all_true = np.concatenate([r.predictions["y_true"].to_numpy() for r in results])
    all_pred = np.concatenate([r.predictions["y_pred"].to_numpy() for r in results])

    lo = float(min(all_true.min(), all_pred.min()))
    hi = float(max(all_true.max(), all_pred.max()))
    pad = 0.05 * max(hi - lo, 1.0e-9)
    lims = (lo - pad, hi + pad)

    display_names = {
        "instantaneous_linear": "Instantaneous linear",
        "exponential_common_tau": "Exponential response\ncommon $\\tau$",
        "exponential_separate_tau": "Exponential response\nseparate $\\tau_i$",
        "random_forest": "Random forest",
    }

    for ax, result in zip(axes, results):
        pred = result.predictions

        ax.scatter(
            pred["y_true"],
            pred["y_pred"],
            s=12,
            alpha=0.6,
        )
        ax.plot(lims, lims, "r--", linewidth=1.2)

        title = display_names.get(result.model_name, result.model_name.replace("_", " "))
        ax.set_title(title)
        ax.set_xlabel("True LoS variation [mas]")
        ax.set_ylabel("Predicted LoS variation [mas]")
        ax.set_xlim(*lims)
        ax.set_ylim(*lims)
        ax.grid(True, alpha=0.3)

        ax.text(
            0.03,
            0.97,
            f"RMSE = {result.metrics['rmse']:.3f} mas\n"
            f"$R^2$ = {result.metrics['r2']:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )

    # Hide unused axes if fewer than four models are plotted.
    for ax in axes[len(results):]:
        ax.axis("off")

    fig.suptitle("Thermal-response models and random-forest comparison")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
def _plot_predictions_original(results: list[ThermalResponseFit], out_path: Path) -> None:
    if not results:
        return

    n = len(results)
    fig, axes = plt.subplots(
        1,
        n,
        figsize=(5.0 * n, 4.5),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    if n == 1:
        axes = [axes]

    all_true = np.concatenate([r.predictions["y_true"].to_numpy() for r in results])
    all_pred = np.concatenate([r.predictions["y_pred"].to_numpy() for r in results])

    lo = float(min(all_true.min(), all_pred.min()))
    hi = float(max(all_true.max(), all_pred.max()))
    pad = 0.05 * max(hi - lo, 1.0e-9)
    lims = (lo - pad, hi + pad)

    for ax, result in zip(axes, results):
        pred = result.predictions

        ax.scatter(
            pred["y_true"],
            pred["y_pred"],
            s=12,
            alpha=0.6,
        )
        ax.plot(lims, lims, "r--", linewidth=1.2)

        ax.set_title(result.model_name.replace("_", " "))
        ax.set_xlabel("True LoS variation [mas]")
        ax.set_ylabel("Predicted LoS variation [mas]")
        ax.set_xlim(*lims)
        ax.set_ylim(*lims)
        ax.grid(True, alpha=0.3)

        ax.text(
            0.03,
            0.97,
            f"RMSE = {result.metrics['rmse']:.3f} mas\n"
            f"$R^2$ = {result.metrics['r2']:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )

    fig.suptitle("Exponential thermal-response model comparison")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_common_tau_diagnostics(result: ThermalResponseFit, out_path: Path) -> None:
    if result.diagnostics.empty or "tau" not in result.diagnostics.columns:
        return

    diag = result.diagnostics.sort_values("tau")

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.plot(diag["tau"], diag["train_rmse"], marker="o", markersize=3)
    ax.set_xscale("log")
    ax.set_xlabel(r"Common thermal response time $\tau$ [OBMT rev]")
    ax.set_ylabel("Training RMSE [mas]")
    ax.set_title("Common-tau grid search")
    ax.grid(True, alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def _plot_separate_tau_diagnostics(result: ThermalResponseFit, out_path: Path) -> None:
    """Plot coordinate grid-search diagnostics for the separate-tau model.

    The separate-tau search scans the tau grid for one feature at a time while
    keeping the other tau values fixed. This plot shows the training RMSE as a
    function of the trial tau for each feature and each coordinate-search
    iteration.
    """
    if result.diagnostics.empty:
        return

    diag = result.diagnostics.copy()

    required = {"iteration", "feature", "trial_tau", "train_mse"}
    missing = required - set(diag.columns)
    if missing:
        print(
            f"Skipping separate-tau diagnostic plot because columns are missing: "
            f"{sorted(missing)}"
        )
        return

    diag["train_rmse"] = np.sqrt(diag["train_mse"])

    features = list(result.taus.keys())
    n_features = len(features)

    fig, axes = plt.subplots(
        1,
        n_features,
        figsize=(5.2 * n_features, 4.4),
        sharey=True,
        constrained_layout=True,
    )

    if n_features == 1:
        axes = [axes]

    for ax, feature in zip(axes, features):
        sub = diag[diag["feature"] == feature].copy()
        if sub.empty:
            ax.set_title(feature)
            ax.text(
                0.5,
                0.5,
                "No diagnostics",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            continue

        for iteration, part in sub.groupby("iteration"):
            part = part.sort_values("trial_tau")
            ax.plot(
                part["trial_tau"],
                part["train_rmse"],
                marker="o",
                markersize=3,
                linewidth=1.2,
                alpha=0.75,
                label=f"iteration {iteration}",
            )

        selected_tau = result.taus.get(feature)
        if selected_tau is not None:
            ax.axvline(
                selected_tau,
                linestyle="--",
                linewidth=1.5,
                color="black",
                label=fr"selected $\tau$ = {selected_tau:.3g}",
            )

        ax.set_xscale("log")
        ax.set_xlabel(r"Trial $\tau$ [OBMT rev]")
        ax.set_title(feature)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Training RMSE [mas]")

    fig.suptitle("Separate-tau coordinate grid search")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
def _fit_random_forest_baseline(
    model_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    obmt_col: str,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    config,
) -> ThermalResponseFit:
    """Fit a random-forest baseline on the same split as the thermal-response models."""
    X = model_df[feature_cols]
    y = model_df[target_col]

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    model_random_state = getattr(
        config.rf,
        "model_random_state",
        getattr(config.rf, "random_state", 42),
    )

    model = RandomForestRegressor(
        n_estimators=config.rf.n_estimators,
        random_state=model_random_state,
        n_jobs=config.rf.n_jobs,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    from gaia_bav_ring_temp.thermal_response import evaluate_prediction

    metrics = evaluate_prediction(y_test, y_pred)

    pred_df = pd.DataFrame(
        {
            "obmt": model_df.iloc[test_idx][obmt_col].to_numpy(dtype=float),
            "y_true": y_test.to_numpy(dtype=float),
            "y_pred": y_pred,
            "residual": y_test.to_numpy(dtype=float) - y_pred,
        }
    ).sort_values("obmt").reset_index(drop=True)

    importance_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "coefficient": model.feature_importances_,
            "quantity": "random_forest_feature_importance",
        }
    )

    return ThermalResponseFit(
        model_name="random_forest",
        taus={col: 0.0 for col in feature_cols},
        intercept=0.0,
        coefficients=importance_df,
        metrics=metrics,
        predictions=pred_df,
        diagnostics=pd.DataFrame(),
    )
def main() -> None:
    args = parse_args()

    config = load_config(args.config)

    enriched, detrend_summary = _prepare_enriched_dataset(config)

    feature_cols = [f"{sensor}_avg" for sensor in config.sensors]

    model_df = select_model_columns(
        enriched,
        sensors=config.sensors,
        target_col="y_mas",
    )

    # The exponential response is a time-domain operation, so we sort by OBMT.
    model_df = model_df.sort_values(config.los_obmt_col).reset_index(drop=True)

    train_idx, test_idx = _get_random_split_indices(model_df, config)

    tau_grid = make_tau_grid(
        tau_min=args.tau_min,
        tau_max=args.tau_max,
        n_grid=args.n_tau_grid,
    )

    out_dir = ensure_dir(Path(config.output_dir) / args.out_subdir)

    save_dataframe(model_df, out_dir / "model_dataset_used.csv", index=False)

    if detrend_summary is not None and not detrend_summary.empty:
        save_dataframe(detrend_summary, out_dir / "temperature_detrend_summary.csv", index=False)

    split_summary = {
        "config": args.config,
        "n_rows": int(len(model_df)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "feature_columns": feature_cols,
        "target_col": "y_mas",
        "obmt_col": config.los_obmt_col,
        "tau_min": args.tau_min,
        "tau_max": args.tau_max,
        "n_tau_grid": args.n_tau_grid,
        "alpha": args.alpha,
    }
    save_json(split_summary, out_dir / "run_summary.json")

    results: list[ThermalResponseFit] = []

    if "instantaneous" in args.models:
        instantaneous = fit_instantaneous_linear(
            df=model_df,
            feature_cols=feature_cols,
            target_col="y_mas",
            obmt_col=config.los_obmt_col,
            train_idx=train_idx,
            test_idx=test_idx,
            alpha=args.alpha,
        )
        results.append(instantaneous)
        _save_fit_outputs(instantaneous, out_dir)

    common_tau_result = None
    if "common_tau" in args.models:
        common_tau_result = fit_common_tau_grid(
            df=model_df,
            feature_cols=feature_cols,
            target_col="y_mas",
            obmt_col=config.los_obmt_col,
            train_idx=train_idx,
            test_idx=test_idx,
            tau_grid=tau_grid,
            alpha=args.alpha,
        )
        results.append(common_tau_result)
        _save_fit_outputs(common_tau_result, out_dir)
        _plot_common_tau_diagnostics(
            common_tau_result,
            out_dir / "common_tau_grid_search.png",
        )

    if "separate_tau" in args.models:
        if common_tau_result is not None:
            initial_tau = list(common_tau_result.taus.values())[0]
        else:
            initial_tau = float(np.median(tau_grid))

        separate_tau_result = fit_separate_taus_coordinate_grid(
            df=model_df,
            feature_cols=feature_cols,
            target_col="y_mas",
            obmt_col=config.los_obmt_col,
            train_idx=train_idx,
            test_idx=test_idx,
            tau_grid=tau_grid,
            initial_tau=initial_tau,
            max_iter=args.max_iter,
            alpha=args.alpha,
        )
        results.append(separate_tau_result)
        _save_fit_outputs(separate_tau_result, out_dir)

        _plot_separate_tau_diagnostics(
            separate_tau_result,
            out_dir / "separate_tau_grid_search.png",
        )

    if "random_forest" in args.models:
        random_forest_result = _fit_random_forest_baseline(
            model_df=model_df,
            feature_cols=feature_cols,
            target_col="y_mas",
            obmt_col=config.los_obmt_col,
            train_idx=train_idx,
            test_idx=test_idx,
            config=config,
        )
        results.append(random_forest_result)
        _save_fit_outputs(random_forest_result, out_dir)

    comparison = pd.DataFrame([fit_summary_row(r) for r in results])
    save_dataframe(comparison, out_dir / "thermal_response_comparison.csv", index=False)

    _plot_predictions(
        results,
        out_dir / "thermal_response_predictions_vs_truth.png",
    )

    print("Finished exponential thermal-response model experiment.")
    print(f"Output directory: {out_dir}")
    print()
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
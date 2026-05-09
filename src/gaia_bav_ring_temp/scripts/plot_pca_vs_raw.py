from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METRICS = ("rmse", "r2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot PCA-vs-raw model comparison results."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Path to pca_vs_raw_comparison.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to <csv parent>/plots",
    )
    return parser.parse_args()


def load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {"feature_space", "model", "n_components", "rmse", "r2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in {path}: {sorted(missing)}"
        )

    df = df.copy()
    df["feature_space"] = df["feature_space"].astype(str)
    df["model"] = df["model"].astype(str)
    df["n_components"] = pd.to_numeric(df["n_components"], errors="coerce")

    return df


def save_best_pca_table(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    pca_df = df[df["feature_space"] == "pca"].copy()
    if pca_df.empty:
        best = pd.DataFrame(
            columns=["model", "n_components", "rmse", "r2", "mae", "mse", "n_test"]
        )
    else:
        best = (
            pca_df.sort_values(["model", "rmse"])
            .groupby("model", as_index=False)
            .first()
        )

    best.to_csv(out_dir / "best_pca_by_model.csv", index=False)
    return best


def save_raw_vs_best_pca_table(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    raw_df = df[df["feature_space"] == "raw"].copy()
    pca_df = df[df["feature_space"] == "pca"].copy()

    if raw_df.empty or pca_df.empty:
        merged = pd.DataFrame()
        merged.to_csv(out_dir / "raw_vs_best_pca_summary.csv", index=False)
        return merged

    best_pca = (
        pca_df.sort_values(["model", "rmse"])
        .groupby("model", as_index=False)
        .first()
        .rename(
            columns={
                "n_components": "best_pca_components",
                "rmse": "best_pca_rmse",
                "r2": "best_pca_r2",
                "mae": "best_pca_mae",
                "mse": "best_pca_mse",
            }
        )
    )

    raw_df = raw_df.rename(
        columns={
            "n_components": "raw_components",
            "rmse": "raw_rmse",
            "r2": "raw_r2",
            "mae": "raw_mae",
            "mse": "raw_mse",
        }
    )

    keep_cols = [
        "model",
        "raw_components",
        "raw_rmse",
        "raw_r2",
        "raw_mae",
        "raw_mse",
        "best_pca_components",
        "best_pca_rmse",
        "best_pca_r2",
        "best_pca_mae",
        "best_pca_mse",
    ]
    merged = pd.merge(raw_df, best_pca, on="model", how="inner")[keep_cols]
    merged["delta_rmse_best_pca_minus_raw"] = (
        merged["best_pca_rmse"] - merged["raw_rmse"]
    )
    merged["delta_r2_best_pca_minus_raw"] = (
        merged["best_pca_r2"] - merged["raw_r2"]
    )

    merged.to_csv(out_dir / "raw_vs_best_pca_summary.csv", index=False)
    return merged


def plot_metric_vs_components(
    df: pd.DataFrame,
    metric: str,
    out_path: Path,
) -> None:
    pca_df = df[df["feature_space"] == "pca"].copy()
    raw_df = df[df["feature_space"] == "raw"].copy()

    if pca_df.empty:
        print(f"Skipping {metric}: no PCA rows found.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    models = sorted(pca_df["model"].unique())
    for model_name in models:
        sub = (
            pca_df[pca_df["model"] == model_name]
            .sort_values("n_components")
            .copy()
        )
        if sub.empty:
            continue

        line, = ax.plot(
            sub["n_components"],
            sub[metric],
            marker="o",
            label=f"{model_name} (PCA)",
        )

        raw_sub = raw_df[raw_df["model"] == model_name]
        if not raw_sub.empty:
            raw_value = float(raw_sub.iloc[0][metric])
            ax.axhline(
                raw_value,
                linestyle="--",
                linewidth=1.5,
                color=line.get_color(),
                alpha=0.8,
                label=f"{model_name} raw",
            )

    ax.set_xlabel("Number of principal components")
    ax.set_ylabel(metric.upper() if metric != "r2" else r"$R^2$")
    ax.set_title(f"{metric.upper() if metric != 'r2' else 'R^2'} vs number of PCA components")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_best_pca_vs_raw_bars(df: pd.DataFrame, metric: str, out_path: Path) -> None:
    summary = save_raw_vs_best_pca_table(df, out_path.parent)
    if summary.empty:
        print(f"Skipping {metric} bar plot: no raw/PCA comparison available.")
        return

    summary = summary.sort_values("model").reset_index(drop=True)

    x = range(len(summary))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        [i - width / 2 for i in x],
        summary[f"raw_{metric}"],
        width=width,
        label="raw",
    )
    ax.bar(
        [i + width / 2 for i in x],
        summary[f"best_pca_{metric}"],
        width=width,
        label="best PCA",
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(summary["model"])
    ax.set_ylabel(metric.upper() if metric != "r2" else r"$R^2$")
    ax.set_title(
        f"Raw features vs best PCA setup by model ({metric.upper() if metric != 'r2' else 'R^2'})"
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_text_summary(df: pd.DataFrame, out_path: Path) -> None:
    raw_df = df[df["feature_space"] == "raw"].copy()
    pca_df = df[df["feature_space"] == "pca"].copy()

    lines: list[str] = []
    lines.append("# PCA vs raw summary")
    lines.append("")

    if raw_df.empty:
        lines.append("No raw-feature rows were found.")
        lines.append("")
    else:
        lines.append("## Raw feature baselines")
        lines.append("")
        for _, row in raw_df.sort_values("model").iterrows():
            lines.append(
                f"- {row['model']}: RMSE={row['rmse']:.6f}, "
                f"MAE={row.get('mae', float('nan')):.6f}, "
                f"R2={row['r2']:.6f}"
            )
        lines.append("")

    if pca_df.empty:
        lines.append("No PCA rows were found.")
        lines.append("")
    else:
        lines.append("## Best PCA setup by model")
        lines.append("")
        best = (
            pca_df.sort_values(["model", "rmse"])
            .groupby("model", as_index=False)
            .first()
        )
        for _, row in best.sort_values("model").iterrows():
            lines.append(
                f"- {row['model']}: best PCA uses {int(row['n_components'])} component(s), "
                f"RMSE={row['rmse']:.6f}, "
                f"MAE={row.get('mae', float('nan')):.6f}, "
                f"R2={row['r2']:.6f}"
            )
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    df = load_results(args.csv)

    out_dir = args.out_dir or (args.csv.parent / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    save_best_pca_table(df, out_dir)
    save_raw_vs_best_pca_table(df, out_dir)
    write_text_summary(df, out_dir / "pca_vs_raw_summary.md")

    plot_metric_vs_components(df, "rmse", out_dir / "rmse_vs_pca_components.png")
    plot_metric_vs_components(df, "r2", out_dir / "r2_vs_pca_components.png")

    plot_best_pca_vs_raw_bars(df, "rmse", out_dir / "raw_vs_best_pca_rmse.png")
    plot_best_pca_vs_raw_bars(df, "r2", out_dir / "raw_vs_best_pca_r2.png")

    print(f"Saved plots and summary tables to: {out_dir}")


if __name__ == "__main__":
    main()
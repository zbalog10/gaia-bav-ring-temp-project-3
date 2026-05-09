"""Command-line entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_experiment
from .settings import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Gaia BAV ring temperature experiment.")
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to a TOML experiment config.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = load_config(args.config)
    result = run_experiment(config)
    comparison = result["comparison"]

    print(f"Experiment: {config.name}")
    comparison = comparison.sort_values(["validation", "model"]).reset_index(drop=True)

    for validation_name, group in comparison.groupby("validation", sort=False):
        print()
        print(f"Validation: {validation_name}")
        for row in group.itertuples(index=False):
            print(
                f"  {row.model:<14}"
                f"RMSE={row.rmse:.6f}  "
                f"MAE={row.mae:.6f}  "
                f"R2={row.r2:.6f}  "
                f"n_test={row.n_test}"
            )

    print()
    print(f"Saved outputs to: {config.output_dir}")


if __name__ == "__main__":
    main()

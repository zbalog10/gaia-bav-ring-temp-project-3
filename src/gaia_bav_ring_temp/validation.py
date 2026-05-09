"""Validation split helpers for random and time-blocked experiments."""

from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(slots=True)
class SplitData:
    """Container for train/test partitions plus their OBMT coordinates."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    obmt_train: pd.Series
    obmt_test: pd.Series

    @property
    def n_train(self) -> int:
        return int(len(self.y_train))

    @property
    def n_test(self) -> int:
        return int(len(self.y_test))


@dataclass(slots=True)
class ValidationSummary:
    """Metadata describing how a given validation split was constructed."""

    mode: str
    n_total: int
    n_train: int
    n_test: int
    test_fraction: float
    gap_size: int = 0
    train_obmt_min: float | None = None
    train_obmt_max: float | None = None
    test_obmt_min: float | None = None
    test_obmt_max: float | None = None

    def to_dict(self) -> dict[str, int | float | str | None]:
        return {
            "mode": self.mode,
            "n_total": self.n_total,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "test_fraction": self.test_fraction,
            "gap_size": self.gap_size,
            "train_obmt_min": self.train_obmt_min,
            "train_obmt_max": self.train_obmt_max,
            "test_obmt_min": self.test_obmt_min,
            "test_obmt_max": self.test_obmt_max,
        }


@dataclass(slots=True)
class ValidationRun:
    """One named validation run: split data plus metadata."""

    name: str
    split: SplitData
    summary: ValidationSummary


def _build_split_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    obmt_train: pd.Series,
    obmt_test: pd.Series,
) -> SplitData:
    return SplitData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        obmt_train=obmt_train,
        obmt_test=obmt_test,
    )


def make_random_split(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    obmt_col: str,
    test_size: float,
    random_state: int,
) -> ValidationRun:
    """Random train/test split identical in spirit to the original notebooks."""

    X = df[feature_cols]
    y = df[target_col]
    obmt = df[obmt_col]

    X_train, X_test, y_train, y_test, obmt_train, obmt_test = train_test_split(
        X, y, obmt, test_size=test_size, random_state=random_state
    )
    split = _build_split_data(X_train, X_test, y_train, y_test, obmt_train, obmt_test)
    summary = ValidationSummary(
        mode="random_split",
        n_total=len(df),
        n_train=split.n_train,
        n_test=split.n_test,
        test_fraction=split.n_test / len(df),
        gap_size=0,
        train_obmt_min=float(obmt_train.min()),
        train_obmt_max=float(obmt_train.max()),
        test_obmt_min=float(obmt_test.min()),
        test_obmt_max=float(obmt_test.max()),
    )
    return ValidationRun(name="random_split", split=split, summary=summary)


def make_time_blocked_split(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    obmt_col: str,
    test_size: float,
    gap_size: int = 0,
) -> ValidationRun:
    """Chronological holdout split using the final contiguous OBMT block as test set.

    The dataframe is sorted by ``obmt_col``. The last ``test_size`` fraction becomes the
    test block. ``gap_size`` rows immediately before the test block are dropped from both
    training and test to reduce leakage from rolling-window features.
    """

    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size!r}")
    if gap_size < 0:
        raise ValueError(f"gap_size must be >= 0, got {gap_size!r}")

    sorted_df = df.sort_values(obmt_col).reset_index(drop=True)
    n_total = len(sorted_df)
    n_test = max(1, int(math.ceil(n_total * test_size)))
    test_start = n_total - n_test
    train_end = test_start - gap_size

    if train_end <= 0:
        raise ValueError(
            "Time-blocked split leaves no training data. "
            f"n_total={n_total}, test_size={test_size}, gap_size={gap_size}."
        )

    train_df = sorted_df.iloc[:train_end].copy()
    test_df = sorted_df.iloc[test_start:].copy()

    split = _build_split_data(
        X_train=train_df[feature_cols],
        X_test=test_df[feature_cols],
        y_train=train_df[target_col],
        y_test=test_df[target_col],
        obmt_train=train_df[obmt_col],
        obmt_test=test_df[obmt_col],
    )
    summary = ValidationSummary(
        mode="time_blocked",
        n_total=n_total,
        n_train=split.n_train,
        n_test=split.n_test,
        test_fraction=split.n_test / n_total,
        gap_size=gap_size,
        train_obmt_min=float(train_df[obmt_col].min()),
        train_obmt_max=float(train_df[obmt_col].max()),
        test_obmt_min=float(test_df[obmt_col].min()),
        test_obmt_max=float(test_df[obmt_col].max()),
    )
    return ValidationRun(name="time_blocked", split=split, summary=summary)


def build_validation_runs(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    obmt_col: str,
    random_test_size: float,
    random_state: int,
    time_blocked_test_size: float,
    time_blocked_gap_size: int,
) -> list[ValidationRun]:
    """Construct the standard pair of validation runs for comparison."""

    return [
        make_random_split(
            df=df,
            feature_cols=feature_cols,
            target_col=target_col,
            obmt_col=obmt_col,
            test_size=random_test_size,
            random_state=random_state,
        ),
        make_time_blocked_split(
            df=df,
            feature_cols=feature_cols,
            target_col=target_col,
            obmt_col=obmt_col,
            test_size=time_blocked_test_size,
            gap_size=time_blocked_gap_size,
        ),
    ]

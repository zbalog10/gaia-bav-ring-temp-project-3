"""File I/O helpers."""

from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
from joblib import dump


def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_dataframe(df: pd.DataFrame, path: str | Path, index: bool = False) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)


def save_json(data: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def save_model(model: object, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    dump(model, path)

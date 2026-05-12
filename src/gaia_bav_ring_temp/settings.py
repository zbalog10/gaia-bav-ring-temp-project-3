"""Configuration models and config loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import tomllib

from .constants import DEFAULT_SCALE_FACTOR_MAS, DEFAULT_SENSOR_COLUMNS


@dataclass(slots=True)
class RandomForestConfig:
    n_estimators: int = 100
    model_random_state: int = 42
    split_random_state: int = 42
    test_size: float = 0.3
    n_jobs: int = -1


@dataclass(slots=True)
class TimeBlockedConfig:
    test_size: float = 0.3
    gap_size: int = 0

@dataclass(slots=True)
class TemperatureDetrendConfig:
    enabled: bool = False
    method: str = "linear"
    keep_raw_columns: bool = True
    preserve_mean: bool = True

    # Parameters for notebook-style hybrid rolling detrending
    switch_obmt: float | None = None
    fft_initial_window: int = 50
    anchor_obmt: float | None = None
    forced_window_size: int | None = None

@dataclass(slots=True)
class ModelsConfig:
    enabled: list[str] = field(
        default_factory=lambda: ["random_forest", "linear", "ridge", "mlp"]
    )


@dataclass(slots=True)
class RidgeConfig:
    alpha: float = 1.0


@dataclass(slots=True)
class MlpConfig:
    hidden_layer_sizes: tuple[int, ...] = (32, 16, 8)
    activation: str = "relu"
    alpha: float = 1.0e-4
    learning_rate_init: float = 1.0e-3
    max_iter: int = 2000
    early_stopping: bool = True
    validation_fraction: float = 0.1
    n_iter_no_change: int = 20
    random_state: int = 42


@dataclass(slots=True)
class ExperimentConfig:
    name: str
    ring_temp_csv: Path
    los_csv: Path
    output_dir: Path
    sensors: tuple[str, ...] = field(default_factory=lambda: DEFAULT_SENSOR_COLUMNS)
    ring_obmt_col: str = "OBMT_rev"
    los_obmt_col: str = "obmtRev"
    target_col: str = "DN_LINE_OF_SIGHT_VARIATIONS"
    trend_col: str | None = "trend"
    target_scale: float = DEFAULT_SCALE_FACTOR_MAS
    obmt_min: float | None = None
    obmt_max: float | None = None
    save_enriched_dataset: bool = True
    save_model: bool = True
    make_plots: bool = True
    compute_shap: bool = False
    rf: RandomForestConfig = field(default_factory=RandomForestConfig)
    time_blocked: TimeBlockedConfig = field(default_factory=TimeBlockedConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    ridge: RidgeConfig = field(default_factory=RidgeConfig)
    mlp: MlpConfig = field(default_factory=MlpConfig)
    temperature_detrend: TemperatureDetrendConfig = field(
        default_factory=TemperatureDetrendConfig
    )


    @property
    def obmt_range(self) -> tuple[float, float] | None:
        if self.obmt_min is None or self.obmt_max is None:
            return None
        return (self.obmt_min, self.obmt_max)


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    raw = tomllib.loads(path.read_text(encoding="utf-8"))

    rf_raw = raw.get("random_forest", {})
    rf_cfg = RandomForestConfig(
        n_estimators=rf_raw.get("n_estimators", 100),
        model_random_state=rf_raw.get("model_random_state", rf_raw.get("random_state", 42)),
        split_random_state=rf_raw.get("split_random_state", rf_raw.get("random_state", 42)),
        test_size=rf_raw.get("test_size", 0.3),
        n_jobs=rf_raw.get("n_jobs", -1),
    )

    tb_raw = raw.get("time_blocked", {})
    tb_cfg = TimeBlockedConfig(
        test_size=tb_raw.get("test_size", rf_cfg.test_size),
        gap_size=tb_raw.get("gap_size", 0),
    )

    models_raw = raw.get("models", {})
    models_cfg = ModelsConfig(
        enabled=models_raw.get(
            "enabled",
            ["random_forest", "linear", "ridge", "mlp"],
        )
    )

    ridge_raw = raw.get("ridge", {})
    ridge_cfg = RidgeConfig(
        alpha=ridge_raw.get("alpha", 1.0),
    )

    mlp_raw = raw.get("mlp", {})
    mlp_cfg = MlpConfig(
        hidden_layer_sizes=tuple(mlp_raw.get("hidden_layer_sizes", [32, 16, 8])),
        activation=mlp_raw.get("activation", "relu"),
        alpha=mlp_raw.get("alpha", 1.0e-4),
        learning_rate_init=mlp_raw.get("learning_rate_init", 1.0e-3),
        max_iter=mlp_raw.get("max_iter", 2000),
        early_stopping=mlp_raw.get("early_stopping", True),
        validation_fraction=mlp_raw.get("validation_fraction", 0.1),
        n_iter_no_change=mlp_raw.get("n_iter_no_change", 20),
        random_state=mlp_raw.get("random_state", 42),
    )

    detrend_raw = raw.get("temperature_detrend", {})
    detrend_cfg = TemperatureDetrendConfig(
        enabled=detrend_raw.get("enabled", False),
        method=detrend_raw.get("method", "linear"),
        keep_raw_columns=detrend_raw.get("keep_raw_columns", True),
        preserve_mean=detrend_raw.get("preserve_mean", True),
        switch_obmt=detrend_raw.get("switch_obmt"),
        fft_initial_window=detrend_raw.get("fft_initial_window", 50),
        anchor_obmt=detrend_raw.get("anchor_obmt"),
        forced_window_size=detrend_raw.get("forced_window_size"),
    )

    exp_raw = raw["experiment"]
    sensors_raw = raw.get("sensors", {}).get("columns", list(DEFAULT_SENSOR_COLUMNS))

    return ExperimentConfig(
        name=exp_raw["name"],
        ring_temp_csv=Path(exp_raw["ring_temp_csv"]),
        los_csv=Path(exp_raw["los_csv"]),
        output_dir=Path(exp_raw["output_dir"]),
        sensors=tuple(sensors_raw),
        ring_obmt_col=exp_raw.get("ring_obmt_col", "OBMT_rev"),
        los_obmt_col=exp_raw.get("los_obmt_col", "obmtRev"),
        target_col=exp_raw.get("target_col", "DN_LINE_OF_SIGHT_VARIATIONS"),
        trend_col=exp_raw.get("trend_col", "trend"),
        target_scale=exp_raw.get("target_scale", DEFAULT_SCALE_FACTOR_MAS),
        obmt_min=exp_raw.get("obmt_min"),
        obmt_max=exp_raw.get("obmt_max"),
        save_enriched_dataset=exp_raw.get("save_enriched_dataset", True),
        save_model=exp_raw.get("save_model", True),
        make_plots=exp_raw.get("make_plots", True),
        compute_shap=exp_raw.get("compute_shap", False),
        rf=rf_cfg,
        time_blocked=tb_cfg,
        models=models_cfg,
        ridge=ridge_cfg,
        mlp=mlp_cfg,
        temperature_detrend=detrend_cfg,
    )

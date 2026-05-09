"""Project-wide constants."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_SCALE_FACTOR_MAS = 2.0626480624709636e8
DEFAULT_SENSOR_COLUMNS = ("NEI00054", "NEI00230", "NEI00321")


@dataclass(frozen=True)
class Layer:
    start: float
    end: float
    color: str
    label: str | None = None


DEFAULT_LAYERS: tuple[Layer, ...] = (
    Layer(16406.73, 16414.53, "yellow", "slow slew"),
    Layer(16414.53, 16490.09, "orange", r"SAA$_{0\_SM}$"),
    Layer(16490.09, 16514.16, "red", r"SAA$_{0}$"),
    Layer(16515.15, 16527.14, "blue", r"SAA$_{5}$"),
    Layer(16528.19, 16540.15, "green", r"SAA$_{15}$"),
    Layer(16541.17, 16553.16, "grey", r"SAA$_{28}$"),
    Layer(16554.17, 16566.17, "magenta", r"SAA$_{43}$"),
    Layer(16567.18, 16571.17, "cyan", r"SAA$_{45}$"),
    Layer(16571.38, 16579.20, "yellow", None),
)

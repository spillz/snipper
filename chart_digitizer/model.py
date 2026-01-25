from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Literal


ChartKind = Literal["line", "scatter", "column", "bar", "area"]
StrideMode = Literal["continuous", "categorical"]


@dataclass(frozen=True)
class SeriesCalibration:
    name: str
    roi_px: Tuple[int, int, int, int]
    x_axis_px: Tuple[int, int]
    y_axis_px: Tuple[int, int]
    x0_val: str
    x1_val: str
    y0_val: str
    y1_val: str
    categories: str
    x_scale: str
    y_scale: str
    date_fmt: str
    x_step: float
    x_step_unit: str
    y_step: float
    y_step_unit: str


@dataclass
class PointFlags:
    enabled: bool = True


@dataclass
class Series:
    id: int
    name: str
    color_bgr: Tuple[int, int, int]
    calibration: SeriesCalibration

    # Primary chart kind for extraction/edit/export.
    # Backwards-compat: old code used mode in {"line","scatter"}.
    chart_kind: ChartKind = "line"

    # For stacked charts, user adds boundary series; export can emit deltas.
    stacked: bool = False

    # Sampling strategy for categorical axes.
    stride_mode: StrideMode = "continuous"

    # Prefer outline/edge pixels (often optimal for bar/area/column charts).
    prefer_outline: bool = True

    enabled: bool = True


    # Optional seed pixel (set when user clicks the chart)
    seed_px: Optional[Tuple[int, int]] = None
    # Optional extra seed pixels for improved tracking (Ctrl+click)
    extra_seeds_px: List[Tuple[int, int]] = field(default_factory=list)

    # data points (x,y) in chart units; for grid-based kinds these align to x_grid/categories
    points: List[Tuple[float, float]] = field(default_factory=list)

    # pixel points for overlay
    px_points: List[Tuple[int, int]] = field(default_factory=list)

    # per-point enabled flags (same length as points); used for NA
    point_enabled: List[bool] = field(default_factory=list)

    @property
    def mode(self) -> str:
        """Backwards-compat alias."""
        return self.chart_kind

    @mode.setter
    def mode(self, v: str) -> None:
        self.chart_kind = v  # type: ignore[assignment]


@dataclass
class ChartState:
    # ROI bounds in pixels inclusive
    xmin_px: int = 0
    ymin_px: int = 0
    xmax_px: int = 0
    ymax_px: int = 0

    # axis anchor pixels (None means use ROI bounds)
    x0_px: Optional[int] = None
    x1_px: Optional[int] = None
    y0_px: Optional[int] = None
    y1_px: Optional[int] = None

    # axis values as strings (parsed depending on scale)
    x0_val: str = ""
    x1_val: str = ""
    y0_val: str = ""
    y1_val: str = ""

    # output grid step (used for continuous stride in line/column/area)
    x_step: float = 1.0

    # color tolerance
    color_tol: int = 30

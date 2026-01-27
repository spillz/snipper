from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


ChartKind = Literal["line", "scatter", "column", "bar", "area"]
StrideMode = Literal["continuous", "categorical"]


@dataclass(frozen=True)
class CalibrationConfig:
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
    sample_mode: str


@dataclass
class Series:
    id: int
    name: str
    color_bgr: Tuple[int, int, int]
    calibration: CalibrationConfig

    # Primary chart kind for extraction/edit/export.
    # Backwards-compat: old code used mode in {"line","scatter"}.
    chart_kind: ChartKind = "line"

    # For stacked charts, user adds boundary series; export can emit deltas.
    stacked: bool = False

    # Sampling strategy for categorical axes.
    stride_mode: StrideMode = "continuous"

    # Prefer outline/edge pixels (often optimal for bar/area/column charts).
    prefer_outline: bool = True
    # Per-series color tolerance.
    color_tol: int = 20
    # Per-series scatter template match threshold.
    scatter_match_thresh: float = 0.6

    enabled: bool = True

    # Optional per-series mask bitmap (uint8, 0/255) in full image coords.
    mask_bitmap: Optional["np.ndarray"] = None
    # If True, treat the mask as an exclusion (negative) mask.
    mask_invert: bool = False

    # Optional seed pixel (set when user clicks the chart)
    seed_px: Optional[Tuple[int, int]] = None
    # Optional bbox (x0,y0,x1,y1) in image px for scatter template
    seed_bbox_px: Optional[Tuple[int, int, int, int]] = None
    # Optional extra template bboxes in image px for scatter
    scatter_seed_bboxes_px: List[Tuple[int, int, int, int]] = field(default_factory=list)
    # Optional marker bounds for scatter overlay (x0,y0,x1,y1) in image px
    scatter_marker_bboxes_px: List[Tuple[int, int, int, int]] = field(default_factory=list)
    # Optional marker bounds (x0,y0,x1,y1) in image px for scatter overlay
    seed_marker_bbox_px: Optional[Tuple[int, int, int, int]] = None
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
class ChartDocument:
    # Reserved for future persistence/annotation workflows; keep even if unused today.
    chart_uri: Optional[str] = None
    chart_blob: Optional[bytes] = None
    image_sha256: Optional[str] = None
    series: List[Series] = field(default_factory=list)
    calibration_configs: List[CalibrationConfig] = field(default_factory=list)


from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

@dataclass
class PointFlags:
    enabled: bool = True

@dataclass
class Series:
    id: int
    name: str
    color_bgr: Tuple[int,int,int]
    mode: str  # 'line' or 'scatter'
    enabled: bool = True

    # data points (x,y) in chart units; for line these align to x_grid
    points: List[Tuple[float,float]] = field(default_factory=list)
    # pixel points for overlay
    px_points: List[Tuple[int,int]] = field(default_factory=list)
    # per-point enabled flags (same length as points); used for NA
    point_enabled: List[bool] = field(default_factory=list)

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

    # output grid for line mode
    x_step: float = 1.0

    # color tolerance
    color_tol: int = 30

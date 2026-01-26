from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class CalibrationOverlayState:
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

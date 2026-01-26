
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional
from datetime import datetime, timezone

class AxisScale(str, Enum):
    LINEAR = "linear"
    LOG10 = "log10"
    DATE = "date"
    CATEGORICAL = "categorical"

def _parse_date(s: str, fmt: str) -> float:
    # returns unix seconds (UTC)
    dt = datetime.strptime(s.strip(), fmt)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()

def _format_date(ts: float, fmt: str) -> str:
    dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
    return dt.strftime(fmt)

@dataclass
class AxisCalibration:
    # pixel anchors
    p0: float
    p1: float
    # value anchors (stored numeric; for DATE this is unix seconds)
    v0: float
    v1: float
    scale: AxisScale = AxisScale.LINEAR

    def is_valid(self) -> bool:
        return self.p0 != self.p1 and self.v0 != self.v1

    def px_to_value(self, p: float) -> float:
        if self.scale in (AxisScale.LINEAR, AxisScale.CATEGORICAL):
            t = (p - self.p0) / (self.p1 - self.p0)
            return self.v0 + t * (self.v1 - self.v0)
        if self.scale == AxisScale.LOG10:
            import math
            if self.v0 <= 0 or self.v1 <= 0:
                raise ValueError("Log scale requires positive v0 and v1.")
            t = (p - self.p0) / (self.p1 - self.p0)
            lv0 = math.log10(self.v0)
            lv1 = math.log10(self.v1)
            return 10 ** (lv0 + t * (lv1 - lv0))
        if self.scale == AxisScale.DATE:
            # stored as unix seconds; linear in time
            t = (p - self.p0) / (self.p1 - self.p0)
            return self.v0 + t * (self.v1 - self.v0)
        raise ValueError(f"Unsupported scale: {self.scale}")

    def value_to_px(self, v: float) -> float:
        if self.scale in (AxisScale.LINEAR, AxisScale.DATE, AxisScale.CATEGORICAL):
            t = (v - self.v0) / (self.v1 - self.v0)
            return self.p0 + t * (self.p1 - self.p0)
        if self.scale == AxisScale.LOG10:
            import math
            if v <= 0 or self.v0 <= 0 or self.v1 <= 0:
                raise ValueError("Log scale requires positive values.")
            lv = math.log10(v)
            lv0 = math.log10(self.v0)
            lv1 = math.log10(self.v1)
            t = (lv - lv0) / (lv1 - lv0)
            return self.p0 + t * (self.p1 - self.p0)
        raise ValueError(f"Unsupported scale: {self.scale}")

@dataclass
class Calibration:
    x: AxisCalibration
    y: AxisCalibration
    x_date_format: str = "%Y"
    y_date_format: str = "%Y"

    def x_px_to_data(self, xpx: float) -> float:
        return self.x.px_to_value(xpx)

    def y_px_to_data(self, ypx: float) -> float:
        return self.y.px_to_value(ypx)

    def x_data_to_px(self, x: float) -> float:
        return self.x.value_to_px(x)

    def y_data_to_px(self, y: float) -> float:
        return self.y.value_to_px(y)

    def parse_x_value(self, s: str) -> float:
        if self.x.scale == AxisScale.DATE:
            return _parse_date(s, self.x_date_format)
        return float(s)

    def format_x_value(self, v: float) -> str:
        if self.x.scale == AxisScale.DATE:
            return _format_date(v, self.x_date_format)
        # Avoid scientific unless needed
        return str(v)

    def parse_y_value(self, s: str) -> float:
        if self.y.scale == AxisScale.DATE:
            return _parse_date(s, self.y_date_format)
        return float(s)

    def format_y_value(self, v: float) -> str:
        if self.y.scale == AxisScale.DATE:
            return _format_date(v, self.y_date_format)
        return str(v)

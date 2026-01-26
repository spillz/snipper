
from __future__ import annotations

import csv
from typing import Callable, List, Optional, Tuple
from .data_model import Series

def series_to_long_rows(
    series: List[Series],
    *,
    x_formatter: Optional[Callable[[float], str]] = None,
    x_formatter_by_series: Optional[Callable[[Series, float], str]] = None,
) -> List[Tuple[str, float, float | str]]:
    rows: List[Tuple[str, float, float | str]] = []
    for s in series:
        if not s.enabled:
            continue
        for (x,y), ok in zip(s.points, s.point_enabled or [True]*len(s.points)):
            if not ok:
                continue
            if x_formatter_by_series is not None:
                rows.append((s.name, x_formatter_by_series(s, x), y))
            else:
                rows.append((s.name, x_formatter(x) if x_formatter else x, y))
    return rows

def write_long_csv(
    path: str,
    series: List[Series],
    delimiter: str = ",",
    *,
    x_formatter: Optional[Callable[[float], str]] = None,
    x_formatter_by_series: Optional[Callable[[Series, float], str]] = None,
) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=delimiter)
        w.writerow(["series", "x", "y"])
        for s in series:
            if not s.enabled:
                continue
            for i, (x,y) in enumerate(s.points):
                ok = True
                if s.point_enabled and i < len(s.point_enabled):
                    ok = s.point_enabled[i]
                if not ok:
                    continue
                if x_formatter_by_series is not None:
                    w.writerow([s.name, x_formatter_by_series(s, x), y])
                else:
                    w.writerow([s.name, x_formatter(x) if x_formatter else x, y])

def write_wide_csv(
    path: str,
    x_grid: List[float | str],
    series: List[Series],
    delimiter: str = ",",
    *,
    x_formatter: Optional[Callable[[float], str]] = None,
) -> None:
    enabled = [s for s in series if s.enabled]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=delimiter)
        w.writerow(["x"] + [s.name for s in enabled])
        for i, x in enumerate(x_grid):
            row: List[Optional[float] | str] = [x_formatter(x) if x_formatter else x]
            for s in enabled:
                if i >= len(s.points):
                    row.append("")
                    continue
                ok = True
                if s.point_enabled and i < len(s.point_enabled):
                    ok = s.point_enabled[i]
                if not ok:
                    row.append("")
                    continue
                row.append(s.points[i][1])
            w.writerow(row)

def wide_csv_string(
    x_grid: List[float | str],
    series: List[Series],
    delimiter: str = ",",
    *,
    x_formatter: Optional[Callable[[float], str]] = None,
) -> str:
    import io
    buf = io.StringIO()
    enabled = [s for s in series if s.enabled]
    w = csv.writer(buf, delimiter=delimiter, lineterminator="\n")
    w.writerow(["x"] + [s.name for s in enabled])
    for i, x in enumerate(x_grid):
        row = [x_formatter(x) if x_formatter else x]
        for s in enabled:
            if i >= len(s.points):
                row.append("")
                continue
            ok = True
            if s.point_enabled and i < len(s.point_enabled):
                ok = s.point_enabled[i]
            if not ok:
                row.append("")
                continue
            row.append(s.points[i][1])
        w.writerow(row)
    return buf.getvalue().rstrip()

def long_csv_string(
    series: List[Series],
    delimiter: str = ",",
    *,
    x_formatter: Optional[Callable[[float], str]] = None,
    x_formatter_by_series: Optional[Callable[[Series, float], str]] = None,
) -> str:
    import io
    buf = io.StringIO()
    w = csv.writer(buf, delimiter=delimiter, lineterminator="\n")
    w.writerow(["series","x","y"])
    for s in series:
        if not s.enabled:
            continue
        for i,(x,y) in enumerate(s.points):
            ok = True
            if s.point_enabled and i < len(s.point_enabled):
                ok = s.point_enabled[i]
            if not ok:
                continue
            if x_formatter_by_series is not None:
                w.writerow([s.name, x_formatter_by_series(s, x), y])
            else:
                w.writerow([s.name, x_formatter(x) if x_formatter else x, y])
    return buf.getvalue().rstrip()

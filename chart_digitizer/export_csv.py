
from __future__ import annotations

import csv
from typing import List, Optional, Tuple
from .model import Series

def series_to_long_rows(series: List[Series]) -> List[Tuple[str, float, float]]:
    rows: List[Tuple[str,float,float]] = []
    for s in series:
        if not s.enabled:
            continue
        for (x,y), ok in zip(s.points, s.point_enabled or [True]*len(s.points)):
            if not ok:
                continue
            rows.append((s.name, x, y))
    return rows

def write_long_csv(path: str, series: List[Series], delimiter: str = ",") -> None:
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
                w.writerow([s.name, x, y])

def write_wide_csv(path: str, x_grid: List[float], series: List[Series], delimiter: str = ",") -> None:
    enabled = [s for s in series if s.enabled]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=delimiter)
        w.writerow(["x"] + [s.name for s in enabled])
        for i, x in enumerate(x_grid):
            row: List[Optional[float] | str] = [x]
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

def wide_csv_string(x_grid: List[float], series: List[Series], delimiter: str = ",") -> str:
    import io
    buf = io.StringIO()
    enabled = [s for s in series if s.enabled]
    w = csv.writer(buf, delimiter=delimiter, lineterminator="\n")
    w.writerow(["x"] + [s.name for s in enabled])
    for i, x in enumerate(x_grid):
        row = [x]
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

def long_csv_string(series: List[Series], delimiter: str = ",") -> str:
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
            w.writerow([s.name, x, y])
    return buf.getvalue().rstrip()

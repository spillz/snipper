
from __future__ import annotations

import csv
from typing import Callable, Iterable, List, Optional
from .data_model import Series


def _is_point_enabled(s: Series, i: int) -> bool:
    if not s.point_enabled:
        return True
    if i >= len(s.point_enabled):
        return True
    return bool(s.point_enabled[i])


def _iter_long_rows(
    series: List[Series],
    *,
    x_formatter: Optional[Callable[[float], str]] = None,
    x_formatter_by_series: Optional[Callable[[Series, float], str]] = None,
) -> Iterable[List[float | str]]:
    for s in series:
        if not s.enabled:
            continue
        for i, (x, y) in enumerate(s.points):
            if not _is_point_enabled(s, i):
                continue
            if x_formatter_by_series is not None:
                x_out = x_formatter_by_series(s, x)
            else:
                x_out = x_formatter(x) if x_formatter else x
            yield [s.name, x_out, y]


def _iter_wide_rows(
    x_grid: List[float | str],
    series: List[Series],
    *,
    x_formatter: Optional[Callable[[float], str]] = None,
) -> Iterable[List[Optional[float] | str]]:
    enabled = [s for s in series if s.enabled]
    yield ["x"] + [s.name for s in enabled]
    for i, x in enumerate(x_grid):
        row: List[Optional[float] | str] = [x_formatter(x) if x_formatter else x]
        for s in enabled:
            if i >= len(s.points) or not _is_point_enabled(s, i):
                row.append("")
                continue
            row.append(s.points[i][1])
        yield row

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
        for row in _iter_long_rows(series, x_formatter=x_formatter, x_formatter_by_series=x_formatter_by_series):
            w.writerow(row)

def write_wide_csv(
    path: str,
    x_grid: List[float | str],
    series: List[Series],
    delimiter: str = ",",
    *,
    x_formatter: Optional[Callable[[float], str]] = None,
) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=delimiter)
        for row in _iter_wide_rows(x_grid, series, x_formatter=x_formatter):
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
    w = csv.writer(buf, delimiter=delimiter, lineterminator="\n")
    for row in _iter_wide_rows(x_grid, series, x_formatter=x_formatter):
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
    for row in _iter_long_rows(series, x_formatter=x_formatter, x_formatter_by_series=x_formatter_by_series):
        w.writerow(row)
    return buf.getvalue().rstrip()

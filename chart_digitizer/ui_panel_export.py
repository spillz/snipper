from __future__ import annotations

import tkinter as tk
from tkinter import ttk, filedialog
from typing import Callable, List, Optional, Tuple

import numpy as np

from .calibration import AxisScale
from .data_model import Series
from .export_csv import wide_csv_string, long_csv_string, write_wide_csv, write_long_csv


class ExportPanel:
    def __init__(
        self,
        owner,
        parent: tk.Widget,
        *,
        on_append_csv: Callable[[], None],
        on_export_csv: Callable[[], None],
        on_close: Callable[[], None],
    ) -> None:
        self.owner = owner
        frame = ttk.Frame(parent)
        self.frame = frame
        owner._export_frame = frame
        frame.pack(side="bottom", fill="x", pady=(8, 0))

        ttk.Button(frame, text="Append CSV", command=on_append_csv).pack(side="left")
        ttk.Button(frame, text="Export CSV...", command=on_export_csv).pack(side="left", padx=(8, 0))
        ttk.Button(frame, text="Close", command=on_close).pack(side="right")
        frame.update_idletasks()
        frame.configure(height=frame.winfo_reqheight())
        frame.pack_propagate(False)


class Exporter:
    def __init__(self, owner) -> None:
        object.__setattr__(self, "owner", owner)

    def __getattr__(self, name):
        return getattr(self.owner, name)

    def __setattr__(self, name, value) -> None:
        if name == "owner":
            object.__setattr__(self, name, value)
            return
        setattr(self.owner, name, value)

    def _apply_stacked_deltas(self, ordered: List[Series]) -> List[Series]:
        if not any(getattr(s, "stacked", False) for s in ordered):
            return ordered

        prev_by_cal: dict[tuple, Series] = {}
        deltas: List[Series] = []
        for s in ordered:
            if not getattr(s, "stacked", False):
                deltas.append(s)
                continue

            key = self.owner.calibrator._calibration_key_from_series(s)
            prev = prev_by_cal.get(key)

            ds = Series(
                id=s.id,
                name=s.name,
                color_bgr=s.color_bgr,
                chart_kind=s.chart_kind,
                stacked=s.stacked,
                stride_mode=s.stride_mode,
                prefer_outline=s.prefer_outline,
                enabled=s.enabled,
                calibration=s.calibration,
            )
            ds.points = []
            ds.point_enabled = []
            for i, (x, y) in enumerate(s.points):
                ok = bool(s.point_enabled[i]) if s.point_enabled else True
                if prev is None:
                    ds.points.append((float(x), float(y)))
                    ds.point_enabled.append(ok)
                else:
                    ok_prev = bool(prev.point_enabled[i]) if prev.point_enabled else True
                    ok2 = ok and ok_prev
                    y_prev = float(prev.points[i][1])
                    ds.points.append((float(x), float(y) - y_prev))
                    ds.point_enabled.append(ok2)
            deltas.append(ds)
            prev_by_cal[key] = s

        return deltas


    def _prepare_wide_export(self, enabled: List[Series]) -> Tuple[List[float], List[Series]]:
        """
        Prepare x_grid and aligned per-series points for wide CSV export.

        - For continuous stride (line/area/column), assumes series already aligned to a common x-grid.
        - For categorical stride, clusters x positions across series into category bins and aligns each series
          to those bins (handles grouped/overlapping bars with slight x offsets).
        - For stacked charts, if series are marked stacked=True, emits deltas between successive boundaries.
        """
        if not enabled:
            return [], []

        # Separate scatter vs non-scatter
        non_scatter = [s for s in enabled if getattr(s, "chart_kind", s.mode) != "scatter"]
        if not non_scatter:
            return [], []

        # If every non-scatter series is already on a grid (line/continuous), use first series grid.
        # Otherwise, build categorical bins.
        any_categorical = any(getattr(s, "stride_mode", "continuous") == "categorical" for s in non_scatter)

        if not any_categorical:
            x_grid = [float(x) for (x, _y) in non_scatter[0].points]
            # Ensure all series are aligned; if not, we still export as-is (best-effort).
            return x_grid, self._apply_stacked_deltas(non_scatter)

        # --- categorical binning ---
        labels = self.owner.calibrator._parse_categories(non_scatter[0].calibration.categories)
        if labels:
            count = len(labels)
        else:
            count = 0

        # If categories are provided, snap to fixed centers based on x0/x1 pixel anchors.
        if count > 0:
            x_grid = labels

            def _axis_for_series(s: Series) -> str:
                return "y" if getattr(s, "chart_kind", s.mode) == "bar" else "x"

            def _nearest_center_idx(px: float, centers_px: List[float]) -> Tuple[int, float]:
                j = int(np.argmin([abs(px - c) for c in centers_px]))
                return j, abs(px - centers_px[j])

            aligned_series: List[Series] = []
            for s in non_scatter:
                axis = _axis_for_series(s)
                axis_px = s.calibration.y_axis_px if axis == "y" else s.calibration.x_axis_px
                centers_px = self.owner.calibrator._category_centers_px(count, axis=axis, axis_px=axis_px)

                y_by_bin: List[Optional[float]] = [None] * count
                en_by_bin: List[bool] = [False] * count
                dist_by_bin: List[Optional[float]] = [None] * count

                pe = s.point_enabled if s.point_enabled else [True] * len(s.points)
                for i, (_x, y) in enumerate(s.points):
                    if not s.px_points or i >= len(s.px_points):
                        continue
                    px = s.px_points[i][1] if axis == "y" else s.px_points[i][0]
                    b, dist = _nearest_center_idx(float(px), centers_px)
                    if dist_by_bin[b] is None or dist < dist_by_bin[b]:
                        y_by_bin[b] = float(y)
                        en_by_bin[b] = bool(pe[i])
                        dist_by_bin[b] = dist

                ss = Series(
                    id=s.id,
                    name=s.name,
                    color_bgr=s.color_bgr,
                    chart_kind=getattr(s, "chart_kind", s.mode),
                    stacked=bool(getattr(s, "stacked", False)),
                    stride_mode="categorical",
                    prefer_outline=bool(getattr(s, "prefer_outline", True)),
                    enabled=s.enabled,
                    calibration=s.calibration,
                )
                ss.points = [(float(i + 1), 0.0 if y_by_bin[i] is None else float(y_by_bin[i])) for i in range(count)]
                ss.point_enabled = [bool(en_by_bin[i]) for i in range(count)]
                aligned_series.append(ss)

            # --- stacked deltas (optional, grouped by calibration) ---
            ordered = []
            by_id = {s.id: s for s in aligned_series}
            for s0 in non_scatter:
                if s0.id in by_id:
                    ordered.append(by_id[s0.id])

            aligned_series = self._apply_stacked_deltas(ordered)
            return x_grid, aligned_series

        # --- categorical binning fallback (auto) ---
        xs: List[float] = []
        for s in non_scatter:
            xs.extend([float(x) for (x, _y) in (s.points or [])])

        xs = sorted(set(xs))
        if not xs:
            return [], non_scatter

        if len(xs) == 1:
            centers = [xs[0]]
        else:
            diffs = [abs(xs[i+1] - xs[i]) for i in range(len(xs)-1)]
            diffs = [d for d in diffs if d > 0]
            med = float(np.median(diffs)) if diffs else 1.0
            thresh = 0.35 * med

            centers: List[float] = []
            cur = [xs[0]]
            for v in xs[1:]:
                if abs(v - cur[-1]) <= thresh:
                    cur.append(v)
                else:
                    centers.append(float(np.mean(cur)))
                    cur = [v]
            centers.append(float(np.mean(cur)))

        x_grid = [float(i + 1) for i in range(len(centers))]

        def nearest_bin(x: float) -> int:
            j = int(np.argmin([abs(x - c) for c in centers]))
            return j

        aligned_series: List[Series] = []
        for s in non_scatter:
            y_by_bin: List[Optional[float]] = [None] * len(centers)
            en_by_bin: List[bool] = [False] * len(centers)

            # ensure point_enabled exists
            pe = s.point_enabled if s.point_enabled else [True] * len(s.points)

            for i, (x, y) in enumerate(s.points):
                b = nearest_bin(float(x))
                y_by_bin[b] = float(y)
                en_by_bin[b] = bool(pe[i])

            # Build a grid-aligned series (pad missing as 0 but disabled; wide_csv_string will write blank if disabled)
            ss = Series(
                id=s.id,
                name=s.name,
                color_bgr=s.color_bgr,
                chart_kind=getattr(s, "chart_kind", s.mode),
                stacked=bool(getattr(s, "stacked", False)),
                stride_mode="categorical",
                prefer_outline=bool(getattr(s, "prefer_outline", True)),
                enabled=s.enabled,
                calibration=s.calibration,
            )
            ss.points = [(x_grid[i], 0.0 if y_by_bin[i] is None else float(y_by_bin[i])) for i in range(len(x_grid))]
            ss.point_enabled = [bool(en_by_bin[i]) for i in range(len(x_grid))]
            aligned_series.append(ss)

        # --- stacked deltas (optional, grouped by calibration) ---
        ordered = []
        by_id = {s.id: s for s in aligned_series}
        for s0 in non_scatter:
            if s0.id in by_id:
                ordered.append(by_id[s0.id])

        aligned_series = self._apply_stacked_deltas(ordered)
        return x_grid, aligned_series


    def _x_formatter_for_series(self, s: Series):
        cal = self.owner.calibrator._build_calibration(s.calibration)
        if getattr(s, "chart_kind", s.mode) == "bar" and cal.y.scale == AxisScale.DATE:
            return cal.format_y_value
        if cal.x.scale == AxisScale.DATE:
            return cal.format_x_value
        return None


    def _format_x_for_series(self, s: Series, x: float):
        fmt = self._x_formatter_for_series(s)
        return fmt(x) if fmt else x


    def _append_csv(self):
        if not self.series:
            self._show_info("Append CSV", "No series to export yet. Use 'Add series' first.")
            return

        enabled = [s for s in self.series if s.enabled]
        if not enabled:
            return
        selected = None
        if self._active_series_id is not None:
            selected = self.owner.series_actor._get_series(self._active_series_id)
        if selected is None or selected not in enabled:
            selected = enabled[0]
        x_formatter = self._x_formatter_for_series(selected) if selected else None

        # If every enabled series is scatter, export long; otherwise export wide.
        if all(getattr(s, "chart_kind", s.mode) == "scatter" for s in enabled):
            txt = long_csv_string(enabled, x_formatter_by_series=self._format_x_for_series)
        else:
            x_grid, ser = self._prepare_wide_export(enabled)
            if not x_grid or not ser:
                self._show_info("Append CSV", "Nothing to export (no valid points).")
                return
            if x_grid and isinstance(x_grid[0], str):
                x_formatter = None
            txt = wide_csv_string(x_grid, ser, x_formatter=x_formatter)

        self._on_append_text(txt)


    def _export_csv(self):
        if not self.series:
            self._show_info("Export CSV", "No series to export yet. Use 'Add series' first.")
            return

        enabled = [s for s in self.series if s.enabled]
        if not enabled:
            self._show_info("Export CSV", "All series are disabled.")
            return

        path = filedialog.asksaveasfilename(
            parent=self,
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not path:
            return

        selected = None
        if self._active_series_id is not None:
            selected = self.owner.series_actor._get_series(self._active_series_id)
        if selected is None or selected not in enabled:
            selected = enabled[0]
        x_formatter = self._x_formatter_for_series(selected) if selected else None

        if all(getattr(s, "chart_kind", s.mode) == "scatter" for s in enabled):
            write_long_csv(path, enabled, x_formatter_by_series=self._format_x_for_series)
        else:
            x_grid, ser = self._prepare_wide_export(enabled)
            if not x_grid or not ser:
                self._show_info("Export CSV", "Nothing to export (no valid points).")
                return
            if x_grid and isinstance(x_grid[0], str):
                x_formatter = None
            write_wide_csv(path, x_grid, ser, x_formatter=x_formatter)

        self._show_info("Export CSV", f"Saved:\n{path}")


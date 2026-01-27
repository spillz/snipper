from __future__ import annotations

import calendar
from datetime import datetime, timezone
import tkinter as tk
from tkinter import ttk
from typing import Callable, List, Optional, Tuple

import numpy as np

from .calibration import AxisScale, Calibration
from .cv_utils import require_cv2, color_distance_mask
from .data_model import Series
from .extract import (
    build_x_grid,
    build_x_grid_aligned,
    extract_line_series,
    extract_scatter_series,
    extract_scatter_series_fixed_stride,
    enforce_line_grid,
    detect_runs_centers_along_x,
    detect_runs_centers_along_y,
)


class ExtractionPanel:
    def __init__(
        self,
        owner,
        parent: tk.Widget,
        *,
        on_prefer_outline_change: Callable[[], None],
        on_mask_invert_change: Callable[[], None],
        on_clear_mask: Callable[[], None],
        on_rerun_extraction: Callable[[], None],
    ) -> None:
        self.owner = owner
        frame = ttk.LabelFrame(parent, text="Extraction", padding=8)
        self.frame = frame
        owner._extraction_frame = frame
        frame.pack(side="top", fill="x", pady=(8, 0))

        row = ttk.Frame(frame)
        row.pack(fill="x")
        ttk.Label(row, text="Color tol:").pack(side="left")
        ttk.Entry(row, textvariable=owner.var_tol, width=6).pack(side="left", padx=(6, 0))

        match_row = ttk.Frame(frame)
        match_row.pack(fill="x", pady=(6, 0))
        ttk.Label(match_row, text="Match thresh:").pack(side="left")
        owner.ent_match_thresh = ttk.Entry(match_row, textvariable=owner.var_scatter_match_thresh, width=6)
        owner.ent_match_thresh.pack(side="left", padx=(6, 0))

        owner.chk_prefer_outline = ttk.Checkbutton(
            frame,
            text="Span detection (for bars/areas)",
            variable=owner.var_prefer_outline,
            command=on_prefer_outline_change,
        )
        owner.chk_prefer_outline.pack(side="top", anchor="w", pady=(6, 0))

        mask_row = ttk.Frame(frame)
        mask_row.pack(fill="x", pady=(6, 0))
        ttk.Label(mask_row, text="Mask:").pack(side="left")
        owner.chk_mask_invert = ttk.Checkbutton(
            mask_row,
            text="Invert",
            variable=owner.var_mask_invert,
            command=on_mask_invert_change,
        )
        owner.chk_mask_invert.pack(side="left", padx=(6, 0))
        owner.btn_clear_mask = ttk.Button(mask_row, text="Clear", command=on_clear_mask)
        owner.btn_clear_mask.pack(side="left", padx=(6, 0))

        rerun_row = ttk.Frame(frame)
        rerun_row.pack(fill="x", pady=(6, 0))
        ttk.Button(rerun_row, text="Re-run extraction", command=on_rerun_extraction).pack(side="left")
        ttk.Checkbutton(rerun_row, text="Auto rerun", variable=owner.var_auto_rerun).pack(side="left", padx=(8, 0))


class Extractor:
    def __init__(self, owner) -> None:
        object.__setattr__(self, "owner", owner)

    def __getattr__(self, name):
        return getattr(self.owner, name)

    def __setattr__(self, name, value) -> None:
        if name == "owner":
            object.__setattr__(self, name, value)
            return
        setattr(self.owner, name, value)

    def _on_prefer_outline_change(self) -> None:
        if getattr(self, "_suppress_extraction_change", False):
            return
        s = self.owner.series_actor._get_series(self._active_series_id) if self._active_series_id is not None else None
        if s is None:
            return
        s.prefer_outline = bool(self.var_prefer_outline.get())
        if self.var_auto_rerun.get():
            try:
                self._extract_series(s)
            except Exception as e:
                self._show_error("Series extraction failed", str(e))
                return
        self.owner.series_actor._update_tree_row(s)
        self.owner.canvas_actor._redraw_overlay()


    def _on_extraction_setting_change(self) -> None:
        if getattr(self, "_suppress_extraction_change", False):
            return
        s = self.owner.series_actor._get_series(self._active_series_id) if self._active_series_id is not None else None
        if s is None:
            return
        try:
            s.color_tol = int(self.var_tol.get())
        except (tk.TclError, ValueError):
            return
        try:
            s.scatter_match_thresh = float(self.var_scatter_match_thresh.get())
        except (tk.TclError, ValueError):
            return
        if self.var_auto_rerun.get():
            try:
                self._extract_series(s)
            except Exception as e:
                self._show_error("Series extraction failed", str(e))
                return
        self.owner.series_actor._update_tree_row(s)
        self.owner.canvas_actor._redraw_overlay()


    def _update_extraction_controls(self) -> None:
        kind = self.series_mode.get()
        if self._active_series_id is not None:
            s = self.owner.series_actor._get_series(self._active_series_id)
            if s is not None:
                kind = getattr(s, "chart_kind", s.mode)
        is_scatter = (kind == "scatter")
        if getattr(self, "ent_match_thresh", None) is not None:
            self.ent_match_thresh.configure(state=("normal" if is_scatter else "disabled"))
        span_ok = kind in ("column", "bar", "area")
        if getattr(self, "chk_prefer_outline", None) is not None:
            self.chk_prefer_outline.configure(state=("normal" if span_ok else "disabled"))


    def _rerun_active_series(self) -> None:
        if self._active_series_id is None:
            return
        s = self.owner.series_actor._get_series(self._active_series_id)
        if s is None:
            return
        try:
            self._extract_series(s)
        except Exception as e:
            self._show_error("Series extraction failed", str(e))
            return
        self.owner.series_actor._update_tree_row(s)
        self.owner.canvas_actor._redraw_overlay()


    def _last_day_of_month(year: int, month: int) -> int:
        return int(calendar.monthrange(int(year), int(month))[1])

    @classmethod
    def _add_months(cls, dt: datetime, months: int) -> datetime:
        total = (dt.year * 12) + (dt.month - 1) + int(months)
        year = total // 12
        month = (total % 12) + 1
        day = min(dt.day, cls._last_day_of_month(year, month))
        return dt.replace(year=year, month=month, day=day)


    def _build_date_grid_aligned(
        self,
        xmin_val: float,
        xmax_val: float,
        step: float,
        unit: str,
        *,
        anchor: float,
    ) -> List[float]:
        if step <= 0:
            raise ValueError("x_step must be > 0")
        if xmax_val < xmin_val:
            xmin_val, xmax_val = xmax_val, xmin_val

        unit = (unit or "days").lower()
        seconds_per = {
            "seconds": 1.0,
            "minutes": 60.0,
            "hours": 3600.0,
            "days": 86400.0,
            "weeks": 86400.0 * 7,
        }
        if unit in seconds_per:
            step_sec = float(step) * seconds_per[unit]
            return build_x_grid_aligned(xmin_val, xmax_val, step_sec, anchor=float(anchor))

        months_per = {"months": 1, "quarters": 3, "years": 12}
        if unit not in months_per:
            raise ValueError(f"Unsupported date unit: {unit}")

        months_step = int(max(1, round(float(step) * months_per[unit])))
        anchor_dt = datetime.fromtimestamp(float(anchor), tz=timezone.utc)
        xmin_dt = datetime.fromtimestamp(float(xmin_val), tz=timezone.utc)
        xmax_dt = datetime.fromtimestamp(float(xmax_val), tz=timezone.utc)

        anchor_idx = (anchor_dt.year * 12) + (anchor_dt.month - 1)
        xmin_idx = (xmin_dt.year * 12) + (xmin_dt.month - 1)
        delta = xmin_idx - anchor_idx
        k_start = int(delta // months_step)

        cur = self._add_months(anchor_dt, k_start * months_step)
        while cur < xmin_dt:
            k_start += 1
            cur = self._add_months(anchor_dt, k_start * months_step)

        xs: List[float] = []
        while cur <= xmax_dt:
            xs.append(cur.timestamp())
            cur = self._add_months(cur, months_step)

        return xs



    def _apply_series_extraction_to_ui(self, s: Series) -> None:
        self._suppress_extraction_change = True
        try:
            self.var_tol.set(int(getattr(s, "color_tol", self.var_tol.get())))
            self.var_scatter_match_thresh.set(float(getattr(s, "scatter_match_thresh", self.var_scatter_match_thresh.get())))
            self.var_prefer_outline.set(bool(getattr(s, "prefer_outline", True)))
        finally:
            self._suppress_extraction_change = False
        self._update_extraction_controls()


    def _extract_series(self, s: Series):
        """
        Populate s.px_points, s.points, s.point_enabled based on s.chart_kind.

        - line: continuity-first line tracking sampled to an aligned X grid
        - scatter: blob/marker detection
        - column/area: boundary read (top/bottom vs baseline) sampled by stride mode
        - bar: boundary read (left/right vs baseline) along categorical Y centers
        """
        roi = s.calibration.roi_px
        tol = int(getattr(s, "color_tol", self.var_tol.get()))
        cal = self.owner.calibrator._build_calibration(s.calibration)

        kind = getattr(s, "chart_kind", getattr(s, "mode", "line"))
        stride = getattr(s, "stride_mode", "continuous")
        prefer_outline = bool(getattr(s, "prefer_outline", True))

        if s.seed_px is None:
            raise ValueError("Series has no seed point. Click in the chart to create a series.")

        # ---------------- Scatter ----------------
        if kind == "scatter":
            sample_mode = getattr(s.calibration, "sample_mode", "free")
            if sample_mode == "free":
                pts_px = extract_scatter_series(
                    self._bgr,
                    roi,
                    s.color_bgr,
                    tol,
                    seed_px=s.seed_px,
                    seed_bbox_px=s.seed_bbox_px,
                    seed_bboxes_px=(s.scatter_seed_bboxes_px or None),
                    template_match_thresh=float(getattr(s, "scatter_match_thresh", self.var_scatter_match_thresh.get())),
                    use_template_matching=True,
                )
                pts_px = [p for p in pts_px if self.owner.canvas_actor._mask_allows_point(s, p[0], p[1])]
                s.px_points = pts_px
                s.points = [(float(cal.x_px_to_data(x)), float(cal.y_px_to_data(y))) for (x, y) in pts_px]
                s.point_enabled = [True] * len(s.points)
                self.owner.series_actor._update_tree_row(s)
                return

            if sample_mode == "fixed_x":
                x0_val = self.owner.calibrator._require_value(s.calibration.x0_val, "x0", cal=s.calibration)
                roi_xmin_px, _, roi_xmax_px, _ = roi
                xmin = float(cal.x_px_to_data(roi_xmin_px))
                xmax = float(cal.x_px_to_data(roi_xmax_px))
                if xmin > xmax:
                    xmin, xmax = xmax, xmin

                step = float(s.calibration.x_step)
                x_grid: List[float] = []
                xpx_grid: List[int] = []
                if s.calibration.x_scale == AxisScale.CATEGORICAL.value:
                    labels = self.owner.calibrator._parse_categories(s.calibration.categories)
                    if labels:
                        centers = self.owner.calibrator._category_centers_px(len(labels), axis="x", axis_px=s.calibration.x_axis_px)
                        xpx_grid = [int(round(p)) for p in centers]
                        x_grid = [float(i + 1) for i in range(len(labels))]
                if not x_grid:
                    if s.calibration.x_scale == AxisScale.DATE.value:
                        x_grid = self._build_date_grid_aligned(
                            xmin, xmax, step, s.calibration.x_step_unit, anchor=x0_val
                        )
                    else:
                        x_grid = build_x_grid_aligned(xmin, xmax, step, anchor=x0_val)
                    xpx_grid = [int(round(cal.x_data_to_px(x))) for x in x_grid]

                ypx_raw = extract_scatter_series_fixed_stride(
                    self._bgr,
                    roi,
                    s.color_bgr,
                    tol,
                    mode="fixed_x",
                    grid_px=xpx_grid,
                    seed_px=s.seed_px,
                    seed_bbox_px=s.seed_bbox_px,
                    seed_bboxes_px=(s.scatter_seed_bboxes_px or None),
                    template_match_thresh=float(getattr(s, "scatter_match_thresh", self.var_scatter_match_thresh.get())),
                )
                for i, ypx in enumerate(ypx_raw):
                    if ypx is None:
                        continue
                    if not self.owner.canvas_actor._mask_allows_point(s, int(xpx_grid[i]), int(ypx)):
                        ypx_raw[i] = None

                y_data_raw: List[Optional[float]] = []
                for ypx in ypx_raw:
                    y_data_raw.append(None if ypx is None else float(cal.y_px_to_data(int(ypx))))

                y0_val = self.owner.calibrator._require_value(s.calibration.y0_val, "y0", cal=s.calibration)
                y1_val = self.owner.calibrator._require_value(s.calibration.y1_val, "y1", cal=s.calibration)
                fallback_y = 0.5 * (y0_val + y1_val)
                y_filled = enforce_line_grid(list(x_grid), y_data_raw, fallback_y=fallback_y)

                px_points: List[Tuple[int, int]] = []
                enabled: List[bool] = []
                for i, xpx in enumerate(xpx_grid):
                    if ypx_raw[i] is None:
                        ypx = int(round(cal.y_data_to_px(y_filled[i])))
                        enabled.append(False)
                    else:
                        ypx = int(ypx_raw[i])
                        enabled.append(True)
                    px_points.append((int(xpx), int(ypx)))

                s.px_points = px_points
                s.points = [(float(x_grid[i]), float(y_filled[i])) for i in range(len(x_grid))]
                s.point_enabled = enabled
                self.owner.series_actor._update_tree_row(s)
                return

            if sample_mode == "fixed_y":
                y0_val = self.owner.calibrator._require_value(s.calibration.y0_val, "y0", cal=s.calibration)
                roi_ymin_px, roi_ymax_px = roi[1], roi[3]
                ymin = float(cal.y_px_to_data(roi_ymin_px))
                ymax = float(cal.y_px_to_data(roi_ymax_px))
                if ymin > ymax:
                    ymin, ymax = ymax, ymin

                step = float(s.calibration.y_step)
                y_grid: List[float] = []
                ypx_grid: List[int] = []
                if s.calibration.y_scale == AxisScale.CATEGORICAL.value:
                    labels = self.owner.calibrator._parse_categories(s.calibration.categories)
                    if labels:
                        centers = self.owner.calibrator._category_centers_px(len(labels), axis="y", axis_px=s.calibration.y_axis_px)
                        ypx_grid = [int(round(p)) for p in centers]
                        y_grid = [float(i + 1) for i in range(len(labels))]
                if not y_grid:
                    if s.calibration.y_scale == AxisScale.DATE.value:
                        y_grid = self._build_date_grid_aligned(
                            ymin, ymax, step, s.calibration.y_step_unit, anchor=y0_val
                        )
                    else:
                        y_grid = build_x_grid_aligned(ymin, ymax, step, anchor=y0_val)
                    ypx_grid = [int(round(cal.y_data_to_px(y))) for y in y_grid]

                xpx_raw = extract_scatter_series_fixed_stride(
                    self._bgr,
                    roi,
                    s.color_bgr,
                    tol,
                    mode="fixed_y",
                    grid_px=ypx_grid,
                    seed_px=s.seed_px,
                    seed_bbox_px=s.seed_bbox_px,
                    seed_bboxes_px=(s.scatter_seed_bboxes_px or None),
                    template_match_thresh=float(getattr(s, "scatter_match_thresh", self.var_scatter_match_thresh.get())),
                )
                for i, xpx in enumerate(xpx_raw):
                    if xpx is None:
                        continue
                    if not self.owner.canvas_actor._mask_allows_point(s, int(xpx), int(ypx_grid[i])):
                        xpx_raw[i] = None

                x_data_raw: List[Optional[float]] = []
                for xpx in xpx_raw:
                    x_data_raw.append(None if xpx is None else float(cal.x_px_to_data(int(xpx))))

                x0_val = self.owner.calibrator._require_value(s.calibration.x0_val, "x0", cal=s.calibration)
                x1_val = self.owner.calibrator._require_value(s.calibration.x1_val, "x1", cal=s.calibration)
                fallback_x = 0.5 * (x0_val + x1_val)
                x_filled = enforce_line_grid(list(y_grid), x_data_raw, fallback_y=fallback_x)

                px_points = []
                enabled = []
                for i, ypx in enumerate(ypx_grid):
                    if xpx_raw[i] is None:
                        xpx = int(round(cal.x_data_to_px(x_filled[i])))
                        enabled.append(False)
                    else:
                        xpx = int(xpx_raw[i])
                        enabled.append(True)
                    px_points.append((int(xpx), int(ypx)))

                s.px_points = px_points
                s.points = [(float(x_filled[i]), float(y_grid[i])) for i in range(len(y_grid))]
                s.point_enabled = enabled
                self.owner.series_actor._update_tree_row(s)
                return

            raise ValueError(f"Unsupported scatter sample mode: {sample_mode}")

        # Helper: build a uint8 mask for this series inside ROI
        require_cv2()
        import cv2

        x0, y0, x1, y1 = roi
        roi_img = self._bgr[y0:y1, x0:x1]
        m = color_distance_mask(roi_img, s.color_bgr, tol)
        mask = (m.astype(np.uint8) * 255) if (m.dtype != np.uint8 or m.max() <= 1) else m.copy()

        # Optional: prefer outline/edge pixels (useful for stroked bars/areas)
        if prefer_outline:
            er = cv2.erode((mask > 0).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
            edge = ((mask > 0) & (er == 0)).astype(np.uint8) * 255
            # if edge collapses (e.g., extremely thin fills), fall back to fill
            if int(edge.sum()) > 0:
                mask = edge

        mask = self.owner.canvas_actor._apply_series_mask_to_roi(s, roi, mask)

        def _scan_col_edge(x_local: int, *, prefer_top: bool, y_hint_local: int | None, band: int = 10) -> int | None:
            col = (mask[:, x_local] > 0)
            if not np.any(col):
                return None
            if y_hint_local is not None:
                lo = max(0, y_hint_local - band)
                hi = min(mask.shape[0], y_hint_local + band + 1)
                sl = col[lo:hi]
                if np.any(sl):
                    idx = np.where(sl)[0]
                    return int(lo + (idx[0] if prefer_top else idx[-1]))
            idx = np.where(col)[0]
            return int(idx[0] if prefer_top else idx[-1])

        def _scan_row_edge(y_local: int, *, prefer_left: bool, x_hint_local: int | None, band: int = 10) -> int | None:
            row = (mask[y_local, :] > 0)
            if not np.any(row):
                return None
            if x_hint_local is not None:
                lo = max(0, x_hint_local - band)
                hi = min(mask.shape[1], x_hint_local + band + 1)
                sl = row[lo:hi]
                if np.any(sl):
                    idx = np.where(sl)[0]
                    return int(lo + (idx[0] if prefer_left else idx[-1]))
            idx = np.where(row)[0]
            return int(idx[0] if prefer_left else idx[-1])

        # ---------------- Line ----------------
        if kind == "line":
            x0_val = self.owner.calibrator._require_value(s.calibration.x0_val, "x0", cal=s.calibration)
            roi_xmin_px, _, roi_xmax_px, _ = roi
            xmin = float(cal.x_px_to_data(roi_xmin_px))
            xmax = float(cal.x_px_to_data(roi_xmax_px))
            if xmin > xmax:
                xmin, xmax = xmax, xmin

            step = float(s.calibration.x_step)
            x_grid: List[float] = []
            xpx_grid: List[int] = []
            if stride == "categorical":
                labels = self.owner.calibrator._parse_categories(s.calibration.categories)
                if labels:
                    centers = self.owner.calibrator._category_centers_px(len(labels), axis="x", axis_px=s.calibration.x_axis_px)
                    xpx_grid = [int(round(p)) for p in centers]
                    x_grid = [float(i + 1) for i in range(len(labels))]
            if not x_grid:
                if s.calibration.x_scale == AxisScale.DATE.value:
                    x_grid = self._build_date_grid_aligned(
                        xmin, xmax, step, s.calibration.x_step_unit, anchor=x0_val
                    )
                else:
                    x_grid = build_x_grid_aligned(xmin, xmax, step, anchor=x0_val)
                xpx_grid = [int(round(cal.x_data_to_px(x))) for x in x_grid]

            roi_h = max(1, y1 - y0)
            band_reacq_px = max(40, int(0.25 * roi_h))
            max_jump_px = max(30, int(0.20 * roi_h))

            px_pts, ypx_raw = extract_line_series(
                self._bgr, roi, s.color_bgr, tol, xpx_grid,
                seed_px=s.seed_px,
                extra_seeds_px=(s.extra_seeds_px or None),
                band_reacq_px=band_reacq_px,
                max_jump_px=max_jump_px,
            )
            for i, ypx in enumerate(ypx_raw):
                if ypx is None:
                    continue
                if not self.owner.canvas_actor._mask_allows_point(s, int(xpx_grid[i]), int(ypx)):
                    ypx_raw[i] = None

            y_data_raw: List[Optional[float]] = []
            for ypx in ypx_raw:
                y_data_raw.append(None if ypx is None else float(cal.y_px_to_data(int(ypx))))

            y0_val = self.owner.calibrator._require_value(s.calibration.y0_val, "y0", cal=s.calibration)
            y1_val = self.owner.calibrator._require_value(s.calibration.y1_val, "y1", cal=s.calibration)
            fallback_y = 0.5 * (y0_val + y1_val)

            y_filled = enforce_line_grid(x_grid, y_data_raw, fallback_y=fallback_y)

            px_points: List[Tuple[int, int]] = []
            for i, xpx in enumerate(xpx_grid):
                if ypx_raw[i] is None:
                    ypx = int(round(cal.y_data_to_px(y_filled[i])))
                else:
                    ypx = int(ypx_raw[i])
                px_points.append((int(xpx), int(ypx)))

            s.px_points = px_points
            s.points = [(float(x_grid[i]), float(y_filled[i])) for i in range(len(x_grid))]
            s.point_enabled = [True] * len(s.points)
            self.owner.series_actor._update_tree_row(s)
            return

        # ---------------- Column / Area (vertical) ----------------
        if kind in ("column", "area"):
            # Determine sampling xpx_grid
            if stride == "categorical":
                labels = self.owner.calibrator._parse_categories(s.calibration.categories)
                if labels:
                    centers = self.owner.calibrator._category_centers_px(len(labels), axis="x", axis_px=s.calibration.x_axis_px)
                    offset = 0.0
                    if kind == "column" and s.seed_px is not None:
                        offset = self.owner.calibrator._category_offset_px(
                            len(labels),
                            axis="x",
                            seed_px=s.seed_px,
                            axis_px=s.calibration.x_axis_px,
                        )
                    xpx_grid = [int(round(p + offset)) for p in centers]
                    x_grid = [float(i + 1) for i in range(len(labels))]
                else:
                    centers = detect_runs_centers_along_x(mask, min_run_px=2)
                    xpx_grid = [int(x0 + c) for c in centers]
                    x_grid = [float(cal.x_px_to_data(xpx)) for xpx in xpx_grid]
            else:
                x0_val = self.owner.calibrator._require_value(s.calibration.x0_val, "x0", cal=s.calibration)
                roi_xmin_px, _, roi_xmax_px, _ = roi
                xmin = float(cal.x_px_to_data(roi_xmin_px))
                xmax = float(cal.x_px_to_data(roi_xmax_px))
                if xmin > xmax:
                    xmin, xmax = xmax, xmin
                step = float(s.calibration.x_step)
                if s.calibration.x_scale == AxisScale.DATE.value:
                    x_grid = self._build_date_grid_aligned(
                        xmin, xmax, step, s.calibration.x_step_unit, anchor=x0_val
                    )
                else:
                    x_grid = build_x_grid_aligned(xmin, xmax, step, anchor=x0_val)
                xpx_grid = [int(round(cal.x_data_to_px(x))) for x in x_grid]

            # Baseline pixel for sign (prefer y=0 when in range; else use y0 axis pixel)
            baseline_y_px = None
            try:
                y0px = int(round(cal.y_data_to_px(0.0)))
                if y0 <= y0px <= y1:
                    baseline_y_px = y0px
            except Exception:
                baseline_y_px = None
            if baseline_y_px is None:
                baseline_y_px = s.calibration.y_axis_px[0]  # y0 tick (defaults to bottom ROI)

            sx, sy = s.seed_px
            prefer_top = (sy < int(baseline_y_px))

            ypx_raw: List[Optional[int]] = []
            px_points: List[Tuple[int, int]] = []
            y_hint_local: Optional[int] = int(sy - y0) if (y0 <= sy < y1) else None

            for xpx in xpx_grid:
                if xpx < x0 or xpx >= x1:
                    ypx_raw.append(None)
                    continue
                xc = int(xpx - x0)
                y_local = _scan_col_edge(xc, prefer_top=prefer_top, y_hint_local=y_hint_local, band=12)
                if y_local is None:
                    ypx_raw.append(None)
                    continue
                ypx = int(y0 + y_local)
                ypx_raw.append(ypx)
                px_points.append((int(xpx), int(ypx)))
                y_hint_local = int(y_local)

            # Convert to data and optionally fill gaps (continuous stride only)
            y_data_raw: List[Optional[float]] = []
            for ypx in ypx_raw:
                y_data_raw.append(None if ypx is None else float(cal.y_px_to_data(int(ypx))))

            if stride == "continuous":
                y0_val = self.owner.calibrator._require_value(s.calibration.y0_val, "y0", cal=s.calibration)
                y1_val = self.owner.calibrator._require_value(s.calibration.y1_val, "y1", cal=s.calibration)
                fallback_y = 0.5 * (y0_val + y1_val)
                y_filled = enforce_line_grid(list(x_grid), y_data_raw, fallback_y=fallback_y)

                # Ensure px_points aligned to x_grid
                px_aligned: List[Tuple[int, int]] = []
                for i, xpx in enumerate(xpx_grid):
                    if ypx_raw[i] is None:
                        ypx = int(round(cal.y_data_to_px(y_filled[i])))
                    else:
                        ypx = int(ypx_raw[i])
                    px_aligned.append((int(xpx), int(ypx)))
                s.px_points = px_aligned
                s.points = [(float(x_grid[i]), float(y_filled[i])) for i in range(len(x_grid))]
                s.point_enabled = [True] * len(s.points)
            else:
                # categorical: keep only detected points; no interpolation
                pts: List[Tuple[float, float]] = []
                en: List[bool] = []
                pxs: List[Tuple[int, int]] = []
                for i, xval in enumerate(x_grid):
                    if y_data_raw[i] is None:
                        continue
                    pts.append((float(xval), float(y_data_raw[i])))
                    en.append(True)
                    pxs.append((int(xpx_grid[i]), int(ypx_raw[i])))
                s.points = pts
                s.point_enabled = en
                s.px_points = pxs

            self.owner.series_actor._update_tree_row(s)
            return

        # ---------------- Bar (horizontal) ----------------
        if kind == "bar":
            # For bars, categorical is the dominant case (categories along Y).
            labels = self.owner.calibrator._parse_categories(s.calibration.categories)
            if labels:
                centers = self.owner.calibrator._category_centers_px(len(labels), axis="y", axis_px=s.calibration.y_axis_px)
                offset = 0.0
                if s.seed_px is not None:
                    offset = self.owner.calibrator._category_offset_px(
                        len(labels),
                        axis="y",
                        seed_px=s.seed_px,
                        axis_px=s.calibration.y_axis_px,
                    )
                ypx_grid = [int(round(p + offset)) for p in centers]
                cat_grid = [float(i + 1) for i in range(len(labels))]
            else:
                centers = detect_runs_centers_along_y(mask, min_run_px=2)
                # Sort centers in display order (top->bottom). Categories typically go from top to bottom.
                centers = sorted(centers)

                ypx_grid = [int(y0 + c) for c in centers]
                # category coordinate (x axis in exported table): use y-axis data coordinate of the category center
                cat_grid = [float(cal.y_px_to_data(ypx)) for ypx in ypx_grid]

            baseline_x_px = None
            try:
                x0px = int(round(cal.x_data_to_px(0.0)))
                if x0 <= x0px <= x1:
                    baseline_x_px = x0px
            except Exception:
                baseline_x_px = None
            if baseline_x_px is None:
                baseline_x_px = s.calibration.x_axis_px[0]  # x0 tick (defaults to left ROI)

            sx, sy = s.seed_px
            prefer_right = (sx > int(baseline_x_px))

            xpx_raw: List[Optional[int]] = []
            px_points: List[Tuple[int, int]] = []
            x_hint_local: Optional[int] = int(sx - x0) if (x0 <= sx < x1) else None

            for ypx in ypx_grid:
                if ypx < y0 or ypx >= y1:
                    xpx_raw.append(None)
                    continue
                yc = int(ypx - y0)
                x_local = _scan_row_edge(yc, prefer_left=(not prefer_right), x_hint_local=x_hint_local, band=12)
                if x_local is None:
                    xpx_raw.append(None)
                    continue
                xpx = int(x0 + x_local)
                xpx_raw.append(xpx)
                px_points.append((int(xpx), int(ypx)))
                x_hint_local = int(x_local)

            # Convert to values: x is bar value, category is cat_grid
            pts: List[Tuple[float, float]] = []
            en: List[bool] = []
            pxs: List[Tuple[int, int]] = []

            for i, cat in enumerate(cat_grid):
                if xpx_raw[i] is None:
                    continue
                val = float(cal.x_px_to_data(int(xpx_raw[i])))
                pts.append((float(cat), float(val)))
                en.append(True)
                pxs.append((int(xpx_raw[i]), int(ypx_grid[i])))

            s.points = pts
            s.point_enabled = en
            s.px_points = pxs
            self.owner.series_actor._update_tree_row(s)
            return

        raise ValueError(f"Unsupported chart kind: {kind}")



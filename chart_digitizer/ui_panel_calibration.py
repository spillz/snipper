from __future__ import annotations

from datetime import datetime
import tkinter as tk
from tkinter import ttk
from typing import Callable, List, Optional, Tuple

import numpy as np

from .calibration import AxisScale, AxisCalibration, Calibration
from .data_model import CalibrationConfig, Series
from .date_utils import parse_date_safe, add_months


class CalibrationPanel:
    def __init__(
        self,
        owner,
        parent: tk.Widget,
        *,
        on_sample_mode_change: Callable[[], None],
        on_x_scale_change: Callable[[], None],
        on_y_scale_change: Callable[[], None],
        on_x_step_unit_change: Callable[[], None],
        on_y_step_unit_change: Callable[[], None],
        on_apply_calibration_active: Callable[[], None],
        on_apply_calibration_choose: Callable[[], None],
        on_apply_calibration_all: Callable[[], None],
    ) -> None:
        self.owner = owner
        frame = ttk.LabelFrame(parent, text="Calibration", padding=8)
        self.frame = frame
        owner._calibration_frame = frame
        frame.pack(side="top", fill="x")

        owner.axis_px_values = tk.StringVar(value="")
        ttk.Label(frame, textvariable=owner.axis_px_values).pack(fill="x", pady=(2, 0))

        sample_row = ttk.Frame(frame)
        sample_row.pack(fill="x", pady=(6, 0))
        ttk.Label(sample_row, text="Sample:").pack(side="left")
        owner.cmb_sample_mode = ttk.Combobox(
            sample_row,
            textvariable=owner.var_sample_mode,
            state="readonly",
            width=10,
            values=("Free", "Fixed X", "Fixed Y"),
        )
        owner.cmb_sample_mode.pack(side="left", padx=(6, 0))
        owner.cmb_sample_mode.bind("<<ComboboxSelected>>", lambda _e: on_sample_mode_change())

        grid = ttk.Frame(frame)
        grid.pack(fill="x")
        for col in range(6):
            grid.columnconfigure(col, weight=1)

        x_axis_hdr = ttk.Frame(grid)
        x_axis_hdr.grid(row=0, column=0, sticky="ew", pady=(6, 0), columnspan=6)
        ttk.Label(x_axis_hdr, text="X axis").pack(side="left")
        ttk.Separator(x_axis_hdr, orient="horizontal").pack(side="left", fill="x", expand=True, padx=(8, 0))
        ttk.Label(grid, text="scale").grid(row=1, column=0, sticky="w")
        owner.cmb_x_scale = ttk.Combobox(
            grid,
            textvariable=owner.x_scale,
            state="readonly",
            width=10,
            values=[AxisScale.LINEAR.value, AxisScale.LOG10.value, AxisScale.DATE.value, AxisScale.CATEGORICAL.value],
        )
        owner.cmb_x_scale.grid(row=1, column=1, sticky="w", padx=(6, 0))
        owner.cmb_x_scale.bind("<<ComboboxSelected>>", lambda _e: on_x_scale_change())

        owner.lbl_date_fmt = ttk.Label(grid, text="date fmt")
        self.date_fmt_options = [
            "%Y",
            "%Y-%m",
            "%Y-%m-%d",
            "%Y/%m",
            "%Y/%m/%d",
            "%m/%d/%Y",
            "%d/%m/%Y",
            "%b %Y",
            "%d %b %Y",
            "%Y-%m-%d %H",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d %H:%M:%S",
        ]
        owner.cmb_date_fmt = ttk.Combobox(
            grid,
            textvariable=owner.date_fmt,
            state="normal",
            width=14,
            values=self.date_fmt_options,
        )

        owner.var_x0_label = tk.StringVar(value="x0")
        owner.var_x1_label = tk.StringVar(value="x1")
        owner.lbl_x0 = ttk.Label(grid, textvariable=owner.var_x0_label)
        owner.ent_x0 = ttk.Entry(grid, textvariable=owner.var_x0_val, width=10)
        owner.lbl_x1 = ttk.Label(grid, textvariable=owner.var_x1_label)
        owner.ent_x1 = ttk.Entry(grid, textvariable=owner.var_x1_val, width=10)
        owner.var_x_step_label = tk.StringVar(value="step")
        owner.lbl_x_step = ttk.Label(grid, textvariable=owner.var_x_step_label)
        owner.ent_x_step = ttk.Entry(grid, textvariable=owner.var_x_step, width=8)
        owner.lbl_x_step_units = ttk.Label(grid, text="unit")
        owner.cmb_x_step_units = ttk.Combobox(
            grid,
            textvariable=owner.var_x_step_unit,
            state="readonly",
            width=9,
            values=("seconds", "minutes", "hours", "days", "weeks", "months", "quarters", "years"),
        )
        owner.cmb_x_step_units.bind("<<ComboboxSelected>>", lambda _e: on_x_step_unit_change())

        owner.lbl_x_step.grid(row=1, column=2, sticky="w", padx=(8, 0))
        owner.ent_x_step.grid(row=1, column=3, sticky="w")
        owner.lbl_x_step_units.grid(row=1, column=4, sticky="w", padx=(8, 0))
        owner.cmb_x_step_units.grid(row=1, column=5, sticky="w")
        owner.lbl_x0.grid(row=3, column=0, sticky="w", pady=(6, 0))
        owner.ent_x0.grid(row=3, column=1, sticky="w", pady=(6, 0))
        owner.lbl_x1.grid(row=3, column=2, sticky="w", padx=(8, 0), pady=(6, 0))
        owner.ent_x1.grid(row=3, column=3, sticky="w", pady=(6, 0))

        y_axis_hdr = ttk.Frame(grid)
        y_axis_hdr.grid(row=5, column=0, sticky="ew", pady=(10, 0), columnspan=6)
        ttk.Label(y_axis_hdr, text="Y axis").pack(side="left")
        ttk.Separator(y_axis_hdr, orient="horizontal").pack(side="left", fill="x", expand=True, padx=(8, 0))
        ttk.Label(grid, text="scale").grid(row=6, column=0, sticky="w", pady=(6, 0))
        owner.cmb_y_scale = ttk.Combobox(
            grid,
            textvariable=owner.y_scale,
            state="readonly",
            width=10,
            values=[AxisScale.LINEAR.value, AxisScale.LOG10.value, AxisScale.DATE.value, AxisScale.CATEGORICAL.value],
        )
        owner.cmb_y_scale.grid(row=6, column=1, sticky="w", padx=(6, 0), pady=(6, 0))
        owner.cmb_y_scale.bind("<<ComboboxSelected>>", lambda _e: on_y_scale_change())

        owner.lbl_y0 = ttk.Label(grid, text="y0")
        owner.ent_y0 = ttk.Entry(grid, textvariable=owner.var_y0_val, width=10)
        owner.lbl_y1 = ttk.Label(grid, text="y1")
        owner.ent_y1 = ttk.Entry(grid, textvariable=owner.var_y1_val, width=10)
        owner.var_y_step_label = tk.StringVar(value="step")
        owner.lbl_y_step = ttk.Label(grid, textvariable=owner.var_y_step_label)
        owner.ent_y_step = ttk.Entry(grid, textvariable=owner.var_y_step, width=8)
        owner.lbl_y_step_units = ttk.Label(grid, text="unit")
        owner.cmb_y_step_units = ttk.Combobox(
            grid,
            textvariable=owner.var_y_step_unit,
            state="readonly",
            width=9,
            values=("seconds", "minutes", "hours", "days", "weeks", "months", "quarters", "years"),
        )
        owner.cmb_y_step_units.bind("<<ComboboxSelected>>", lambda _e: on_y_step_unit_change())

        owner.lbl_y_step.grid(row=6, column=2, sticky="w", padx=(8, 0))
        owner.ent_y_step.grid(row=6, column=3, sticky="w")
        owner.lbl_y_step_units.grid(row=6, column=4, sticky="w", padx=(8, 0))
        owner.cmb_y_step_units.grid(row=6, column=5, sticky="w")
        owner.lbl_y0.grid(row=8, column=0, sticky="w", pady=(6, 0))
        owner.ent_y0.grid(row=8, column=1, sticky="w", pady=(6, 0))
        owner.lbl_y1.grid(row=8, column=2, sticky="w", padx=(8, 0), pady=(6, 0))
        owner.ent_y1.grid(row=8, column=3, sticky="w", pady=(6, 0))

        owner.lbl_categories = ttk.Label(grid, text="categories")
        owner.ent_categories = ttk.Entry(grid, textvariable=owner.var_categories, width=28)
        owner.lbl_categories_count = ttk.Label(grid, text="")

        cal_btns = ttk.Frame(frame)
        cal_btns.pack(fill="x", pady=(8, 0))
        ttk.Button(cal_btns, text="Update", command=on_apply_calibration_active).pack(side="left")
        ttk.Button(cal_btns, text="Apply to...", command=on_apply_calibration_choose).pack(side="left", padx=(8, 0))
        ttk.Button(cal_btns, text="Apply to all", command=on_apply_calibration_all).pack(side="left", padx=(8, 0))

    def ensure_categories_default(self) -> None:
        o = self.owner
        if not o.var_categories.get().strip():
            o.var_categories.set("x1;x2;x3")

    def update_category_count(self) -> None:
        o = self.owner
        if not getattr(o, "_cat_row_visible", False):
            o.lbl_categories_count.configure(text="")
            return
        count = len(o.calibrator._parse_categories())
        o.lbl_categories_count.configure(text=f"{count} cats")

    def refresh_scale_ui(self) -> None:
        o = self.owner
        cal = o.calibrator
        date_axis = cal._date_fmt_axis()
        is_date = cal._is_x_date_scale() if date_axis == "x" else cal._is_y_date_scale()
        is_bar = o.series_mode.get() == "bar"
        cat_axis = "y" if is_bar else "x"
        cat_scale = o.y_scale.get() if cat_axis == "y" else o.x_scale.get()
        is_categorical = cat_scale == AxisScale.CATEGORICAL.value

        # Limit categorical/date options to the active axis.
        x_values = [AxisScale.LINEAR.value, AxisScale.LOG10.value]
        y_values = [AxisScale.LINEAR.value, AxisScale.LOG10.value]
        if o.series_mode.get() == "scatter":
            sample_mode = cal._sample_label_to_mode(o.var_sample_mode.get())
            if sample_mode == "free":
                cat_axis = "x"
                is_categorical = False
            elif sample_mode == "fixed_x":
                x_values.append(AxisScale.DATE.value)
                x_values.append(AxisScale.CATEGORICAL.value)
                cat_axis = "x"
                cat_scale = o.x_scale.get()
                is_categorical = cat_scale == AxisScale.CATEGORICAL.value
            elif sample_mode == "fixed_y":
                y_values.append(AxisScale.DATE.value)
                y_values.append(AxisScale.CATEGORICAL.value)
                cat_axis = "y"
                cat_scale = o.y_scale.get()
                is_categorical = cat_scale == AxisScale.CATEGORICAL.value
        elif is_bar:
            y_values.append(AxisScale.DATE.value)
            y_values.append(AxisScale.CATEGORICAL.value)
        else:
            x_values.append(AxisScale.DATE.value)
            x_values.append(AxisScale.CATEGORICAL.value)
        o.cmb_x_scale.configure(values=x_values)
        o.cmb_y_scale.configure(values=y_values)
        if o.x_scale.get() not in x_values:
            o.x_scale.set(AxisScale.LINEAR.value)
        if o.y_scale.get() not in y_values:
            o.y_scale.set(AxisScale.LINEAR.value)

        if is_date and not is_categorical:
            if date_axis == "x":
                o.lbl_date_fmt.configure(text="date fmt")
                o.var_x0_label.set("x0 (date)")
                o.var_x1_label.set("x1 (date)")
                o.var_x_step_label.set("step")
            else:
                o.lbl_date_fmt.configure(text="date fmt")
                o.var_y_step_label.set("step")
        else:
            o.var_x0_label.set("x0")
            o.var_x1_label.set("x1")
            o.var_x_step_label.set("step")
            o.var_y_step_label.set("step")

        o.lbl_date_fmt.grid_remove()
        o.cmb_date_fmt.grid_remove()
        if is_date and not is_categorical:
            if date_axis == "x":
                o.lbl_date_fmt.grid(row=2, column=0, sticky="w", pady=(6, 0))
                o.cmb_date_fmt.grid(row=2, column=1, sticky="w", pady=(6, 0))
            else:
                o.lbl_date_fmt.grid(row=7, column=0, sticky="w", pady=(6, 0))
                o.cmb_date_fmt.grid(row=7, column=1, sticky="w", pady=(6, 0))

        if is_categorical:
            if cat_axis == "y":
                o.lbl_categories.configure(text="Categories")
                o.lbl_y0.grid_remove()
                o.ent_y0.grid_remove()
                o.lbl_y1.grid_remove()
                o.ent_y1.grid_remove()
                o.lbl_y_step.grid_remove()
                o.ent_y_step.grid_remove()
                o.lbl_y_step_units.grid_remove()
                o.cmb_y_step_units.grid_remove()
                o.lbl_x0.grid()
                o.ent_x0.grid()
                o.lbl_x1.grid()
                o.ent_x1.grid()
            else:
                o.lbl_categories.configure(text="Categories")
                o.lbl_x0.grid_remove()
                o.ent_x0.grid_remove()
                o.lbl_x1.grid_remove()
                o.ent_x1.grid_remove()
                o.lbl_x_step.grid_remove()
                o.ent_x_step.grid_remove()
                o.lbl_x_step_units.grid_remove()
                o.cmb_x_step_units.grid_remove()
                o.lbl_y0.grid()
                o.ent_y0.grid()
                o.lbl_y1.grid()
                o.ent_y1.grid()
            if cat_axis == "y":
                o.lbl_categories.grid(row=9, column=0, sticky="w", pady=(6, 0))
                o.ent_categories.grid(row=9, column=1, columnspan=4, sticky="ew", pady=(6, 0))
                o.lbl_categories_count.grid(row=9, column=5, sticky="e", padx=(8, 0), pady=(6, 0))
            else:
                o.lbl_categories.grid(row=4, column=0, sticky="w", pady=(6, 0))
                o.ent_categories.grid(row=4, column=1, columnspan=4, sticky="ew", pady=(6, 0))
                o.lbl_categories_count.grid(row=4, column=5, sticky="e", padx=(8, 0), pady=(6, 0))
            o._cat_row_visible = True
            self.ensure_categories_default()
            self.update_category_count()
        else:
            o.lbl_categories.grid_remove()
            o.ent_categories.grid_remove()
            o.lbl_categories_count.grid_remove()
            o._cat_row_visible = False
            o.lbl_x0.grid()
            o.ent_x0.grid()
            o.lbl_x1.grid()
            o.ent_x1.grid()
            o.lbl_x_step.grid()
            o.ent_x_step.grid()
            o.lbl_x_step_units.grid()
            o.cmb_x_step_units.grid()
            o.lbl_y0.grid()
            o.ent_y0.grid()
            o.lbl_y1.grid()
            o.ent_y1.grid()
            o.lbl_y_step.grid()
            o.ent_y_step.grid()
            o.lbl_y_step_units.grid()
            o.cmb_y_step_units.grid()

        sample_mode = cal._sample_label_to_mode(o.var_sample_mode.get())
        scatter_x = (o.series_mode.get() == "scatter" and sample_mode == "fixed_x")
        show_x_step = ((not is_bar) and (o.series_mode.get() != "scatter") and not (is_categorical and cat_axis == "x")) or scatter_x
        if show_x_step:
            o.lbl_x_step.grid()
            o.ent_x_step.grid()
            o.lbl_x_step_units.grid()
            o.cmb_x_step_units.grid()
        else:
            o.lbl_x_step.grid_remove()
            o.ent_x_step.grid_remove()
            o.lbl_x_step_units.grid_remove()
            o.cmb_x_step_units.grid_remove()

        show_x_units = (is_date and show_x_step and date_axis == "x")
        if show_x_units:
            o.lbl_x_step_units.grid()
            o.cmb_x_step_units.grid()
        else:
            o.lbl_x_step_units.grid_remove()
            o.cmb_x_step_units.grid_remove()

        scatter_y = (o.series_mode.get() == "scatter" and sample_mode == "fixed_y")
        show_y_step = (is_bar and not (is_categorical and cat_axis == "y")) or scatter_y
        if show_y_step:
            o.lbl_y_step.grid()
            o.ent_y_step.grid()
            o.lbl_y_step_units.grid()
            o.cmb_y_step_units.grid()
        else:
            o.lbl_y_step.grid_remove()
            o.ent_y_step.grid_remove()
            o.lbl_y_step_units.grid_remove()
            o.cmb_y_step_units.grid_remove()

        show_y_units = (is_date and show_y_step and date_axis == "y")
        if show_y_units:
            o.lbl_y_step_units.grid()
            o.cmb_y_step_units.grid()
        else:
            o.lbl_y_step_units.grid_remove()
            o.cmb_y_step_units.grid_remove()

        o.cmb_x_scale.configure(state="readonly")


class Calibrator:
    def __init__(self, owner) -> None:
        object.__setattr__(self, "owner", owner)

    def __getattr__(self, name):
        return getattr(self.owner, name)

    def __setattr__(self, name, value) -> None:
        if name == "owner":
            object.__setattr__(self, name, value)
            return
        setattr(self.owner, name, value)

    def _is_x_date_scale(self) -> bool:
        return self.x_scale.get() == AxisScale.DATE.value


    def _is_y_date_scale(self) -> bool:
        return self.y_scale.get() == AxisScale.DATE.value


    def _date_fmt_axis(self) -> str:
        if self.series_mode.get() == "scatter":
            if self.y_scale.get() == AxisScale.DATE.value and self.x_scale.get() != AxisScale.DATE.value:
                return "y"
            return "x"
        return "y" if self.series_mode.get() == "bar" else "x"


    def _stride_mode(self) -> str:
        if self.series_mode.get() == "bar":
            return "categorical" if self.y_scale.get() == AxisScale.CATEGORICAL.value else "continuous"
        return "categorical" if self.x_scale.get() == AxisScale.CATEGORICAL.value else "continuous"


    def _stride_mode_for_calibration(self, cal: CalibrationConfig, chart_kind: str) -> str:
        if chart_kind == "bar":
            return "categorical" if cal.y_scale == AxisScale.CATEGORICAL.value else "continuous"
        return "categorical" if cal.x_scale == AxisScale.CATEGORICAL.value else "continuous"


    def _on_x_scale_change(self) -> None:
        self._refresh_scale_ui()
        if self._is_x_date_scale() and self._date_fmt_axis() == "x":
            self._ensure_date_values("x")
            self._set_default_date_unit()
        self.canvas_actor._update_tip()


    def _on_y_scale_change(self) -> None:
        self._refresh_scale_ui()
        if self._is_y_date_scale() and self._date_fmt_axis() == "y":
            self._ensure_date_values("y")
            self._set_default_date_unit()
        self.canvas_actor._update_tip()


    def _on_date_fmt_change(self) -> None:
        if self._date_fmt_axis() == "x" and self._is_x_date_scale():
            self._ensure_date_values("x")
            self._set_default_date_unit()
        if self._date_fmt_axis() == "y" and self._is_y_date_scale():
            self._ensure_date_values("y")
            self._set_default_date_unit()


    def _on_x_step_unit_changed(self) -> None:
        self._date_unit_user_set = True


    def _on_y_step_unit_changed(self) -> None:
        self._date_unit_user_set = True


    def _refresh_scale_ui(self) -> None:
        self.calibration_panel.refresh_scale_ui()


    def _default_date_unit_for_fmt(self, fmt: str) -> str:
        fmt = fmt or ""
        if "%S" in fmt:
            return "seconds"
        if "%M" in fmt:
            return "minutes"
        if "%H" in fmt:
            return "hours"
        if "%d" in fmt or "%j" in fmt:
            return "days"
        if "%U" in fmt or "%W" in fmt:
            return "weeks"
        if "%m" in fmt or "%b" in fmt or "%B" in fmt:
            return "months"
        if "Q" in fmt or "q" in fmt:
            return "quarters"
        if "%Y" in fmt or "%y" in fmt:
            return "years"
        return "days"


    def _set_default_date_unit(self) -> None:
        default = self._default_date_unit_for_fmt(self.date_fmt.get())
        if self._date_fmt_axis() == "y":
            if (not self._date_unit_user_set) or (self.var_y_step_unit.get() == (self._date_unit_auto or "")):
                self.var_y_step_unit.set(default)
                self._date_unit_auto = default
        else:
            if (not self._date_unit_user_set) or (self.var_x_step_unit.get() == (self._date_unit_auto or "")):
                self.var_x_step_unit.set(default)
                self._date_unit_auto = default


    def _ensure_date_values(self, axis: str = "x") -> None:
        fmt = self.date_fmt.get().strip() or "%Y"

        def _valid_date(s: str) -> bool:
            try:
                datetime.strptime(s.strip(), fmt)
                return True
            except Exception:
                return False

        base = datetime(2000, 1, 1)
        if axis == "y":
            x0 = self.var_y0_val.get().strip()
            x1 = self.var_y1_val.get().strip()
        else:
            x0 = self.var_x0_val.get().strip()
            x1 = self.var_x1_val.get().strip()

        if not x0 or not _valid_date(x0):
            if axis == "y":
                self.var_y0_val.set(base.strftime(fmt))
            else:
                self.var_x0_val.set(base.strftime(fmt))
        if not x1 or not _valid_date(x1):
            x1_dt = add_months(base, 12)
            if axis == "y":
                self.var_y1_val.set(x1_dt.strftime(fmt))
            else:
                self.var_x1_val.set(x1_dt.strftime(fmt))


    def _sample_label_to_mode(self, label: str) -> str:
        label = (label or "").strip().lower()
        if label == "fixed x":
            return "fixed_x"
        if label == "fixed y":
            return "fixed_y"
        return "free"


    def _sample_mode_to_label(self, mode: str) -> str:
        mode = (mode or "").strip().lower()
        if mode == "fixed_x":
            return "Fixed X"
        if mode == "fixed_y":
            return "Fixed Y"
        return "Free"


    def _allowed_sample_labels(self, series_mode: str) -> List[str]:
        if series_mode == "scatter":
            return ["Free", "Fixed X", "Fixed Y"]
        if series_mode == "bar":
            return ["Fixed Y"]
        return ["Fixed X"]


    def _on_sample_mode_change(self) -> None:
        self._update_sample_mode_ui()
        self._refresh_scale_ui()


    def _update_sample_mode_ui(self) -> None:
        allowed = self._allowed_sample_labels(self.series_mode.get())
        if self.var_sample_mode.get() not in allowed:
            self.var_sample_mode.set(allowed[0])
        self.cmb_sample_mode.configure(values=allowed)

        mode = self._sample_label_to_mode(self.var_sample_mode.get())
        x_state = "normal" if mode == "fixed_x" else "disabled"
        y_state = "normal" if mode == "fixed_y" else "disabled"
        self.ent_x_step.configure(state=x_state)
        self.cmb_x_step_units.configure(state=("readonly" if x_state == "normal" else "disabled"))
        self.ent_y_step.configure(state=y_state)
        self.cmb_y_step_units.configure(state=("readonly" if y_state == "normal" else "disabled"))


    def _normalize_categories(self, raw: str) -> str:
        parts = [p.strip() for p in (raw or "").split(";")]
        return ";".join([p for p in parts if p])


    def _parse_categories(self, raw: Optional[str] = None) -> List[str]:
        if raw is None:
            raw = self.var_categories.get()
        parts = [p.strip() for p in (raw or "").split(";")]
        return [p for p in parts if p]


    def _category_centers_px(self, count: int, *, axis: str, axis_px: Optional[Tuple[int, int]] = None) -> List[float]:
        if count <= 1:
            if axis_px is None:
                if axis == "y":
                    p0, p1 = self.canvas_actor._y_axis_px()
                else:
                    p0, p1 = self.canvas_actor._x_axis_px()
            else:
                p0, p1 = axis_px
            return [float(p0)]

        if axis_px is None:
            if axis == "y":
                p0, p1 = self.canvas_actor._y_axis_px()
            else:
                p0, p1 = self.canvas_actor._x_axis_px()
        else:
            p0, p1 = axis_px
        step = (float(p1) - float(p0)) / float(count - 1)
        return [float(p0) + (i * step) for i in range(count)]


    def _category_offset_px(
        self,
        count: int,
        *,
        axis: str,
        seed_px: Tuple[int, int],
        axis_px: Optional[Tuple[int, int]] = None,
    ) -> float:
        centers = self._category_centers_px(count, axis=axis, axis_px=axis_px)
        if not centers:
            return 0.0
        if axis == "y":
            pos = float(seed_px[1])
        else:
            pos = float(seed_px[0])
        idx = int(np.argmin([abs(pos - c) for c in centers]))
        return float(pos - centers[idx])


    def _calibration_key_from_ui(self) -> tuple:
        data_region = self.canvas_actor._data_region_px()
        x_axis_px = self.canvas_actor._x_axis_px()
        y_axis_px = self.canvas_actor._y_axis_px()
        x0_val = (self.var_x0_val.get() or "").strip()
        x1_val = (self.var_x1_val.get() or "").strip()
        y0_val = (self.var_y0_val.get() or "").strip()
        y1_val = (self.var_y1_val.get() or "").strip()
        categories = self._normalize_categories(self.var_categories.get())
        x_scale = (self.x_scale.get() or "").strip()
        y_scale = (self.y_scale.get() or "").strip()
        date_fmt = (self.date_fmt.get() or "").strip()
        x_step = float(self.var_x_step.get())
        x_step_unit = (self.var_x_step_unit.get() or "").strip().lower()
        y_step = float(self.var_y_step.get())
        y_step_unit = (self.var_y_step_unit.get() or "").strip().lower()
        sample_mode = self._sample_label_to_mode(self.var_sample_mode.get())
        return (
            data_region,
            x_axis_px,
            y_axis_px,
            x0_val,
            x1_val,
            y0_val,
            y1_val,
            categories,
            x_scale,
            y_scale,
            date_fmt,
            x_step,
            x_step_unit,
            y_step,
            y_step_unit,
            sample_mode,
        )


    def _calibration_key_from_series(self, s: Series) -> tuple:
        cal = s.calibration
        return (
            cal.data_region_px,
            cal.x_axis_px,
            cal.y_axis_px,
            cal.x0_val,
            cal.x1_val,
            cal.y0_val,
            cal.y1_val,
            cal.categories,
            cal.x_scale,
            cal.y_scale,
            cal.date_fmt,
            cal.x_step,
            cal.x_step_unit,
            cal.y_step,
            cal.y_step_unit,
            cal.sample_mode,
        )


    def _make_series_calibration_from_ui(self) -> CalibrationConfig:
        key = self._calibration_key_from_ui()
        name = self._calibration_names.get(key)
        if name is None:
            name = f"calib{self._next_calibration_id}"
            self._next_calibration_id += 1
            self._calibration_names[key] = name
        (
            data_region,
            x_axis_px,
            y_axis_px,
            x0_val,
            x1_val,
            y0_val,
            y1_val,
            categories,
            x_scale,
            y_scale,
            date_fmt,
            x_step,
            x_step_unit,
            y_step,
            y_step_unit,
            sample_mode,
        ) = key
        return CalibrationConfig(
            name=name,
            data_region_px=data_region,
            x_axis_px=x_axis_px,
            y_axis_px=y_axis_px,
            x0_val=x0_val,
            x1_val=x1_val,
            y0_val=y0_val,
            y1_val=y1_val,
            categories=categories,
            x_scale=x_scale,
            y_scale=y_scale,
            date_fmt=date_fmt,
            x_step=float(x_step),
            x_step_unit=x_step_unit,
            y_step=float(y_step),
            y_step_unit=y_step_unit,
            sample_mode=sample_mode,
        )


    def _apply_series_calibration_to_ui(self, cal: CalibrationConfig) -> None:
        self.state.xmin_px, self.state.ymin_px, self.state.xmax_px, self.state.ymax_px = cal.data_region_px
        self.state.x0_px, self.state.x1_px = cal.x_axis_px
        self.state.y0_px, self.state.y1_px = cal.y_axis_px
        self.var_x0_val.set(cal.x0_val)
        self.var_x1_val.set(cal.x1_val)
        self.var_y0_val.set(cal.y0_val)
        self.var_y1_val.set(cal.y1_val)
        self.var_categories.set(cal.categories)
        self.x_scale.set(cal.x_scale)
        self.y_scale.set(cal.y_scale)
        self.date_fmt.set(cal.date_fmt)
        self.var_x_step.set(float(cal.x_step))
        self.var_x_step_unit.set(cal.x_step_unit)
        self.var_y_step.set(float(cal.y_step))
        self.var_y_step_unit.set(cal.y_step_unit)
        self.var_sample_mode.set(self._sample_mode_to_label(cal.sample_mode))
        self._date_unit_user_set = True
        self._refresh_scale_ui()
        self._update_sample_mode_ui()
        self.canvas_actor._update_tip()
        self.canvas_actor._redraw_overlay()


    def _pixel_bounds_changes(self, old: CalibrationConfig, new: CalibrationConfig) -> List[str]:
        changes: List[str] = []
        if old.data_region_px != new.data_region_px:
            changes.append("Data region")
        if old.x_axis_px != new.x_axis_px:
            changes.append("X axis pixels")
        if old.y_axis_px != new.y_axis_px:
            changes.append("Y axis pixels")
        if old.x_step_unit != new.x_step_unit:
            changes.append("X step unit")
        if old.y_step_unit != new.y_step_unit:
            changes.append("Y step unit")
        if old.sample_mode != new.sample_mode:
            changes.append("Sample mode")
        return changes


    def _apply_calibration_to_series(self, targets: List[Series]) -> None:
        if not targets:
            self._show_info("Calibration", "No series to update.")
            return

        mode = self.series_mode.get()
        mismatched = [s for s in targets if getattr(s, "chart_kind", s.mode) != mode]
        if mismatched:
            self._show_info(
                "Calibration",
                "Series type changes are not supported. Create a new series for a different type."
            )
            targets = [s for s in targets if s not in mismatched]
        if not targets:
            self._show_info("Calibration", "No series updated (series type mismatch).")
            return

        new_cal = self._make_series_calibration_from_ui()
        to_reextract: List[Series] = []
        change_set: set[str] = set()
        updated = 0

        for s in targets:
            changes = self._pixel_bounds_changes(s.calibration, new_cal)
            updated += 1
            if changes:
                to_reextract.append(s)
                change_set.update(changes)
            s.calibration = new_cal
            s.stride_mode = self._stride_mode_for_calibration(new_cal, getattr(s, "chart_kind", s.mode))
            self.series_actor._update_tree_row(s)

        if to_reextract:
            self._show_info(
                "Calibration",
                f"Re-extracting {len(to_reextract)} series because pixel bounds changed: "
                + ", ".join(sorted(change_set))
            )
            for s in to_reextract:
                try:
                    self.extractor._extract_series(s)
                except Exception as e:
                    self._show_error("Series extraction failed", str(e))
        else:
            self._show_info("Calibration", f"Updated {updated} series. No re-extraction needed.")
        self.canvas_actor._redraw_overlay()


    def _apply_calibration_to_choose(self) -> None:
        if not self.series:
            self._show_info("Calibration", "No series to update yet.")
            return

        dlg = tk.Toplevel(self)
        dlg.title("Apply calibration to...")
        dlg.transient(self)
        dlg.grab_set()

        ttk.Label(dlg, text="Select series to update:").pack(anchor="w", padx=12, pady=(12, 6))
        frm = ttk.Frame(dlg)
        frm.pack(fill="both", expand=True, padx=12)

        vars_by_id: dict[int, tk.BooleanVar] = {}
        for s in self.series:
            v = tk.BooleanVar(value=(s.id == self._active_series_id))
            vars_by_id[s.id] = v
            ttk.Checkbutton(frm, text=f"{s.name} ({s.calibration.name})", variable=v).pack(anchor="w")

        btns = ttk.Frame(dlg)
        btns.pack(fill="x", padx=12, pady=12)

        def _on_ok() -> None:
            chosen = [self.series_actor._get_series(sid) for sid, v in vars_by_id.items() if v.get()]
            targets = [s for s in chosen if s is not None]
            dlg.destroy()
            if not targets:
                self._show_info("Calibration", "No series selected.")
                return
            self._apply_calibration_to_series(targets)

        def _on_cancel() -> None:
            dlg.destroy()

        ttk.Button(btns, text="OK", command=_on_ok).pack(side="right")
        ttk.Button(btns, text="Cancel", command=_on_cancel).pack(side="right", padx=(0, 8))

        dlg.bind("<Return>", lambda _e: _on_ok())
        dlg.bind("<Escape>", lambda _e: _on_cancel())
        dlg.wait_window(dlg)


    def _apply_calibration_to_active(self) -> None:
        if self._active_series_id is None:
            self._show_info("Calibration", "Select a series first.")
            return
        s = self.series_actor._get_series(self._active_series_id)
        if s is None:
            return
        self._apply_calibration_to_series([s])


    def _apply_calibration_to_all(self) -> None:
        if not self.series:
            self._show_info("Calibration", "No series to update yet.")
            return
        self._apply_calibration_to_series(list(self.series))


    def _update_axis_pixels_label(self) -> None:
        rx0, ry0, rx1, ry1 = self.canvas_actor._data_region_px()
        x0_px, x1_px = self.canvas_actor._x_axis_px()
        y0_px, y1_px = self.canvas_actor._y_axis_px()
        self.axis_px_values.set(
            f"image size: {self._iw}x{self._ih}\n"
            f"data region: x: {rx0}:{rx1}, y: {ry0}:{ry1}\n"
            f"x ticks x0:{x0_px} x1:{x1_px}, y ticks y0:{y0_px} y1:{y1_px}"
        )


    def _require_value(self, s: str, label: str, *, cal: Optional[CalibrationConfig] = None) -> float:
        s = (s or "").strip()
        if not s:
            raise ValueError(f"Missing {label} value. Enter it in the Calibration panel.")
        chart_cal = self._build_calibration(cal)
        if label.startswith("x"):
            return float(chart_cal.parse_x_value(s))
        if label.startswith("y"):
            return float(chart_cal.parse_y_value(s))
        return float(s)


    def _build_calibration(self, cal: Optional[CalibrationConfig] = None) -> Calibration:
        # Use data region bounds if axis pixels not set
        if cal is None:
            x0, y0, x1, y1 = self.canvas_actor._data_region_px()
            x0_px, x1_px = self.canvas_actor._x_axis_px()
            y0_px, y1_px = self.canvas_actor._y_axis_px()
            xs = AxisScale(self.x_scale.get())
            ys = AxisScale(self.y_scale.get())
            date_fmt = self.date_fmt.get().strip() or "%Y"
            x0v_str = (self.var_x0_val.get() or "").strip()
            x1v_str = (self.var_x1_val.get() or "").strip()
            y0v_str = (self.var_y0_val.get() or "").strip()
            y1v_str = (self.var_y1_val.get() or "").strip()
        else:
            x0, y0, x1, y1 = cal.data_region_px
            x0_px, x1_px = cal.x_axis_px
            y0_px, y1_px = cal.y_axis_px
            xs = AxisScale(cal.x_scale)
            ys = AxisScale(cal.y_scale)
            date_fmt = (cal.date_fmt or "").strip() or "%Y"
            x0v_str = (cal.x0_val or "").strip()
            x1v_str = (cal.x1_val or "").strip()
            y0v_str = (cal.y0_val or "").strip()
            y1v_str = (cal.y1_val or "").strip()

        # Parse X values
        if xs == AxisScale.DATE:
            x0v = parse_date_safe(x0v_str or "2000", date_fmt)
            x1v = parse_date_safe(x1v_str or "2001", date_fmt)
        else:
            x0v = float(x0v_str) if x0v_str else 0.0
            x1v = float(x1v_str) if x1v_str else 1.0

        if ys == AxisScale.DATE:
            y0v = parse_date_safe(y0v_str or "2000", date_fmt)
            y1v = parse_date_safe(y1v_str or "2001", date_fmt)
        else:
            y0v = float(y0v_str) if y0v_str else 0.0
            y1v = float(y1v_str) if y1v_str else 1.0

        xcal = AxisCalibration(p0=float(x0_px), p1=float(x1_px), v0=float(x0v), v1=float(x1v), scale=xs)
        ycal = AxisCalibration(p0=float(y0_px), p1=float(y1_px), v0=float(y0v), v1=float(y1v), scale=ys)
        return Calibration(x=xcal, y=ycal, x_date_format=date_fmt, y_date_format=date_fmt)


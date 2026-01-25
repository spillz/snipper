
from __future__ import annotations

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import tkinter.font as tkfont
from typing import Callable, Optional, Tuple, List

import bisect
import calendar
import platform
import numpy as np
from PIL import Image, ImageTk
from datetime import datetime, timezone

from .cv_utils import require_cv2, pil_to_bgr, color_distance_mask
from .model import ChartState, Series, SeriesCalibration
from .calibration import AxisScale, AxisCalibration, ChartCalibration
from .extract import (
    build_x_grid, build_x_grid_aligned,
    extract_line_series, extract_scatter_series, extract_scatter_series_fixed_stride, enforce_line_grid,
    detect_runs_centers_along_x, detect_runs_centers_along_y,
)
from .export_csv import wide_csv_string, long_csv_string, write_wide_csv, write_long_csv


class ChartDigitizerDialog(tk.Toplevel):
    def __init__(self, parent: tk.Tk, *, image: Image.Image, on_append_text: Callable[[str], None]):
        super().__init__(parent)
        self.title("Chart → CSV")
        self.geometry("1180x760")
        self.resizable(True, True)
        if platform.system().lower() != "windows":
            self.transient(parent)  # modeless: no grab_set
        self._on_append_text = on_append_text

        self._pil = image.convert("RGB")
        self._bgr = pil_to_bgr(self._pil)
        self._iw, self._ih = self._pil.size

        self.state = ChartState(xmin_px=0, ymin_px=0, xmax_px=self._iw, ymax_px=self._ih)
        self.series: List[Series] = []
        self._next_series_id = 1
        self._calibration_names: dict[tuple, str] = {}
        self._next_calibration_id = 1

        # Tool mode
        self.tool_mode = tk.StringVar(value="roi")  # edit|roi|xaxis|yaxis|addseries
        self.series_mode = tk.StringVar(value="line")  # line|scatter|column|bar|area
        self.var_stacked = tk.BooleanVar(value=False)
        self.var_prefer_outline = tk.BooleanVar(value=True)
        self.var_fit_image = tk.BooleanVar(value=True)


        # Axis scale types
        self.x_scale = tk.StringVar(value=AxisScale.LINEAR.value)
        self.y_scale = tk.StringVar(value=AxisScale.LINEAR.value)
        self.date_fmt = tk.StringVar(value="%Y")

        # Axis value entries
        self.var_x0_val = tk.StringVar(value="0")
        self.var_x1_val = tk.StringVar(value="100")
        self.var_y0_val = tk.StringVar(value="0")
        self.var_y1_val = tk.StringVar(value="100")
        self.var_categories = tk.StringVar(value="cat1;cat2;cat3")

        self.var_x_step = tk.DoubleVar(value=1.0)
        self.var_x_step_unit = tk.StringVar(value="days")
        self.var_y_step = tk.DoubleVar(value=1.0)
        self.var_y_step_unit = tk.StringVar(value="days")
        self.var_tol = tk.IntVar(value=20)
        self.var_sample_mode = tk.StringVar(value="Free")
        self.var_scatter_match_thresh = tk.DoubleVar(value=0.6)

        self._date_unit_user_set = False
        self._date_unit_auto: Optional[str] = None

        # editing / selection
        self._active_series_id: Optional[int] = None
        self._drag_idx: Optional[int] = None  # point index in active series
        self._drag_series_mode: Optional[str] = None  # "line" or "scatter" (used for post-drag resequencing)
        self._toggle_dragging: bool = False
        self._toggle_seen: set[int] = set()
        self._edit_radius = 8
        self._suppress_series_mode_change = False
        self._scatter_rb_start: Optional[Tuple[int, int]] = None
        self._scatter_rb_id: Optional[int] = None
        self._scatter_rb_active: bool = False

        # axis click staging
        self._pending_axis: Optional[str] = None  # 'x0','x1','y0','y1' progress

        self._build_ui()
        self._on_series_mode_change()
        self._render_image()
        self._update_tip()

    # ---------- UI ----------
    def _build_ui(self):
        root = ttk.Frame(self, padding=8)
        root.pack(fill="both", expand=True)

        self._panes = ttk.Panedwindow(root, orient="horizontal")
        self._panes.pack(fill="both", expand=True)

        left = ttk.Frame(self._panes)
        right = ttk.Frame(self._panes, width=340)
        self._panes.add(left, weight=2)
        self._panes.add(right, weight=1)

        # Toolbar
        bar = ttk.Frame(left)
        bar.pack(side="top", fill="x")

        ttk.Label(bar, text="Tool:").pack(side="left")
        for lbl, val in [("Set Region","roi"), ("Set X ticks","xaxis"), ("Set Y ticks","yaxis"), ("Add series","addseries"),("Edit series","editseries")]:
            ttk.Radiobutton(bar, text=lbl, value=val, variable=self.tool_mode, command=self._on_tool_change).pack(side="left", padx=(8,0))

        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=10)
        ttk.Label(bar, text="Series type:").pack(side="left")

        series_combo = ttk.Combobox(
            bar, textvariable=self.series_mode, state="readonly", width=10,
            values=("line", "scatter", "column", "bar", "area")
        )
        series_combo.pack(side="left", padx=(6,0))
        series_combo.bind("<<ComboboxSelected>>", lambda e: self._on_series_mode_change())

        ttk.Checkbutton(
            bar, text="Stacked", variable=self.var_stacked,
            command=self._on_series_mode_change
        ).pack(side="left", padx=(10,0))

        # Loupe and Tip
        loupe_frm = ttk.Frame(left)
        loupe_frm.pack(side="top", fill="x", pady=(8,0))
        ttk.Label(loupe_frm, text="Loupe:").pack(side="left")
        self.loupe = tk.Canvas(loupe_frm, width=120, height=120, highlightthickness=1, highlightbackground="#666")
        self.loupe.pack(side="left", padx=(6,0))
        ttk.Checkbutton(
            loupe_frm, text="Fit", variable=self.var_fit_image,
            command=self._on_fit_toggle
        ).pack(side="left", padx=(10,0))
        self.tip_var = tk.StringVar(value="")
        self.tip_label = ttk.Label(loupe_frm, textvariable=self.tip_var, wraplength=900, justify="left")
        self.tip_label.pack(side="left", padx=(10,0), fill="x", expand=True)

        # Canvas
        self.canvas = tk.Canvas(left, background="#111", highlightthickness=1, highlightbackground="#333")
        self.canvas.pack(side="bottom", fill="both", expand=True, pady=(8,0))
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Shift-ButtonPress-1>", self._on_scatter_rubberband_press)
        self.canvas.bind("<Shift-B1-Motion>", self._on_scatter_rubberband_motion)
        self.canvas.bind("<Shift-ButtonRelease-1>", self._on_scatter_rubberband_release)
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<Button-3>", self._on_right_click)
        self.canvas.bind("<Control-ButtonPress-1>", self._on_ctrl_toggle_press)
        self.canvas.bind("<Control-B1-Motion>", self._on_ctrl_toggle_drag)
        self.canvas.bind("<Control-ButtonRelease-1>", self._on_ctrl_toggle_release)
        self.canvas.bind("<Control-ButtonPress-3>", self._on_ctrl_toggle_press)
        self.canvas.bind("<Control-B3-Motion>", self._on_ctrl_toggle_drag)
        self.canvas.bind("<Control-ButtonRelease-3>", self._on_ctrl_toggle_release)

        # Right panel: axis config
        ax = ttk.LabelFrame(right, text="Calibration", padding=8)
        ax.pack(side="top", fill="x")

        self.axis_px_values = tk.StringVar(value="")
        ttk.Label(ax, textvariable=self.axis_px_values).pack(fill="x", pady=(2, 0))

        sample_row = ttk.Frame(ax)
        sample_row.pack(fill="x", pady=(6, 0))
        ttk.Label(sample_row, text="Sample:").pack(side="left")
        self.cmb_sample_mode = ttk.Combobox(
            sample_row, textvariable=self.var_sample_mode, state="readonly", width=10,
            values=("Free", "Fixed X", "Fixed Y")
        )
        self.cmb_sample_mode.pack(side="left", padx=(6, 0))
        self.cmb_sample_mode.bind("<<ComboboxSelected>>", lambda _e: self._on_sample_mode_change())

        grid = ttk.Frame(ax)
        grid.pack(fill="x")
        for col in range(6):
            grid.columnconfigure(col, weight=1)

        x_axis_hdr = ttk.Frame(grid)
        x_axis_hdr.grid(row=0, column=0, sticky="ew", pady=(6, 0), columnspan=6)
        ttk.Label(x_axis_hdr, text="X axis").pack(side="left")
        ttk.Separator(x_axis_hdr, orient="horizontal").pack(side="left", fill="x", expand=True, padx=(8, 0))
        ttk.Label(grid, text="scale").grid(row=1, column=0, sticky="w")
        self.cmb_x_scale = ttk.Combobox(
            grid, textvariable=self.x_scale, state="readonly", width=10,
            values=[AxisScale.LINEAR.value, AxisScale.LOG10.value, AxisScale.DATE.value, AxisScale.CATEGORICAL.value]
        )
        self.cmb_x_scale.grid(row=1, column=1, sticky="w", padx=(6, 0))
        self.cmb_x_scale.bind("<<ComboboxSelected>>", lambda _e: self._on_x_scale_change())

        self.lbl_date_fmt = ttk.Label(grid, text="date fmt")
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
        self.cmb_date_fmt = ttk.Combobox(
            grid, textvariable=self.date_fmt, state="normal", width=14,
            values=self.date_fmt_options
        )

        self.var_x0_label = tk.StringVar(value="x0")
        self.var_x1_label = tk.StringVar(value="x1")
        self.lbl_x0 = ttk.Label(grid, textvariable=self.var_x0_label)
        self.ent_x0 = ttk.Entry(grid, textvariable=self.var_x0_val, width=10)
        self.lbl_x1 = ttk.Label(grid, textvariable=self.var_x1_label)
        self.ent_x1 = ttk.Entry(grid, textvariable=self.var_x1_val, width=10)
        self.var_x_step_label = tk.StringVar(value="step")
        self.lbl_x_step = ttk.Label(grid, textvariable=self.var_x_step_label)
        self.ent_x_step = ttk.Entry(grid, textvariable=self.var_x_step, width=8)
        self.lbl_x_step_units = ttk.Label(grid, text="unit")
        self.cmb_x_step_units = ttk.Combobox(
            grid, textvariable=self.var_x_step_unit, state="readonly", width=9,
            values=("seconds", "minutes", "hours", "days", "weeks", "months", "quarters", "years")
        )
        self.cmb_x_step_units.bind("<<ComboboxSelected>>", lambda _e: self._on_x_step_unit_changed())

        self.lbl_x_step.grid(row=1, column=2, sticky="w", padx=(8, 0))
        self.ent_x_step.grid(row=1, column=3, sticky="w")
        self.lbl_x_step_units.grid(row=1, column=4, sticky="w", padx=(8, 0))
        self.cmb_x_step_units.grid(row=1, column=5, sticky="w")
        self.lbl_x0.grid(row=3, column=0, sticky="w", pady=(6, 0))
        self.ent_x0.grid(row=3, column=1, sticky="w", pady=(6, 0))
        self.lbl_x1.grid(row=3, column=2, sticky="w", padx=(8, 0), pady=(6, 0))
        self.ent_x1.grid(row=3, column=3, sticky="w", pady=(6, 0))

        y_axis_hdr = ttk.Frame(grid)
        y_axis_hdr.grid(row=5, column=0, sticky="ew", pady=(10, 0), columnspan=6)
        ttk.Label(y_axis_hdr, text="Y axis").pack(side="left")
        ttk.Separator(y_axis_hdr, orient="horizontal").pack(side="left", fill="x", expand=True, padx=(8, 0))
        ttk.Label(grid, text="scale").grid(row=6, column=0, sticky="w", pady=(6,0))
        self.cmb_y_scale = ttk.Combobox(
            grid, textvariable=self.y_scale, state="readonly", width=10,
            values=[AxisScale.LINEAR.value, AxisScale.LOG10.value, AxisScale.DATE.value, AxisScale.CATEGORICAL.value]
        )
        self.cmb_y_scale.grid(row=6, column=1, sticky="w", padx=(6,0), pady=(6,0))
        self.cmb_y_scale.bind("<<ComboboxSelected>>", lambda _e: self._on_y_scale_change())

        self.lbl_y0 = ttk.Label(grid, text="y0")
        self.ent_y0 = ttk.Entry(grid, textvariable=self.var_y0_val, width=10)
        self.lbl_y1 = ttk.Label(grid, text="y1")
        self.ent_y1 = ttk.Entry(grid, textvariable=self.var_y1_val, width=10)
        self.var_y_step_label = tk.StringVar(value="step")
        self.lbl_y_step = ttk.Label(grid, textvariable=self.var_y_step_label)
        self.ent_y_step = ttk.Entry(grid, textvariable=self.var_y_step, width=8)
        self.lbl_y_step_units = ttk.Label(grid, text="unit")
        self.cmb_y_step_units = ttk.Combobox(
            grid, textvariable=self.var_y_step_unit, state="readonly", width=9,
            values=("seconds", "minutes", "hours", "days", "weeks", "months", "quarters", "years")
        )
        self.cmb_y_step_units.bind("<<ComboboxSelected>>", lambda _e: self._on_y_step_unit_changed())

        self.lbl_y_step.grid(row=6, column=2, sticky="w", padx=(8, 0))
        self.ent_y_step.grid(row=6, column=3, sticky="w")
        self.lbl_y_step_units.grid(row=6, column=4, sticky="w", padx=(8, 0))
        self.cmb_y_step_units.grid(row=6, column=5, sticky="w")
        self.lbl_y0.grid(row=8, column=0, sticky="w", pady=(6,0))
        self.ent_y0.grid(row=8, column=1, sticky="w", pady=(6,0))
        self.lbl_y1.grid(row=8, column=2, sticky="w", padx=(8,0), pady=(6,0))
        self.ent_y1.grid(row=8, column=3, sticky="w", pady=(6,0))

        self.lbl_categories = ttk.Label(grid, text="categories")
        self.ent_categories = ttk.Entry(grid, textvariable=self.var_categories, width=28)
        self.lbl_categories_count = ttk.Label(grid, text="")

        cal_btns = ttk.Frame(ax)
        cal_btns.pack(fill="x", pady=(8,0))
        ttk.Button(cal_btns, text="Apply to selected", command=self._apply_calibration_to_selected).pack(side="left")
        ttk.Button(cal_btns, text="Apply to all", command=self._apply_calibration_to_all).pack(side="left", padx=(8,0))

        # Extraction
        ext = ttk.LabelFrame(right, text="Extraction", padding=8)
        ext.pack(side="top", fill="x", pady=(8,0))
        row = ttk.Frame(ext)
        row.pack(fill="x")
        ttk.Label(row, text="Color tol:").pack(side="left")
        ttk.Entry(row, textvariable=self.var_tol, width=6).pack(side="left", padx=(6,0))
        match_row = ttk.Frame(ext)
        match_row.pack(fill="x", pady=(6,0))
        ttk.Label(match_row, text="Match thresh:").pack(side="left")
        ttk.Entry(match_row, textvariable=self.var_scatter_match_thresh, width=6).pack(side="left", padx=(6,0))
        ttk.Checkbutton(
            ext, text="Span mode (bars/areas)", variable=self.var_prefer_outline,
            command=self._on_series_mode_change
        ).pack(side="top", anchor="w", pady=(6,0))

        # Series list
        lst = ttk.LabelFrame(right, text="Series", padding=8)
        lst.pack(side="top", fill="both", expand=True, pady=(8,0))

        self.tree = ttk.Treeview(lst, columns=("enabled","name","cal","n"), show="headings", selectmode="browse", height=12)
        self.tree.heading("enabled", text="On")
        self.tree.heading("name", text="Name")
        self.tree.heading("cal", text="Cal")
        self.tree.heading("n", text="Pts")
        self.tree.column("enabled", width=34, anchor="center")
        self.tree.column("name", width=170, anchor="w")
        self.tree.column("cal", width=60, anchor="center")
        self.tree.column("n", width=60, anchor="e")
        self._configure_tree_rowheight()
        self.tree.pack(fill="both", expand=True)

        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        self.tree.bind("<Double-1>", self._on_tree_double_click)

        btns = ttk.Frame(lst)
        btns.pack(fill="x", pady=(8,0))
        ttk.Button(btns, text="Toggle On/Off", command=self._toggle_series_enabled).pack(side="left")
        ttk.Button(btns, text="Delete", command=self._delete_series).pack(side="left", padx=(8,0))

        # Export buttons
        exp = ttk.Frame(right)
        exp.pack(side="bottom", fill="x", pady=(8,0))

        ttk.Button(exp, text="Append CSV", command=self._append_csv).pack(side="left")
        ttk.Button(exp, text="Export CSV…", command=self._export_csv).pack(side="left", padx=(8,0))
        ttk.Button(exp, text="Close", command=self.destroy).pack(side="right")

        self.after(0, self._set_default_pane_ratio)
        self.date_fmt.trace_add("write", lambda *_: self._on_date_fmt_change())
        self.var_categories.trace_add("write", lambda *_: self._update_category_count())
        self._refresh_scale_ui()

    def _configure_tree_rowheight(self):
        style = ttk.Style(self)
        font_name = style.lookup("Treeview", "font") or "TkDefaultFont"
        try:
            font = tkfont.nametofont(font_name)
        except Exception:
            font = tkfont.nametofont("TkDefaultFont")
        rowheight = max(18, int(font.metrics("linespace")) + 6)
        style.configure("Series.Treeview", rowheight=rowheight)
        self.tree.configure(style="Series.Treeview")

    def _set_default_pane_ratio(self):
        if not getattr(self, "_panes", None):
            return
        self._panes.update_idletasks()
        total = self._panes.winfo_width()
        if total <= 1:
            return
        # Left ~67%, right ~33% by default.
        self._panes.sashpos(0, int(total * 0.67))

    def _is_x_date_scale(self) -> bool:
        return self.x_scale.get() == AxisScale.DATE.value

    def _is_y_date_scale(self) -> bool:
        return self.y_scale.get() == AxisScale.DATE.value

    def _date_fmt_axis(self) -> str:
        return "y" if self.series_mode.get() == "bar" else "x"

    def _stride_mode(self) -> str:
        if self.series_mode.get() == "bar":
            return "categorical" if self.y_scale.get() == AxisScale.CATEGORICAL.value else "continuous"
        return "categorical" if self.x_scale.get() == AxisScale.CATEGORICAL.value else "continuous"

    def _stride_mode_for_calibration(self, cal: SeriesCalibration, chart_kind: str) -> str:
        if chart_kind == "bar":
            return "categorical" if cal.y_scale == AxisScale.CATEGORICAL.value else "continuous"
        return "categorical" if cal.x_scale == AxisScale.CATEGORICAL.value else "continuous"

    def _on_x_scale_change(self) -> None:
        self._refresh_scale_ui()
        if self._is_x_date_scale() and self._date_fmt_axis() == "x":
            self._ensure_date_values("x")
            self._set_default_date_unit()
        self._update_tip()

    def _on_y_scale_change(self) -> None:
        self._refresh_scale_ui()
        if self._is_y_date_scale() and self._date_fmt_axis() == "y":
            self._ensure_date_values("y")
            self._set_default_date_unit()
        self._update_tip()

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
        date_axis = self._date_fmt_axis()
        is_date = self._is_x_date_scale() if date_axis == "x" else self._is_y_date_scale()
        is_bar = self.series_mode.get() == "bar"
        cat_axis = "y" if is_bar else "x"
        cat_scale = self.y_scale.get() if cat_axis == "y" else self.x_scale.get()
        is_categorical = cat_scale == AxisScale.CATEGORICAL.value

        # Limit categorical option to the active categorical axis.
        x_values = [AxisScale.LINEAR.value, AxisScale.LOG10.value]
        y_values = [AxisScale.LINEAR.value, AxisScale.LOG10.value]
        if is_bar:
            y_values.append(AxisScale.DATE.value)
            y_values.append(AxisScale.CATEGORICAL.value)
        else:
            x_values.append(AxisScale.DATE.value)
            x_values.append(AxisScale.CATEGORICAL.value)
        self.cmb_x_scale.configure(values=x_values)
        self.cmb_y_scale.configure(values=y_values)
        if self.x_scale.get() not in x_values:
            self.x_scale.set(AxisScale.LINEAR.value)
        if self.y_scale.get() not in y_values:
            self.y_scale.set(AxisScale.LINEAR.value)

        if is_date and not is_categorical:
            if date_axis == "x":
                self.lbl_date_fmt.configure(text="date fmt")
                self.var_x0_label.set("x0 (date)")
                self.var_x1_label.set("x1 (date)")
                self.var_x_step_label.set("step")
            else:
                self.lbl_date_fmt.configure(text="date fmt")
                self.var_y_step_label.set("step")
        else:
            self.var_x0_label.set("x0")
            self.var_x1_label.set("x1")
            self.var_x_step_label.set("step")
            self.var_y_step_label.set("step")

        self.lbl_date_fmt.grid_remove()
        self.cmb_date_fmt.grid_remove()
        if is_date and not is_categorical:
            if date_axis == "x":
                self.lbl_date_fmt.grid(row=2, column=0, sticky="w", pady=(6, 0))
                self.cmb_date_fmt.grid(row=2, column=1, sticky="w", pady=(6, 0))
            else:
                self.lbl_date_fmt.grid(row=7, column=0, sticky="w", pady=(6, 0))
                self.cmb_date_fmt.grid(row=7, column=1, sticky="w", pady=(6, 0))

        if is_categorical:
            if cat_axis == "y":
                self.lbl_categories.configure(text="Categories (Y)")
                self.lbl_y0.grid_remove()
                self.ent_y0.grid_remove()
                self.lbl_y1.grid_remove()
                self.ent_y1.grid_remove()
                self.lbl_y_step.grid_remove()
                self.ent_y_step.grid_remove()
                self.lbl_y_step_units.grid_remove()
                self.cmb_y_step_units.grid_remove()
                self.lbl_x0.grid()
                self.ent_x0.grid()
                self.lbl_x1.grid()
                self.ent_x1.grid()
            else:
                self.lbl_categories.configure(text="Categories (X)")
                self.lbl_x0.grid_remove()
                self.ent_x0.grid_remove()
                self.lbl_x1.grid_remove()
                self.ent_x1.grid_remove()
                self.lbl_x_step.grid_remove()
                self.ent_x_step.grid_remove()
                self.lbl_x_step_units.grid_remove()
                self.cmb_x_step_units.grid_remove()
                self.lbl_y0.grid()
                self.ent_y0.grid()
                self.lbl_y1.grid()
                self.ent_y1.grid()
            if cat_axis == "y":
                self.lbl_categories.grid(row=9, column=0, sticky="w", pady=(6, 0))
                self.ent_categories.grid(row=9, column=1, columnspan=4, sticky="ew", pady=(6, 0))
                self.lbl_categories_count.grid(row=9, column=5, sticky="e", padx=(8, 0), pady=(6, 0))
            else:
                self.lbl_categories.grid(row=4, column=0, sticky="w", pady=(6, 0))
                self.ent_categories.grid(row=4, column=1, columnspan=4, sticky="ew", pady=(6, 0))
                self.lbl_categories_count.grid(row=4, column=5, sticky="e", padx=(8, 0), pady=(6, 0))
            self._cat_row_visible = True
            self._ensure_categories_default()
            self._update_category_count()
        else:
            self.lbl_categories.grid_remove()
            self.ent_categories.grid_remove()
            self.lbl_categories_count.grid_remove()
            self._cat_row_visible = False
            self.lbl_x0.grid()
            self.ent_x0.grid()
            self.lbl_x1.grid()
            self.ent_x1.grid()
            self.lbl_x_step.grid()
            self.ent_x_step.grid()
            self.lbl_x_step_units.grid()
            self.cmb_x_step_units.grid()
            self.lbl_y0.grid()
            self.ent_y0.grid()
            self.lbl_y1.grid()
            self.ent_y1.grid()
            self.lbl_y_step.grid()
            self.ent_y_step.grid()
            self.lbl_y_step_units.grid()
            self.cmb_y_step_units.grid()

        sample_mode = self._sample_label_to_mode(self.var_sample_mode.get())
        scatter_x = (self.series_mode.get() == "scatter" and sample_mode == "fixed_x")
        show_x_step = ((not is_bar) and (self.series_mode.get() != "scatter") and not (is_categorical and cat_axis == "x")) or scatter_x
        if show_x_step:
            self.lbl_x_step.grid()
            self.ent_x_step.grid()
            self.lbl_x_step_units.grid()
            self.cmb_x_step_units.grid()
        else:
            self.lbl_x_step.grid_remove()
            self.ent_x_step.grid_remove()
            self.lbl_x_step_units.grid_remove()
            self.cmb_x_step_units.grid_remove()

        show_x_units = (is_date and show_x_step and date_axis == "x")
        if show_x_units:
            self.lbl_x_step_units.grid()
            self.cmb_x_step_units.grid()
        else:
            self.lbl_x_step_units.grid_remove()
            self.cmb_x_step_units.grid_remove()

        scatter_y = (self.series_mode.get() == "scatter" and sample_mode == "fixed_y")
        show_y_step = (is_bar and not (is_categorical and cat_axis == "y")) or scatter_y
        if show_y_step:
            self.lbl_y_step.grid()
            self.ent_y_step.grid()
            self.lbl_y_step_units.grid()
            self.cmb_y_step_units.grid()
        else:
            self.lbl_y_step.grid_remove()
            self.ent_y_step.grid_remove()
            self.lbl_y_step_units.grid_remove()
            self.cmb_y_step_units.grid_remove()

        show_y_units = (is_date and show_y_step and date_axis == "y")
        if show_y_units:
            self.lbl_y_step_units.grid()
            self.cmb_y_step_units.grid()
        else:
            self.lbl_y_step_units.grid_remove()
            self.cmb_y_step_units.grid_remove()

        self.cmb_x_scale.configure(state="readonly")

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

    def _ensure_categories_default(self) -> None:
        if not self.var_categories.get().strip():
            self.var_categories.set("x1;x2;x3")

    def _normalize_categories(self, raw: str) -> str:
        parts = [p.strip() for p in (raw or "").split(";")]
        return ";".join([p for p in parts if p])

    def _parse_categories(self, raw: Optional[str] = None) -> List[str]:
        if raw is None:
            raw = self.var_categories.get()
        parts = [p.strip() for p in (raw or "").split(";")]
        return [p for p in parts if p]

    def _update_category_count(self) -> None:
        if not getattr(self, "_cat_row_visible", False):
            self.lbl_categories_count.configure(text="")
            return
        count = len(self._parse_categories())
        self.lbl_categories_count.configure(text=f"{count} cats")

    def _category_centers_px(self, count: int, *, axis: str, axis_px: Optional[Tuple[int, int]] = None) -> List[float]:
        if count <= 1:
            if axis_px is None:
                if axis == "y":
                    p0, p1 = self._y_axis_px()
                else:
                    p0, p1 = self._x_axis_px()
            else:
                p0, p1 = axis_px
            return [float(p0)]

        if axis_px is None:
            if axis == "y":
                p0, p1 = self._y_axis_px()
            else:
                p0, p1 = self._x_axis_px()
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
        roi = self._roi_px()
        x_axis_px = self._x_axis_px()
        y_axis_px = self._y_axis_px()
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
        scatter_match_thresh = float(self.var_scatter_match_thresh.get())
        return (
            roi,
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
            scatter_match_thresh,
        )

    def _calibration_key_from_series(self, s: Series) -> tuple:
        cal = s.calibration
        return (
            cal.roi_px,
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
            cal.scatter_match_thresh,
        )

    def _make_series_calibration_from_ui(self) -> SeriesCalibration:
        key = self._calibration_key_from_ui()
        name = self._calibration_names.get(key)
        if name is None:
            name = f"calib{self._next_calibration_id}"
            self._next_calibration_id += 1
            self._calibration_names[key] = name
        (
            roi,
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
            scatter_match_thresh,
        ) = key
        return SeriesCalibration(
            name=name,
            roi_px=roi,
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
            scatter_match_thresh=float(scatter_match_thresh),
        )

    def _apply_series_calibration_to_ui(self, cal: SeriesCalibration) -> None:
        self.state.xmin_px, self.state.ymin_px, self.state.xmax_px, self.state.ymax_px = cal.roi_px
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
        self.var_scatter_match_thresh.set(float(cal.scatter_match_thresh))
        self._date_unit_user_set = True
        self._refresh_scale_ui()
        self._update_sample_mode_ui()
        self._update_tip()
        self._redraw_overlay()

    def _set_ui_mode_from_series(self, s: Series) -> None:
        kind = getattr(s, "chart_kind", getattr(s, "mode", "line"))
        self.series_mode.set(kind)

    def _pixel_bounds_changes(self, old: SeriesCalibration, new: SeriesCalibration) -> List[str]:
        changes: List[str] = []
        if old.roi_px != new.roi_px:
            changes.append("ROI")
        if old.x_axis_px != new.x_axis_px:
            changes.append("X axis pixels")
        if old.y_axis_px != new.y_axis_px:
            changes.append("Y axis pixels")
        return changes

    def _apply_calibration_to_series(self, targets: List[Series]) -> None:
        if not targets:
            messagebox.showinfo("Calibration", "No series to update.")
            return

        mode = self.series_mode.get()
        mismatched = [s for s in targets if getattr(s, "chart_kind", s.mode) != mode]
        if mismatched:
            messagebox.showinfo(
                "Calibration",
                "Series type changes are not supported. Create a new series for a different type."
            )
            targets = [s for s in targets if s not in mismatched]
        if not targets:
            messagebox.showinfo("Calibration", "No series updated (series type mismatch).")
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
            self._update_tree_row(s)

        if to_reextract:
            messagebox.showinfo(
                "Calibration",
                f"Re-extracting {len(to_reextract)} series because pixel bounds changed: "
                + ", ".join(sorted(change_set))
            )
            for s in to_reextract:
                try:
                    self._extract_series(s)
                except Exception as e:
                    messagebox.showerror("Series extraction failed", str(e))
        else:
            messagebox.showinfo("Calibration", f"Updated {updated} series. No re-extraction needed.")
        self._redraw_overlay()

    def _apply_calibration_to_selected(self) -> None:
        if self._active_series_id is None:
            messagebox.showinfo("Calibration", "Select a series first.")
            return
        s = self._get_series(self._active_series_id)
        if s is None:
            return
        self._apply_calibration_to_series([s])

    def _apply_calibration_to_all(self) -> None:
        if not self.series:
            messagebox.showinfo("Calibration", "No series to update yet.")
            return
        self._apply_calibration_to_series(list(self.series))

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
            x1_dt = self._add_months(base, 12)
            if axis == "y":
                self.var_y1_val.set(x1_dt.strftime(fmt))
            else:
                self.var_x1_val.set(x1_dt.strftime(fmt))

    @staticmethod
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


    def _update_tip(self):
        mode = self.tool_mode.get()
        series_mode = self.series_mode.get()  # line|scatter|column|bar|area

        if mode == "roi":
            msg = (
                "Set Region: Click and drag to define the rectangular region of interest (ROI). "
                "Only pixels inside this region are scanned for the series by their color. "
                "If you do not set tick pixels, tick positions default to the region edges."
            )
        elif mode == "xaxis":
            msg = (
                "Set X ticks: Click any two tick positions on the vertical axis. "
                "Then enter their values in the Calibration panel (x0 and x1) to the right. "
                "These ticks define the mapping from image pixels to chart units for the X axis."
            )
        elif mode == "yaxis":
            msg = (
                "Set Y ticks: Click any two tick positions on the vertical axis. "
                "Then enter their values in the Calibration panel (y0 and y1) to the right. "
                "These ticks define the mapping from image pixels to chart units for the Y axis."
            )
        elif mode == "addseries":
            if series_mode == "scatter":
                msg = (
                    "Add series (Scatter): Click a marker color to pick the series. "
                    "Shift+drag draws a template window for matching scatter markers. "
                    "The tool detects colored marker blobs inside the region and exports their (x,y) coordinates."
                )
            elif series_mode == "line":
                msg = (
                    "Add series (Line): Click directly on a line in the chart to generate the series data. "
                    "The tool tracks the line from the clicked point across the region and samples it onto the X grid. "
                    "X step controls the output sampling grid; missing samples are interpolated/flatlined. "
                    "Color tol controls how closely pixels must match the clicked color. "
                    "Ctrl+click adds detection seeds (shown as cyan rings) and rebuilds the line trace."
                )
            elif series_mode == "column":
                msg = (
                    "Add series (Column): Click inside a bar segment to pick its color. "
                    "The tool reads the bar boundary (usually the outline) to estimate the value. "
                    "Set X scale to categorical to auto-detect bar centers; otherwise samples on the X grid. "
                    "Stacked indicates the series is a cumulative boundary; export can emit deltas between boundaries."
                )
            elif series_mode == "bar":
                msg = (
                    "Add series (Bar): Click inside a horizontal bar segment to pick its color. "
                    "The tool reads the bar boundary (usually the outline) to estimate the value. "
                    "Set Y scale to categorical to auto-detect bar centers along Y. "
                    "Stacked indicates the series is a cumulative boundary; export can emit deltas between boundaries."
                )
            else:  # area
                msg = (
                    "Add series (Area): Click inside a filled area to pick its color. "
                    "The tool reads the outer area boundary (usually the outline) to estimate the value. "
                    "Set X scale to categorical to auto-detect distinct x-runs; otherwise samples on the X grid."
                )
        elif mode == "editseries":
            if series_mode == "scatter":
                msg = (
                    "Edit series (Scatter): Select a series in the table. Right-click toggles a point NA/disabled. "
                    "Drag points to reposition. Right-click away from points inserts a new point (auto-sorted by X). "
                    "Ctrl+drag (left or right button) toggles enable/NA for points you sweep over. "
                    "Edits affect exported CSV values."
                )
            else:
                msg = (
                    "Edit series: Select a series in the table, then drag points to correct values. "
                    "For line/column/area, dragging is vertical only (X fixed to the sampling grid/category). "
                    "For bars, dragging adjusts the bar length (X) while category position (Y) stays fixed. "
                    "Right-click a point toggles NA/disabled. Edits affect exported CSV values."
                )
        else:
            # fallback / none
            msg = (
                "Select a tool mode above. Region controls what pixels are scanned; X/Y ticks control how pixels map to data units."
            )

        self.tip_var.set(msg)

    def _on_series_mode_change(self):
        if getattr(self, "_suppress_series_mode_change", False):
            return
        if self.tool_mode.get() == "editseries" and self._active_series_id is not None:
            s = self._get_series(self._active_series_id)
            if s is not None:
                desired = getattr(s, "chart_kind", s.mode)
                if self.series_mode.get() != desired:
                    messagebox.showinfo(
                        "Series type",
                        "Series type changes are not supported. Create a new series for a different type."
                    )
                    self._suppress_series_mode_change = True
                    self.series_mode.set(desired)
                    self._suppress_series_mode_change = False
                    return
        if self.series_mode.get() in ("column", "bar", "area"):
            self.var_prefer_outline.set(True)
        else:
            self.var_prefer_outline.set(False)
        if self.series_mode.get() == "scatter":
            self.var_sample_mode.set("Free")
        self._update_tip()
        self._refresh_scale_ui()
        self._update_sample_mode_ui()

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

    def _on_canvas_configure(self, _evt=None):
        # Avoid thrashing when resizing: schedule a single re-render
        if getattr(self, "_render_after_id", None) is not None:
            try:
                self.after_cancel(self._render_after_id)
            except Exception:
                pass
        self._render_after_id = self.after(30, self._render_image)

    def _on_tool_change(self):
        mode = self.tool_mode.get()
        self._pending_axis = None
        if mode == "xaxis":
            self._pending_axis = "x0"
        elif mode == "yaxis":
            self._pending_axis = "y0"
        elif mode == "editseries" and self._active_series_id is not None:
            s = self._get_series(self._active_series_id)
            if s is not None:
                self._set_ui_mode_from_series(s)
                self._apply_series_calibration_to_ui(s.calibration)
        self._update_tip()
        self._redraw_overlay()

    # ---------- Image rendering ----------
    def _render_image(self):
        self.canvas.delete("all")
        # Fit image into canvas (optional upscale)
        self.canvas.update_idletasks()
        cw = max(10, self.canvas.winfo_width())
        ch = max(10, self.canvas.winfo_height())

        # scale factor
        sx = cw / self._iw
        sy = ch / self._ih
        if self.var_fit_image.get():
            self._scale = min(sx, sy)
        else:
            self._scale = min(1.0, sx, sy)
        disp_w = int(self._iw * self._scale)
        disp_h = int(self._ih * self._scale)

        self._offx = (cw - disp_w)//2
        self._offy = (ch - disp_h)//2

        disp = self._pil.resize((disp_w, disp_h), Image.NEAREST)
        self._photo = ImageTk.PhotoImage(disp)
        self.canvas.create_image(self._offx, self._offy, image=self._photo, anchor="nw", tags=("img",))

        self._redraw_overlay()

    def _on_fit_toggle(self):
        self._render_image()

    def _redraw_overlay(self):
        self.canvas.delete("overlay")

        # ROI rectangle
        x0,y0,x1,y1 = self._roi_px()
        ax0, ay0 = self._to_canvas(x0, y0)
        ax1, ay1 = self._to_canvas(x1, y1)
        self.canvas.create_rectangle(ax0, ay0, ax1, ay1, outline="#00B4FF", width=2, tags=("overlay",))
        self.canvas.create_text(ax1-50, ay1-18, text="Region", fill="#00B4FF", anchor="nw", tags=("overlay",))

        # Axis markers
        # X axis pixels
        x0_px, x1_px = self._x_axis_px()
        y0_px, y1_px = self._y_axis_px()

        # draw vertical lines for x0/x1
        for lbl, xpx in [("x0", x0_px), ("x1", x1_px)]:
            cx, cy0 = self._to_canvas(xpx, y0)
            _, cy1 = self._to_canvas(xpx, y1)
            self.canvas.create_line(cx, cy0, cx, cy1, fill="#66FF66", width=1, tags=("overlay",))
            self.canvas.create_text(cx+2, ay0+2, text=lbl, fill="#66FF66", anchor="nw", tags=("overlay",))

        # draw horizontal lines for y0/y1
        for lbl, ypx in [("y0", y0_px), ("y1", y1_px)]:
            cx0, cy = self._to_canvas(x0, ypx)
            cx1, _ = self._to_canvas(x1, ypx)
            self.canvas.create_line(cx0, cy, cx1, cy, fill="#FFCC66", width=1, tags=("overlay",))
            self.canvas.create_text(ax0+2, cy+2, text=lbl, fill="#FFCC66", anchor="nw", tags=("overlay",))

        # # active series overlay
        # if self._active_series_id is not None:
        #     s = self._get_series(self._active_series_id)
        #     if s and s.px_points:
        #         pts = [self._to_canvas(x,y) for (x,y) in s.px_points]
        #         # polyline for line mode
        #         if s.mode == "line" and len(pts) >= 2:
        #             flat = [v for xy in pts for v in xy]
        #             self.canvas.create_line(*flat, fill="yellow", width=2, tags=("overlay","series"))
        #         for i,(cx,cy) in enumerate(pts):
        #             r = 3
        #             fill = "yellow" if (not s.point_enabled or s.point_enabled[i]) else "#666"
        #             self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r, fill=fill, outline="", tags=("overlay","pt",f"pt_{i}"))
        # active series overlay
        if self._active_series_id is not None:
            s = self._get_series(self._active_series_id)
            if s and s.px_points:
                pts = [self._to_canvas(x, y) for (x, y) in s.px_points]

                # polyline for line mode
                if s.mode in ("line","column","area","bar") and len(pts) >= 2:
                    flat = [v for xy in pts for v in xy]

                    # "halo" (understroke then overstoke)
                    self.canvas.create_line(
                        *flat, fill="black", width=5,
                        capstyle="round", joinstyle="round",
                        tags=("overlay", "series")
                    )
                    self.canvas.create_line(
                        *flat, fill="white", width=2,
                        capstyle="round", joinstyle="round",
                        tags=("overlay", "series")
                    )

                for i, (cx, cy) in enumerate(pts):
                    r = 3
                    enabled = (not s.point_enabled) or s.point_enabled[i]

                    if enabled:
                        # outer ring + inner dot
                        self.canvas.create_oval(cx-(r+2), cy-(r+2), cx+(r+2), cy+(r+2),
                                                outline="black", width=2, fill="",
                                                tags=("overlay","pt",f"pt_{i}"))
                        self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r,
                                                outline="white", width=2, fill="",
                                                tags=("overlay","pt",f"pt_{i}"))
                    else:
                        self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r,
                                                fill="#666", outline="",
                                                tags=("overlay","pt",f"pt_{i}"))
            # detection seed markers
        if self._active_series_id is not None:
            s = self._get_series(self._active_series_id)
            if s and (s.seed_px or s.extra_seeds_px):
                seeds = []
                if s.seed_px:
                    seeds.append((s.seed_px, "#00B4FF"))
                for ex in (s.extra_seeds_px or []):
                    seeds.append((ex, "#00E5FF"))
                for (sx, sy), color in seeds:
                    marker_bbox = getattr(s, "seed_marker_bbox_px", None)
                    if marker_bbox:
                        mx0, my0, mx1, my1 = marker_bbox
                        cx0, cy0 = self._to_canvas(int(mx0), int(my0))
                        cx1, cy1 = self._to_canvas(int(mx1), int(my1))
                        self.canvas.create_oval(cx0, cy0, cx1, cy1,
                                                outline="black", width=2, fill="",
                                                tags=("overlay","seed"))
                        self.canvas.create_oval(cx0+1, cy0+1, cx1-1, cy1-1,
                                                outline=color, width=2, fill="",
                                                tags=("overlay","seed"))
                    else:
                        cx, cy = self._to_canvas(int(sx), int(sy))
                        r = 5
                        self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r,
                                                outline="black", width=2, fill="",
                                                tags=("overlay","seed"))
                        self.canvas.create_oval(cx-(r-1), cy-(r-1), cx+(r-1), cy+(r-1),
                                                outline=color, width=2, fill="",
                                                tags=("overlay","seed"))
        self._update_axis_pixels_label()

    def _update_axis_pixels_label(self) -> None:
        rx0, ry0, rx1, ry1 = self._roi_px()
        x0_px, x1_px = self._x_axis_px()
        y0_px, y1_px = self._y_axis_px()
        self.axis_px_values.set(
            f"image size: {self._iw}x{self._ih}\n"
            f"data region: x: {rx0}:{rx1}, y: {ry0}:{ry1}\n"
            f"x ticks x0:{x0_px} x1:{x1_px}, y ticks y0:{y0_px} y1:{y1_px}"
        )

    # ---------- coordinate transforms ----------
    def _to_canvas(self, xpx: int, ypx: int) -> Tuple[int,int]:
        cx = int(self._offx + xpx * self._scale)
        cy = int(self._offy + ypx * self._scale)
        return cx, cy

    def _to_image_px(self, cx: float, cy: float) -> Tuple[int,int]:
        x = int(round((cx - self._offx) / self._scale))
        y = int(round((cy - self._offy) / self._scale))
        x = max(0, min(self._iw-1, x))
        y = max(0, min(self._ih-1, y))
        return x, y

    def _roi_px(self) -> Tuple[int,int,int,int]:
        return (self.state.xmin_px, self.state.ymin_px, self.state.xmax_px, self.state.ymax_px)

    def _x_axis_px(self) -> Tuple[int,int]:
        x0,y0,x1,y1 = self._roi_px()
        return (self.state.x0_px if self.state.x0_px is not None else x0,
                self.state.x1_px if self.state.x1_px is not None else x1)

    def _y_axis_px(self) -> Tuple[int,int]:
        x0,y0,x1,y1 = self._roi_px()
        return (self.state.y0_px if self.state.y0_px is not None else y1,   # y0 defaults to bottom
                self.state.y1_px if self.state.y1_px is not None else y0)   # y1 defaults to top

    # ---------- clicks ----------
    def _on_click(self, event):
        if event.state & 0x0004:  # Control held: handled by ctrl bindings
            return
        if event.state & 0x0001:  # Shift held: handled by shift bindings
            if self.tool_mode.get() == "addseries" and self.series_mode.get() == "scatter":
                return
        xpx, ypx = self._to_image_px(event.x, event.y)
        mode = self.tool_mode.get()

        if mode == "roi":
            self._roi_drag_start = (xpx, ypx)
            self._roi_dragging = True
            return

        if mode == "xaxis":
            if self._pending_axis == "x0":
                self.state.x0_px = xpx
                self._pending_axis = "x1"
            else:
                self.state.x1_px = xpx
                self._pending_axis = "x0"
            self._redraw_overlay()
            return

        if mode == "yaxis":
            if self._pending_axis == "y0":
                self.state.y0_px = ypx
                self._pending_axis = "y1"
            else:
                self.state.y1_px = ypx
                self._pending_axis = "y0"
            self._redraw_overlay()
            return

        if mode == "addseries":
            self._add_series_from_click(xpx, ypx)
            return

        # mode == none: edit points
        self._start_point_drag(xpx, ypx)


    def _start_point_drag(self, xpx: int, ypx: int):
        """Begin drag-editing a point in the active series (editseries mode only)."""
        if self.tool_mode.get() != "editseries":
            return
        if self._active_series_id is None:
            return

        s = self._get_series(self._active_series_id)
        if not s or not getattr(s, "px_points", None):
            return

        idx = self._nearest_point_index(s.px_points, xpx, ypx, getattr(self, "_edit_radius", 6))
        if idx is None:
            return

        self._drag_idx = int(idx)
        self._drag_series_mode = getattr(s, "mode", None)

    def _on_drag(self, event):
        if self._scatter_rb_active:
            return
        xpx, ypx = self._to_image_px(event.x, event.y)
        mode = self.tool_mode.get()
        if mode == "roi" and getattr(self, "_roi_dragging", False):
            sx, sy = self._roi_drag_start
            x0 = min(sx, xpx); x1 = max(sx, xpx)
            y0 = min(sy, ypx); y1 = max(sy, ypx)

            # enforce min size
            if (x1-x0) >= 5 and (y1-y0) >= 5:
                self.state.xmin_px, self.state.ymin_px, self.state.xmax_px, self.state.ymax_px = x0, y0, x1, y1
            self._redraw_overlay()
            return

        # drag edit point (line mode y only)
        if mode == "editseries" and self._drag_idx is not None and self._active_series_id is not None:
            s = self._get_series(self._active_series_id)
            if not s:
                return

            roi_x0, roi_y0, roi_x1, roi_y1 = self._roi_px()
            xpx = max(roi_x0, min(roi_x1, xpx))
            ypx = max(roi_y0, min(roi_y1, ypx))

            cal = self._build_calibration()

            if s.mode in ("line", "column", "area"):
                # keep x fixed; adjust y only
                x_fixed = s.px_points[self._drag_idx][0]
                s.px_points[self._drag_idx] = (x_fixed, ypx)

                y_data = cal.y_px_to_data(ypx)
                x_data = s.points[self._drag_idx][0]
                s.points[self._drag_idx] = (x_data, float(y_data))

            elif s.mode == "bar":
                # keep category position fixed (y); adjust length along x only
                y_fixed = s.px_points[self._drag_idx][1]
                s.px_points[self._drag_idx] = (xpx, y_fixed)

                x_data = cal.x_px_to_data(xpx)
                cat = s.points[self._drag_idx][0]
                s.points[self._drag_idx] = (float(cat), float(x_data))

            else:  # scatter
                sample_mode = getattr(s.calibration, "sample_mode", "free")
                if sample_mode == "fixed_x":
                    x_fixed = s.px_points[self._drag_idx][0]
                    x_data = s.points[self._drag_idx][0]
                    s.px_points[self._drag_idx] = (x_fixed, ypx)
                    y_data = cal.y_px_to_data(ypx)
                    s.points[self._drag_idx] = (float(x_data), float(y_data))
                elif sample_mode == "fixed_y":
                    y_fixed = s.px_points[self._drag_idx][1]
                    y_data = s.points[self._drag_idx][1]
                    s.px_points[self._drag_idx] = (xpx, y_fixed)
                    x_data = cal.x_px_to_data(xpx)
                    s.points[self._drag_idx] = (float(x_data), float(y_data))
                else:
                    s.px_points[self._drag_idx] = (xpx, ypx)
                    x_data = cal.x_px_to_data(xpx)
                    y_data = cal.y_px_to_data(ypx)
                    s.points[self._drag_idx] = (float(x_data), float(y_data))

            self._update_tree_row(s)
            self._redraw_overlay()
            self._draw_loupe(xpx, ypx)
    def _on_release(self, event):
        if self._scatter_rb_active:
            return
        if self.tool_mode.get() == "roi":
            self._roi_dragging = False

        # If we were dragging a scatter point, resequence by ascending X on release.
        if self.tool_mode.get() == "editseries" and self._active_series_id is not None and self._drag_series_mode == "scatter":
            s = self._get_series(self._active_series_id)
            if s and s.points and s.px_points and len(s.points) == len(s.px_points):
                order = sorted(range(len(s.points)), key=lambda i: (s.points[i][0], s.points[i][1]))
                s.points = [s.points[i] for i in order]
                s.px_points = [s.px_points[i] for i in order]
                if s.point_enabled and len(s.point_enabled) == len(order):
                    s.point_enabled = [s.point_enabled[i] for i in order]
                self._update_tree_row(s)
                self._redraw_overlay()

        self._drag_idx = None
        self._drag_series_mode = None
        self._toggle_dragging = False
        self._toggle_seen.clear()

    def _on_scatter_rubberband_press(self, event):
        if self.tool_mode.get() != "addseries" or self.series_mode.get() != "scatter":
            return
        xpx, ypx = self._to_image_px(event.x, event.y)
        self._scatter_rb_start = (xpx, ypx)
        self._scatter_rb_active = True
        cx, cy = self._to_canvas(xpx, ypx)
        self._scatter_rb_id = self.canvas.create_rectangle(
            cx, cy, cx, cy, outline="#FFD166", width=1, tags=("overlay", "scatter_rb")
        )

    def _on_scatter_rubberband_motion(self, event):
        if not self._scatter_rb_active or self._scatter_rb_start is None:
            return
        xpx, ypx = self._to_image_px(event.x, event.y)
        sx, sy = self._scatter_rb_start
        x0 = min(sx, xpx); x1 = max(sx, xpx)
        y0 = min(sy, ypx); y1 = max(sy, ypx)
        cx0, cy0 = self._to_canvas(x0, y0)
        cx1, cy1 = self._to_canvas(x1, y1)
        if self._scatter_rb_id is not None:
            self.canvas.coords(self._scatter_rb_id, cx0, cy0, cx1, cy1)

    def _on_scatter_rubberband_release(self, event):
        if not self._scatter_rb_active or self._scatter_rb_start is None:
            return
        xpx, ypx = self._to_image_px(event.x, event.y)
        sx, sy = self._scatter_rb_start
        x0 = min(sx, xpx); x1 = max(sx, xpx)
        y0 = min(sy, ypx); y1 = max(sy, ypx)
        self._scatter_rb_active = False
        self._scatter_rb_start = None
        if self._scatter_rb_id is not None:
            self.canvas.delete(self._scatter_rb_id)
            self._scatter_rb_id = None

        if (x1 - x0) < 3 or (y1 - y0) < 3:
            self._add_series_from_click(xpx, ypx)
            return

        self._add_scatter_series_from_bbox((x0, y0, x1, y1))
    def _on_right_click(self, event):
        # Edit-series helpers:
        # - Right-click on point: toggle enabled/NA state
        # - Right-click away from points (scatter only): insert a new point at that location (sorted by X)
        if self.tool_mode.get() != "editseries" or self._active_series_id is None:
            return

        xpx, ypx = self._to_image_px(event.x, event.y)
        s = self._get_series(self._active_series_id)
        if not s:
            return

        # If we have points, check if this right-click hits a point
        idx = None
        if s.px_points:
            idx = self._nearest_point_index(s.px_points, xpx, ypx, self._edit_radius)

        if idx is None:
            # Insert new point for scatter series when clicking away from points
            if s.mode != "scatter":
                return

            roi_x0, roi_y0, roi_x1, roi_y1 = self._roi_px()
            xpx = max(roi_x0, min(roi_x1, xpx))
            ypx = max(roi_y0, min(roi_y1, ypx))

            cal = self._build_calibration()
            x_data = float(cal.x_px_to_data(xpx))
            y_data = float(cal.y_px_to_data(ypx))

            if s.points is None:
                s.points = []
            if s.px_points is None:
                s.px_points = []

            # Ensure enabled flags exist
            if not s.point_enabled:
                s.point_enabled = [True] * len(s.points)

            # Insert by x (stable if same x)
            insert_at = 0
            if s.points:
                xs = [p[0] for p in s.points]
                insert_at = int(bisect.bisect_left(xs, x_data))
            s.points.insert(insert_at, (x_data, y_data))
            s.px_points.insert(insert_at, (int(xpx), int(ypx)))
            s.point_enabled.insert(insert_at, True)

            self._update_tree_row(s)
            self._redraw_overlay()
            return

        # Toggle enable on an existing point
        if not s.point_enabled:
            s.point_enabled = [True] * len(s.points)
        s.point_enabled[idx] = not s.point_enabled[idx]
        self._redraw_overlay()

    def _on_ctrl_toggle_press(self, event):
        if self.tool_mode.get() == "addseries":
            if self._add_extra_seed_from_click(event):
                return
        if self.tool_mode.get() != "editseries" or self._active_series_id is None:
            return
        self._toggle_dragging = True
        self._toggle_seen.clear()
        self._ctrl_toggle_at(event)

    def _on_ctrl_toggle_drag(self, event):
        if self.tool_mode.get() == "addseries":
            return
        if not self._toggle_dragging:
            return
        self._ctrl_toggle_at(event)

    def _on_ctrl_toggle_release(self, event):
        self._toggle_dragging = False
        self._toggle_seen.clear()

    def _ctrl_toggle_at(self, event):
        xpx, ypx = self._to_image_px(event.x, event.y)
        s = self._get_series(self._active_series_id) if (self._active_series_id is not None) else None
        if not s or not s.px_points:
            return
        idx = self._nearest_point_index(s.px_points, xpx, ypx, self._edit_radius)
        if idx is None:
            return
        if idx in self._toggle_seen:
            return
        self._toggle_seen.add(idx)
        if not s.point_enabled:
            s.point_enabled = [True] * len(s.points)
        s.point_enabled[idx] = not s.point_enabled[idx]
        self._redraw_overlay()

    def _nearest_point_index(self, pts: List[Tuple[int,int]], x: int, y: int, r: int) -> Optional[int]:
        r2 = r*r
        best = None
        bestd = 1e18
        for i,(px,py) in enumerate(pts):
            d2 = (px-x)*(px-x) + (py-y)*(py-y)
            if d2 <= r2 and d2 < bestd:
                bestd = d2
                best = i
        return best

    def _add_extra_seed_from_click(self, event) -> bool:
        s = None
        if self._active_series_id is not None:
            s = self._get_series(self._active_series_id)
        if s is None and self.series:
            s = self.series[-1]
        if not s or getattr(s, "chart_kind", s.mode) != "line":
            return False

        xpx, ypx = self._to_image_px(event.x, event.y)
        if s.extra_seeds_px is None:
            s.extra_seeds_px = []
        s.extra_seeds_px.append((xpx, ypx))

        try:
            self._extract_series(s)
        except Exception as e:
            messagebox.showerror("Series extraction failed", str(e))
            return True

        self._update_tree_row(s)
        self._redraw_overlay()
        return True

    # ---------- Loupe ----------
    def _on_motion(self, event):
        xpx, ypx = self._to_image_px(event.x, event.y)
        self._draw_loupe(xpx, ypx)

    def _draw_loupe(self, xpx: int, ypx: int):
        # 12x12 region scaled to 10x
        size = 12
        zoom = 10
        x0 = max(0, xpx - size//2)
        y0 = max(0, ypx - size//2)
        x1 = min(self._iw, x0 + size)
        y1 = min(self._ih, y0 + size)
        crop = self._pil.crop((x0,y0,x1,y1)).resize((size*zoom, size*zoom), Image.NEAREST)
        self._loupe_photo = ImageTk.PhotoImage(crop)
        self.loupe.delete("all")
        self.loupe.create_image(0,0, image=self._loupe_photo, anchor="nw")
        # crosshair
        cx = (xpx - x0) * zoom
        cy = (ypx - y0) * zoom
        self.loupe.create_line(cx, 0, cx, size*zoom, fill="red")
        self.loupe.create_line(0, cy, size*zoom, cy, fill="red")

    # ---------- Series management ----------
    def _add_series_from_click(self, xpx: int, ypx: int):
        self._add_series_from_seed(xpx, ypx, seed_bbox_px=None)

    def _add_scatter_series_from_bbox(self, bbox: Tuple[int, int, int, int]):
        x0, y0, x1, y1 = bbox
        cx = int(round((x0 + x1) / 2.0))
        cy = int(round((y0 + y1) / 2.0))
        self._add_series_from_seed(cx, cy, seed_bbox_px=bbox)

    def _add_series_from_seed(self, xpx: int, ypx: int, *, seed_bbox_px: Optional[Tuple[int, int, int, int]]):
        require_cv2()
        if seed_bbox_px is not None:
            bx0, by0, bx1, by1 = seed_bbox_px
            bx0 = max(0, min(self._iw - 1, int(bx0)))
            bx1 = max(0, min(self._iw - 1, int(bx1)))
            by0 = max(0, min(self._ih - 1, int(by0)))
            by1 = max(0, min(self._ih - 1, int(by1)))
            if bx1 < bx0:
                bx0, bx1 = bx1, bx0
            if by1 < by0:
                by0, by1 = by1, by0
            patch = self._bgr[by0:by1+1, bx0:bx1+1, :]
            if patch.size:
                flat = patch.reshape(-1, 3).astype(np.float32)
                med = np.median(flat, axis=0)
                d = np.linalg.norm(flat - med, axis=1)
                idx = int(np.argmax(d))
                b, g, r = flat[idx].tolist()
                target = (int(b), int(g), int(r))
            else:
                b, g, r = self._bgr[ypx, xpx].tolist()
                target = (int(b), int(g), int(r))
                patch = self._bgr[max(0,ypx-3):ypx+4, max(0,xpx-3):xpx+4, :]
        else:
            b, g, r = self._bgr[ypx, xpx].tolist()
            target = (int(b), int(g), int(r))
            patch = self._bgr[max(0,ypx-3):ypx+4, max(0,xpx-3):xpx+4, :]

        print("picked BGR:", target, "patch median:", np.median(patch.reshape(-1,3), axis=0))

        sid = self._next_series_id
        self._next_series_id += 1

        name = f"Series {sid}"
        mode = self.series_mode.get()
        calibration = self._make_series_calibration_from_ui()

        s = Series(
            id=sid,
            name=name,
            color_bgr=target,
            chart_kind=mode,
            stacked=bool(self.var_stacked.get()),
            stride_mode=str(self._stride_mode_for_calibration(calibration, mode)),
            prefer_outline=bool(self.var_prefer_outline.get()),
            calibration=calibration,
        )
        s.seed_px = (xpx, ypx)
        if seed_bbox_px is not None:
            s.seed_bbox_px = seed_bbox_px
            tol = int(self.var_tol.get())
            mask = color_distance_mask(patch, target, tol).astype(np.uint8)
            nz = np.argwhere(mask > 0)
            if nz.size:
                y_min = int(nz[:, 0].min())
                y_max = int(nz[:, 0].max())
                x_min = int(nz[:, 1].min())
                x_max = int(nz[:, 1].max())
                s.seed_marker_bbox_px = (
                    int(bx0 + x_min),
                    int(by0 + y_min),
                    int(bx0 + x_max),
                    int(by0 + y_max),
                )
        self.series.append(s)
        self._insert_tree_row(s)

        # Extract immediately
        try:
            self._extract_series(s)
        except Exception as e:
            messagebox.showerror("Series extraction failed", str(e))

        self._select_series(sid)
        self._redraw_overlay()

    def _extract_series(self, s: Series):
        """
        Populate s.px_points, s.points, s.point_enabled based on s.chart_kind.

        - line: continuity-first line tracking sampled to an aligned X grid
        - scatter: blob/marker detection
        - column/area: boundary read (top/bottom vs baseline) sampled by stride mode
        - bar: boundary read (left/right vs baseline) along categorical Y centers
        """
        roi = s.calibration.roi_px
        tol = int(self.var_tol.get())
        cal = self._build_calibration(s.calibration)

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
                    template_match_thresh=float(s.calibration.scatter_match_thresh),
                )
                s.px_points = pts_px
                s.points = [(float(cal.x_px_to_data(x)), float(cal.y_px_to_data(y))) for (x, y) in pts_px]
                s.point_enabled = [True] * len(s.points)
                self._update_tree_row(s)
                return

            if sample_mode == "fixed_x":
                x0_val = self._require_value(s.calibration.x0_val, "x0", cal=s.calibration)
                roi_xmin_px, _, roi_xmax_px, _ = roi
                xmin = float(cal.x_px_to_data(roi_xmin_px))
                xmax = float(cal.x_px_to_data(roi_xmax_px))
                if xmin > xmax:
                    xmin, xmax = xmax, xmin

                step = float(s.calibration.x_step)
                x_grid: List[float] = []
                xpx_grid: List[int] = []
                if s.calibration.x_scale == AxisScale.CATEGORICAL.value:
                    labels = self._parse_categories(s.calibration.categories)
                    if labels:
                        centers = self._category_centers_px(len(labels), axis="x", axis_px=s.calibration.x_axis_px)
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
                    template_match_thresh=float(s.calibration.scatter_match_thresh),
                )

                y_data_raw: List[Optional[float]] = []
                for ypx in ypx_raw:
                    y_data_raw.append(None if ypx is None else float(cal.y_px_to_data(int(ypx))))

                y0_val = self._require_value(s.calibration.y0_val, "y0", cal=s.calibration)
                y1_val = self._require_value(s.calibration.y1_val, "y1", cal=s.calibration)
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
                self._update_tree_row(s)
                return

            if sample_mode == "fixed_y":
                y0_val = self._require_value(s.calibration.y0_val, "y0", cal=s.calibration)
                roi_ymin_px, roi_ymax_px = roi[1], roi[3]
                ymin = float(cal.y_px_to_data(roi_ymin_px))
                ymax = float(cal.y_px_to_data(roi_ymax_px))
                if ymin > ymax:
                    ymin, ymax = ymax, ymin

                step = float(s.calibration.y_step)
                y_grid: List[float] = []
                ypx_grid: List[int] = []
                if s.calibration.y_scale == AxisScale.CATEGORICAL.value:
                    labels = self._parse_categories(s.calibration.categories)
                    if labels:
                        centers = self._category_centers_px(len(labels), axis="y", axis_px=s.calibration.y_axis_px)
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
                    template_match_thresh=float(s.calibration.scatter_match_thresh),
                )

                x_data_raw: List[Optional[float]] = []
                for xpx in xpx_raw:
                    x_data_raw.append(None if xpx is None else float(cal.x_px_to_data(int(xpx))))

                x0_val = self._require_value(s.calibration.x0_val, "x0", cal=s.calibration)
                x1_val = self._require_value(s.calibration.x1_val, "x1", cal=s.calibration)
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
                self._update_tree_row(s)
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
            x0_val = self._require_value(s.calibration.x0_val, "x0", cal=s.calibration)
            roi_xmin_px, _, roi_xmax_px, _ = roi
            xmin = float(cal.x_px_to_data(roi_xmin_px))
            xmax = float(cal.x_px_to_data(roi_xmax_px))
            if xmin > xmax:
                xmin, xmax = xmax, xmin

            step = float(s.calibration.x_step)
            x_grid: List[float] = []
            xpx_grid: List[int] = []
            if stride == "categorical":
                labels = self._parse_categories(s.calibration.categories)
                if labels:
                    centers = self._category_centers_px(len(labels), axis="x", axis_px=s.calibration.x_axis_px)
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

            y_data_raw: List[Optional[float]] = []
            for ypx in ypx_raw:
                y_data_raw.append(None if ypx is None else float(cal.y_px_to_data(int(ypx))))

            y0_val = self._require_value(s.calibration.y0_val, "y0", cal=s.calibration)
            y1_val = self._require_value(s.calibration.y1_val, "y1", cal=s.calibration)
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
            self._update_tree_row(s)
            return

        # ---------------- Column / Area (vertical) ----------------
        if kind in ("column", "area"):
            # Determine sampling xpx_grid
            if stride == "categorical":
                labels = self._parse_categories(s.calibration.categories)
                if labels:
                    centers = self._category_centers_px(len(labels), axis="x", axis_px=s.calibration.x_axis_px)
                    offset = 0.0
                    if kind == "column" and s.seed_px is not None:
                        offset = self._category_offset_px(
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
                x0_val = self._require_value(s.calibration.x0_val, "x0", cal=s.calibration)
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
                y0_val = self._require_value(s.calibration.y0_val, "y0", cal=s.calibration)
                y1_val = self._require_value(s.calibration.y1_val, "y1", cal=s.calibration)
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

            self._update_tree_row(s)
            return

        # ---------------- Bar (horizontal) ----------------
        if kind == "bar":
            # For bars, categorical is the dominant case (categories along Y).
            labels = self._parse_categories(s.calibration.categories)
            if labels:
                centers = self._category_centers_px(len(labels), axis="y", axis_px=s.calibration.y_axis_px)
                offset = 0.0
                if s.seed_px is not None:
                    offset = self._category_offset_px(
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
            self._update_tree_row(s)
            return

        raise ValueError(f"Unsupported chart kind: {kind}")


    def _require_value(self, s: str, label: str, *, cal: Optional[SeriesCalibration] = None) -> float:
        s = (s or "").strip()
        if not s:
            raise ValueError(f"Missing {label} value. Enter it in the Calibration panel.")
        chart_cal = self._build_calibration(cal)
        if label.startswith("x"):
            return float(chart_cal.parse_x_value(s))
        if label.startswith("y"):
            return float(chart_cal.parse_y_value(s))
        return float(s)

    def _build_calibration(self, cal: Optional[SeriesCalibration] = None) -> ChartCalibration:
        # Use ROI bounds if axis pixels not set
        if cal is None:
            x0, y0, x1, y1 = self._roi_px()
            x0_px, x1_px = self._x_axis_px()
            y0_px, y1_px = self._y_axis_px()
            xs = AxisScale(self.x_scale.get())
            ys = AxisScale(self.y_scale.get())
            date_fmt = self.date_fmt.get().strip() or "%Y"
            x0v_str = (self.var_x0_val.get() or "").strip()
            x1v_str = (self.var_x1_val.get() or "").strip()
            y0v_str = (self.var_y0_val.get() or "").strip()
            y1v_str = (self.var_y1_val.get() or "").strip()
        else:
            x0, y0, x1, y1 = cal.roi_px
            x0_px, x1_px = cal.x_axis_px
            y0_px, y1_px = cal.y_axis_px
            xs = AxisScale(cal.x_scale)
            ys = AxisScale(cal.y_scale)
            date_fmt = (cal.date_fmt or "").strip() or "%Y"
            x0v_str = (cal.x0_val or "").strip()
            x1v_str = (cal.x1_val or "").strip()
            y0v_str = (cal.y0_val or "").strip()
            y1v_str = (cal.y1_val or "").strip()

        # values may be blank; use placeholders 0..1 to allow editing/overlay
        def parse_float_or_default(v: str, d: float) -> float:
            v = (v or "").strip()
            if not v:
                return d
            if xs == AxisScale.DATE:
                return float(_parse_date_safe(v, date_fmt))
            return float(v)

        import math
        # Parse X values
        if xs == AxisScale.DATE:
            x0v = _parse_date_safe(x0v_str or "2000", date_fmt)
            x1v = _parse_date_safe(x1v_str or "2001", date_fmt)
        else:
            x0v = float(x0v_str) if x0v_str else 0.0
            x1v = float(x1v_str) if x1v_str else 1.0

        if ys == AxisScale.DATE:
            y0v = _parse_date_safe(y0v_str or "2000", date_fmt)
            y1v = _parse_date_safe(y1v_str or "2001", date_fmt)
        else:
            y0v = float(y0v_str) if y0v_str else 0.0
            y1v = float(y1v_str) if y1v_str else 1.0

        xcal = AxisCalibration(p0=float(x0_px), p1=float(x1_px), v0=float(x0v), v1=float(x1v), scale=xs)
        ycal = AxisCalibration(p0=float(y0_px), p1=float(y1_px), v0=float(y0v), v1=float(y1v), scale=ys)
        return ChartCalibration(x=xcal, y=ycal, x_date_format=date_fmt, y_date_format=date_fmt)

    # ---------- Tree interactions ----------
    def _insert_tree_row(self, s: Series):
        self.tree.insert(
            "",
            "end",
            iid=str(s.id),
            values=("Y" if s.enabled else "N", s.name, s.calibration.name, len(s.points)),
        )

    def _update_tree_row(self, s: Series):
        if self.tree.exists(str(s.id)):
            self.tree.item(
                str(s.id),
                values=("Y" if s.enabled else "N", s.name, s.calibration.name, len(s.points)),
            )

    def _on_tree_select(self, _evt=None):
        sel = self.tree.selection()
        if not sel:
            self._active_series_id = None
        else:
            self._active_series_id = int(sel[0])
            if self.tool_mode.get() == "editseries":
                s = self._get_series(self._active_series_id)
                if s is not None:
                    self._set_ui_mode_from_series(s)
                    self._apply_series_calibration_to_ui(s.calibration)
        self._redraw_overlay()

    def _select_series(self, sid: int):
        self.tree.selection_set(str(sid))
        self.tree.see(str(sid))
        self._active_series_id = sid

    def _get_series(self, sid: int) -> Optional[Series]:
        for s in self.series:
            if s.id == sid:
                return s
        return None

    def _toggle_series_enabled(self):
        if self._active_series_id is None:
            return
        s = self._get_series(self._active_series_id)
        if not s:
            return
        s.enabled = not s.enabled
        self._update_tree_row(s)
        self._redraw_overlay()

    def _delete_series(self):
        if self._active_series_id is None:
            return
        sid = self._active_series_id
        self.series = [s for s in self.series if s.id != sid]
        if self.tree.exists(str(sid)):
            self.tree.delete(str(sid))
        self._active_series_id = None
        self._redraw_overlay()

    def _on_tree_double_click(self, event):
        # rename series by double-clicking name cell
        region = self.tree.identify("region", event.x, event.y)
        if region != "cell":
            return
        row = self.tree.identify_row(event.y)
        col = self.tree.identify_column(event.x)
        if not row or col != "#2":
            return
        sid = int(row)
        s = self._get_series(sid)
        if not s:
            return

        new = simpledialog.askstring("Rename series", "Series name:", initialvalue=s.name, parent=self)
        if new is None:
            return
        new = new.strip()
        if not new:
            return
        s.name = new
        self._update_tree_row(s)

    # ---------- Export ----------
    def _apply_stacked_deltas(self, ordered: List[Series]) -> List[Series]:
        if not any(getattr(s, "stacked", False) for s in ordered):
            return ordered

        prev_by_cal: dict[tuple, Series] = {}
        deltas: List[Series] = []
        for s in ordered:
            if not getattr(s, "stacked", False):
                deltas.append(s)
                continue

            key = self._calibration_key_from_series(s)
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
        labels = self._parse_categories(non_scatter[0].calibration.categories)
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
                centers_px = self._category_centers_px(count, axis=axis, axis_px=axis_px)

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
        cal = self._build_calibration(s.calibration)
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
            messagebox.showinfo("Append CSV", "No series to export yet. Use 'Add series' first.")
            return

        enabled = [s for s in self.series if s.enabled]
        if not enabled:
            return
        selected = None
        if self._active_series_id is not None:
            selected = self._get_series(self._active_series_id)
        if selected is None or selected not in enabled:
            selected = enabled[0]
        x_formatter = self._x_formatter_for_series(selected) if selected else None

        # If every enabled series is scatter, export long; otherwise export wide.
        if all(getattr(s, "chart_kind", s.mode) == "scatter" for s in enabled):
            txt = long_csv_string(enabled, x_formatter_by_series=self._format_x_for_series)
        else:
            x_grid, ser = self._prepare_wide_export(enabled)
            if not x_grid or not ser:
                messagebox.showinfo("Append CSV", "Nothing to export (no valid points).")
                return
            if x_grid and isinstance(x_grid[0], str):
                x_formatter = None
            txt = wide_csv_string(x_grid, ser, x_formatter=x_formatter)

        self._on_append_text(txt)

    def _export_csv(self):
        if not self.series:
            messagebox.showinfo("Export CSV", "No series to export yet. Use 'Add series' first.")
            return

        enabled = [s for s in self.series if s.enabled]
        if not enabled:
            messagebox.showinfo("Export CSV", "All series are disabled.")
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
            selected = self._get_series(self._active_series_id)
        if selected is None or selected not in enabled:
            selected = enabled[0]
        x_formatter = self._x_formatter_for_series(selected) if selected else None

        if all(getattr(s, "chart_kind", s.mode) == "scatter" for s in enabled):
            write_long_csv(path, enabled, x_formatter_by_series=self._format_x_for_series)
        else:
            x_grid, ser = self._prepare_wide_export(enabled)
            if not x_grid or not ser:
                messagebox.showinfo("Export CSV", "Nothing to export (no valid points).")
                return
            if x_grid and isinstance(x_grid[0], str):
                x_formatter = None
            write_wide_csv(path, x_grid, ser, x_formatter=x_formatter)

        messagebox.showinfo("Export CSV", f"Saved:\n{path}")

def _parse_date_safe(s: str, fmt: str) -> float:
    from datetime import datetime, timezone
    s = (s or "").strip()
    if not s:
        s = "2000"
    dt = datetime.strptime(s, fmt)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()

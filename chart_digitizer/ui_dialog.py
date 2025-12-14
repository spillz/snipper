
from __future__ import annotations

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from typing import Callable, Optional, Tuple, List

import numpy as np
from PIL import Image, ImageTk

from .cv_utils import require_cv2, pil_to_bgr
from .model import ChartState, Series
from .calibration import AxisScale, AxisCalibration, ChartCalibration
from .extract import build_x_grid, build_x_grid_aligned, extract_line_series, extract_scatter_series, enforce_line_grid
from .export_csv import wide_csv_string, long_csv_string, write_wide_csv, write_long_csv


class ChartDigitizerDialog(tk.Toplevel):
    def __init__(self, parent: tk.Tk, *, image: Image.Image, on_append_text: Callable[[str], None]):
        super().__init__(parent)
        self.title("Chart → CSV (V4)")
        self.geometry("1180x760")
        self.transient(parent)  # modeless: no grab_set
        self._on_append_text = on_append_text

        self._pil = image.convert("RGB")
        self._bgr = pil_to_bgr(self._pil)
        self._iw, self._ih = self._pil.size

        self.state = ChartState(xmin_px=0, ymin_px=0, xmax_px=self._iw, ymax_px=self._ih)
        self.series: List[Series] = []
        self._next_series_id = 1

        # Tool mode
        self.tool_mode = tk.StringVar(value="roi")  # edit|roi|xaxis|yaxis|addseries
        self.chart_mode = tk.StringVar(value="line")  # line|scatter

        # Axis scale types
        self.x_scale = tk.StringVar(value=AxisScale.LINEAR.value)
        self.y_scale = tk.StringVar(value=AxisScale.LINEAR.value)
        self.date_fmt = tk.StringVar(value="%Y")

        # Axis value entries
        self.var_x0_val = tk.StringVar(value="")
        self.var_x1_val = tk.StringVar(value="")
        self.var_y0_val = tk.StringVar(value="")
        self.var_y1_val = tk.StringVar(value="")

        self.var_x_step = tk.DoubleVar(value=1.0)
        self.var_tol = tk.IntVar(value=30)

        # editing / selection
        self._active_series_id: Optional[int] = None
        self._drag_idx: Optional[int] = None  # point index in active series
        self._edit_radius = 8

        # axis click staging
        self._pending_axis: Optional[str] = None  # 'x0','x1','y0','y1' progress

        self._build_ui()
        self._render_image()
        self._update_tip()

    # ---------- UI ----------
    def _build_ui(self):
        root = ttk.Frame(self, padding=8)
        root.pack(fill="both", expand=True)

        left = ttk.Frame(root)
        left.pack(side="left", fill="both", expand=True)

        right = ttk.Frame(root, width=340)
        right.pack(side="right", fill="y")

        # Toolbar
        bar = ttk.Frame(left)
        bar.pack(side="top", fill="x")

        ttk.Label(bar, text="Tool:").pack(side="left")
        for lbl, val in [("Set Region","roi"), ("Set X ticks","xaxis"), ("Set Y ticks","yaxis"), ("Add series","addseries"),("Edit series","editseries")]:
            ttk.Radiobutton(bar, text=lbl, value=val, variable=self.tool_mode, command=self._on_tool_change).pack(side="left", padx=(8,0))

        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=10)
        ttk.Label(bar, text="Chart type:").pack(side="left")
        ttk.Radiobutton(bar, text="Line", value="line", variable=self.chart_mode, command=self._on_chart_mode_change).pack(side="left", padx=(6,0))
        ttk.Radiobutton(bar, text="Scatter", value="scatter", variable=self.chart_mode, command=self._on_chart_mode_change).pack(side="left", padx=(6,0))

        # Loupe and Tip
        loupe_frm = ttk.Frame(left)
        loupe_frm.pack(side="top", fill="x", pady=(8,0))
        ttk.Label(loupe_frm, text="Loupe:").pack(side="left")
        self.loupe = tk.Canvas(loupe_frm, width=120, height=120, highlightthickness=1, highlightbackground="#666")
        self.loupe.pack(side="left", padx=(6,0))
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
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<Button-3>", self._on_right_click)

        # Right panel: axis config
        ax = ttk.LabelFrame(right, text="Calibration", padding=8)
        ax.pack(side="top", fill="x")

        grid = ttk.Frame(ax)
        grid.pack(fill="x")

        ttk.Label(grid, text="X scale").grid(row=0, column=0, sticky="w")
        ttk.Combobox(grid, textvariable=self.x_scale, state="readonly", width=10,
                     values=[AxisScale.LINEAR.value, AxisScale.LOG10.value, AxisScale.DATE.value]).grid(row=0, column=1, sticky="w", padx=(6,0))
        ttk.Label(grid, text="Date fmt").grid(row=0, column=2, sticky="w", padx=(8,0))
        ttk.Entry(grid, textvariable=self.date_fmt, width=10).grid(row=0, column=3, sticky="w")

        ttk.Label(grid, text="Y scale").grid(row=1, column=0, sticky="w", pady=(6,0))
        ttk.Combobox(grid, textvariable=self.y_scale, state="readonly", width=10,
                     values=[AxisScale.LINEAR.value, AxisScale.LOG10.value]).grid(row=1, column=1, sticky="w", padx=(6,0), pady=(6,0))

        ttk.Label(grid, text="x0").grid(row=2, column=0, sticky="w", pady=(8,0))
        ttk.Entry(grid, textvariable=self.var_x0_val, width=12).grid(row=2, column=1, sticky="w", pady=(8,0))
        ttk.Label(grid, text="x1").grid(row=2, column=2, sticky="w", padx=(8,0), pady=(8,0))
        ttk.Entry(grid, textvariable=self.var_x1_val, width=12).grid(row=2, column=3, sticky="w", pady=(8,0))

        ttk.Label(grid, text="y0").grid(row=3, column=0, sticky="w", pady=(6,0))
        ttk.Entry(grid, textvariable=self.var_y0_val, width=12).grid(row=3, column=1, sticky="w", pady=(6,0))
        ttk.Label(grid, text="y1").grid(row=3, column=2, sticky="w", padx=(8,0), pady=(6,0))
        ttk.Entry(grid, textvariable=self.var_y1_val, width=12).grid(row=3, column=3, sticky="w", pady=(6,0))

        ttk.Label(ax, text="Axis pixels: click in 'Set X axis'/'Set Y axis' to set x0/x1/y0/y1 pixels (defaults to ROI bounds).").pack(fill="x", pady=(8,0))

        # Output
        out = ttk.LabelFrame(right, text="Output", padding=8)
        out.pack(side="top", fill="x", pady=(8,0))

        row = ttk.Frame(out)
        row.pack(fill="x")
        ttk.Label(row, text="X step (line):").pack(side="left")
        ttk.Entry(row, textvariable=self.var_x_step, width=10).pack(side="left", padx=(6,0))
        ttk.Label(row, text="Color tol:").pack(side="left", padx=(10,0))
        ttk.Entry(row, textvariable=self.var_tol, width=6).pack(side="left", padx=(6,0))

        # Series list
        lst = ttk.LabelFrame(right, text="Series", padding=8)
        lst.pack(side="top", fill="both", expand=True, pady=(8,0))

        self.tree = ttk.Treeview(lst, columns=("enabled","name","n"), show="headings", selectmode="browse", height=12)
        self.tree.heading("enabled", text="On")
        self.tree.heading("name", text="Name")
        self.tree.heading("n", text="Pts")
        self.tree.column("enabled", width=34, anchor="center")
        self.tree.column("name", width=200, anchor="w")
        self.tree.column("n", width=60, anchor="e")
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

    def _update_tip(self):
        mode = self.tool_mode.get()
        chart_mode = self.chart_mode.get()  # line|scatter

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
            if chart_mode == "line":
                msg = (
                    "Add series (Line): Click directly on a line in the chart to generate the series data. "
                    "The tool tracks the line from the clicked point across the rectangular region and samples it onto the X grid. "
                    "X step controls the output sampling grid in steps left and right from x0; missing samples are interpolated/flatlined. "
                    "The color value set the threshold for matching the same color value. "
                    "For example, if x-axis units are years, then X step = 0.25 will extract values at quarterly increments when you add series."
                )
            else:
                msg = (
                    "Add series (Scatter): Click a marker color to pick the series. "
                    "The tool detects colored marker blobs inside the region and exports their (x,y) coordinates. "
                    "X step is ignored in scatter mode."
                )
        elif mode == "editseries":
            if chart_mode == "line":
                msg = (
                    "Edit series (Line): Select a series in the table, then drag points vertically to correct Y values "
                    "(X is fixed to the sampling grid). Right-click a point to toggle NA/disabled. "
                    "Edits affect exported CSV values."
                )
            else:
                msg = (
                    "Edit series (Scatter): Select a series in the table. Right-click toggles a point NA/disabled. "
                    "Dragging is currently limited (scatter editing can be expanded later). "
                    "Edits affect exported CSV values."
                )
        else:
            # fallback / none
            msg = (
                "Select a tool mode above. Region controls what pixels are scanned; X/Y ticks control how pixels map to data units."
            )

        self.tip_var.set(msg)

    def _on_chart_mode_change(self):
        self._update_tip()

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
        self._update_tip()
        self._redraw_overlay()

    # ---------- Image rendering ----------
    def _render_image(self):
        self.canvas.delete("all")
        # Fit image into canvas (no scaling initially; enable simple scaling)
        self.canvas.update_idletasks()
        cw = max(10, self.canvas.winfo_width())
        ch = max(10, self.canvas.winfo_height())

        # scale factor
        sx = cw / self._iw
        sy = ch / self._ih
        self._scale = min(1.0, sx, sy)  # don't upscale beyond 1 for now
        disp_w = int(self._iw * self._scale)
        disp_h = int(self._ih * self._scale)

        self._offx = (cw - disp_w)//2
        self._offy = (ch - disp_h)//2

        disp = self._pil.resize((disp_w, disp_h), Image.NEAREST)
        self._photo = ImageTk.PhotoImage(disp)
        self.canvas.create_image(self._offx, self._offy, image=self._photo, anchor="nw", tags=("img",))

        self._redraw_overlay()

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
                if s.mode == "line" and len(pts) >= 2:
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

    def _on_drag(self, event):
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

            if s.mode == "line":
                # keep x fixed
                x_fixed = s.px_points[self._drag_idx][0]
                s.px_points[self._drag_idx] = (x_fixed, ypx)

                y_data = cal.y_px_to_data(ypx)
                x_data = s.points[self._drag_idx][0]
                s.points[self._drag_idx] = (x_data, float(y_data))

            else:  # scatter
                s.px_points[self._drag_idx] = (xpx, ypx)
                x_data = cal.x_px_to_data(xpx)
                y_data = cal.y_px_to_data(ypx)
                s.points[self._drag_idx] = (float(x_data), float(y_data))

            self._update_tree_row(s)
            self._redraw_overlay()

    def _on_release(self, event):
        if self.tool_mode.get() == "roi":
            self._roi_dragging = False
        self._drag_idx = None

    def _on_right_click(self, event):
        # Toggle point enable (NA) in none mode
        if self.tool_mode.get() != "editseries" or self._active_series_id is None:
            return
        xpx, ypx = self._to_image_px(event.x, event.y)
        s = self._get_series(self._active_series_id)
        if not s or not s.px_points:
            return
        idx = self._nearest_point_index(s.px_points, xpx, ypx, self._edit_radius)
        if idx is None:
            return
        if not s.point_enabled:
            s.point_enabled = [True]*len(s.points)
        s.point_enabled[idx] = not s.point_enabled[idx]
        self._redraw_overlay()

    def _start_point_drag(self, xpx: int, ypx: int):
        if self._active_series_id is None:
            return
        s = self._get_series(self._active_series_id)
        if not s or not s.px_points:
            return
        idx = self._nearest_point_index(s.px_points, xpx, ypx, self._edit_radius)
        if idx is None:
            return
        self._drag_idx = idx

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
        require_cv2()

        b, g, r = self._bgr[ypx, xpx].tolist()
        target = (int(b), int(g), int(r))

        patch = self._bgr[max(0,ypx-3):ypx+4, max(0,xpx-3):xpx+4, :]
        print("picked BGR:", target, "patch median:", np.median(patch.reshape(-1,3), axis=0))

        sid = self._next_series_id
        self._next_series_id += 1

        name = f"Series {sid}"
        mode = self.chart_mode.get()

        s = Series(id=sid, name=name, color_bgr=target, mode=mode)
        s.seed_px = (xpx, ypx)
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
        roi = self._roi_px()
        tol = int(self.var_tol.get())
        s.color_bgr = s.color_bgr  # already set

        cal = self._build_calibration()

        if s.mode == "scatter":
            pts_px = extract_scatter_series(self._bgr, roi, s.color_bgr, tol)
            s.px_points = pts_px
            s.points = [(cal.x_px_to_data(x), cal.y_px_to_data(y)) for (x,y) in pts_px]
            s.point_enabled = [True]*len(s.points)
            self._update_tree_row(s)
            return

        # line mode: domain comes from ROI bounds, not x0/x1 anchor values
        if False:
            roi_xmin_px, _, roi_xmax_px, _ = self._roi_px()

            xmin = float(cal.x_px_to_data(roi_xmin_px))
            xmax = float(cal.x_px_to_data(roi_xmax_px))

            if xmin > xmax:
                xmin, xmax = xmax, xmin

            step = float(self.var_x_step.get())
            x_grid = build_x_grid(xmin, xmax, step)

        x0_val = self._require_value(self.var_x0_val.get(), "x0")
        x1_val = self._require_value(self.var_x1_val.get(), "x1")
        # xmin = min(x0_val, x1_val)
        # xmax = max(x0_val, x1_val)

        roi_xmin_px, _, roi_xmax_px, _ = self._roi_px()

        xmin = float(cal.x_px_to_data(roi_xmin_px))
        xmax = float(cal.x_px_to_data(roi_xmax_px))
        if xmin > xmax:
            xmin, xmax = xmax, xmin


        step = float(self.var_x_step.get())
        x_grid = build_x_grid_aligned(xmin, xmax, step, anchor=x0_val)
        

        # Convert to pixel x positions
        xpx_grid = [int(round(cal.x_data_to_px(x))) for x in x_grid]

        # Convert to pixel x positions
        xpx_grid = [int(round(cal.x_data_to_px(x))) for x in x_grid]

        px_pts, ypx_raw = extract_line_series(
            self._bgr, roi, s.color_bgr, tol, xpx_grid,
            seed_px=s.seed_px,
        )

        # Convert ypx_raw to data y list aligned to x_grid
        y_data_raw: List[Optional[float]] = []
        for ypx in ypx_raw:
            if ypx is None:
                y_data_raw.append(None)
            else:
                y_data_raw.append(float(cal.y_px_to_data(ypx)))

        # fallback y midpoint
        y0_val = self._require_value(self.var_y0_val.get(), "y0")
        y1_val = self._require_value(self.var_y1_val.get(), "y1")
        fallback_y = 0.5 * (y0_val + y1_val)

        y_filled = enforce_line_grid(x_grid, y_data_raw, fallback_y=fallback_y)

        # Ensure px_points aligned to x_grid (even if missing, place at midline)
        px_points: List[Tuple[int,int]] = []
        for i, xpx in enumerate(xpx_grid):
            if ypx_raw[i] is None:
                ypx = int(round(cal.y_data_to_px(y_filled[i])))
            else:
                ypx = int(ypx_raw[i])
            px_points.append((xpx, ypx))

        s.px_points = px_points
        s.points = [(x_grid[i], y_filled[i]) for i in range(len(x_grid))]
        s.point_enabled = [True]*len(s.points)
        self._update_tree_row(s)

    def _require_value(self, s: str, label: str) -> float:
        s = (s or "").strip()
        if not s:
            raise ValueError(f"Missing {label} value. Enter it in the Calibration panel.")
        cal = self._build_calibration()
        if label.startswith("x"):
            return float(cal.parse_x_value(s))
        return float(s)

    def _build_calibration(self) -> ChartCalibration:
        # Use ROI bounds if axis pixels not set
        x0,y0,x1,y1 = self._roi_px()
        x0_px, x1_px = self._x_axis_px()
        y0_px, y1_px = self._y_axis_px()

        xs = AxisScale(self.x_scale.get())
        ys = AxisScale(self.y_scale.get())
        date_fmt = self.date_fmt.get().strip() or "%Y"

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
        x0v_str = self.var_x0_val.get().strip()
        x1v_str = self.var_x1_val.get().strip()
        if xs == AxisScale.DATE:
            x0v = _parse_date_safe(x0v_str or "2000", date_fmt)
            x1v = _parse_date_safe(x1v_str or "2001", date_fmt)
        else:
            x0v = float(x0v_str) if x0v_str else 0.0
            x1v = float(x1v_str) if x1v_str else 1.0

        y0v_str = self.var_y0_val.get().strip()
        y1v_str = self.var_y1_val.get().strip()
        y0v = float(y0v_str) if y0v_str else 0.0
        y1v = float(y1v_str) if y1v_str else 1.0

        xcal = AxisCalibration(p0=float(x0_px), p1=float(x1_px), v0=float(x0v), v1=float(x1v), scale=xs)
        ycal = AxisCalibration(p0=float(y0_px), p1=float(y1_px), v0=float(y0v), v1=float(y1v), scale=ys)
        return ChartCalibration(x=xcal, y=ycal, x_date_format=date_fmt)

    # ---------- Tree interactions ----------
    def _insert_tree_row(self, s: Series):
        self.tree.insert("", "end", iid=str(s.id), values=("Y" if s.enabled else "N", s.name, len(s.points)))

    def _update_tree_row(self, s: Series):
        if self.tree.exists(str(s.id)):
            self.tree.item(str(s.id), values=("Y" if s.enabled else "N", s.name, len(s.points)))

    def _on_tree_select(self, _evt=None):
        sel = self.tree.selection()
        if not sel:
            self._active_series_id = None
        else:
            self._active_series_id = int(sel[0])
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
    def _append_csv(self):
        if not self.series:
            messagebox.showinfo("Append CSV", "No series to export yet. Use 'Add series' first.")
            return
        mode = self.chart_mode.get()
        enabled = [s for s in self.series if s.enabled]

        if mode == "line":
            # assume all line series use same x_grid (from first)
            if not enabled:
                return
            x_grid = [x for (x, _) in enabled[0].points]
            txt = wide_csv_string(x_grid, enabled)
        else:
            txt = long_csv_string(enabled)

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
        if self.chart_mode.get() == "line":
            x_grid = [x for (x, _) in enabled[0].points]
            write_wide_csv(path, x_grid, enabled)
        else:
            write_long_csv(path, enabled)
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

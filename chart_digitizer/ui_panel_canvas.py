from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Optional, Tuple

import bisect
import numpy as np
from PIL import Image, ImageTk

from .cv_utils import require_cv2, color_distance_mask
from .data_model import Series


class CanvasPanel:
    def __init__(self, owner, parent: tk.Widget, *, actor) -> None:
        self.owner = owner
        self.actor = actor

        frame = ttk.Frame(parent)
        self.frame = frame
        frame.pack(side="top", fill="both", expand=True)

        loupe_frm = ttk.Frame(frame)
        loupe_frm.pack(side="top", fill="x", pady=(8, 0))
        ttk.Label(loupe_frm, text="Loupe:").pack(side="left")
        owner.loupe = tk.Canvas(loupe_frm, width=120, height=120, highlightthickness=1, highlightbackground="#666")
        owner.loupe.pack(side="left", padx=(6, 0))
        ttk.Checkbutton(
            loupe_frm,
            text="Fit",
            variable=owner.var_fit_image,
            command=owner._on_fit_toggle,
        ).pack(side="left", padx=(10, 0))
        owner.tip_var = tk.StringVar(value="")
        owner.tip_label = ttk.Label(loupe_frm, textvariable=owner.tip_var, wraplength=900, justify="left")
        owner.tip_label.pack(side="left", padx=(10, 0), fill="x", expand=True)

        owner.canvas = tk.Canvas(frame, background="#111", highlightthickness=1, highlightbackground="#333")
        owner.canvas.pack(side="bottom", fill="both", expand=True, pady=(8, 0))
        owner.canvas.bind("<Configure>", actor._on_canvas_configure)
        owner.canvas.bind("<Button-1>", actor._on_click)
        owner.canvas.bind("<B1-Motion>", actor._on_drag)
        owner.canvas.bind("<ButtonRelease-1>", actor._on_release)
        owner.canvas.bind("<Shift-ButtonPress-1>", actor._on_scatter_rubberband_press)
        owner.canvas.bind("<Shift-B1-Motion>", actor._on_scatter_rubberband_motion)
        owner.canvas.bind("<Shift-ButtonRelease-1>", actor._on_scatter_rubberband_release)
        owner.canvas.bind("<Control-Shift-ButtonPress-1>", actor._on_scatter_rubberband_press)
        owner.canvas.bind("<Control-Shift-B1-Motion>", actor._on_scatter_rubberband_motion)
        owner.canvas.bind("<Control-Shift-ButtonRelease-1>", actor._on_scatter_rubberband_release)
        owner.canvas.bind("<Motion>", actor._on_motion)
        owner.canvas.bind("<Button-3>", actor._on_right_click)
        owner.canvas.bind("<B3-Motion>", actor._on_right_drag)
        owner.canvas.bind("<ButtonRelease-3>", actor._on_right_release)
        owner.canvas.bind("<Control-ButtonPress-1>", actor._on_ctrl_toggle_press)
        owner.canvas.bind("<Control-B1-Motion>", actor._on_ctrl_toggle_drag)
        owner.canvas.bind("<Control-ButtonRelease-1>", actor._on_ctrl_toggle_release)
        owner.canvas.bind("<Control-ButtonPress-3>", actor._on_ctrl_toggle_press)
        owner.canvas.bind("<Control-B3-Motion>", actor._on_ctrl_toggle_drag)
        owner.canvas.bind("<Control-ButtonRelease-3>", actor._on_ctrl_toggle_release)
        owner.canvas.bind("<Shift-ButtonPress-3>", actor._on_mask_rubberband_press_right)
        owner.canvas.bind("<Shift-B3-Motion>", actor._on_mask_rubberband_motion)
        owner.canvas.bind("<Shift-ButtonRelease-3>", actor._on_mask_rubberband_release)
        owner.canvas.bind("<MouseWheel>", actor._on_mouse_wheel)
        owner.canvas.bind("<Shift-MouseWheel>", actor._on_shift_mouse_wheel)
        owner.canvas.bind("<Leave>", actor._on_canvas_leave)


class CanvasActor:
    def __init__(self, owner) -> None:
        object.__setattr__(self, "owner", owner)

    def __getattr__(self, name):
        return getattr(self.owner, name)

    def __setattr__(self, name, value) -> None:
        if name == "owner":
            object.__setattr__(self, name, value)
            return
        setattr(self.owner, name, value)

    def _on_canvas_configure(self, _evt=None):
        # Avoid thrashing when resizing: schedule a single re-render
        if getattr(self, "_render_after_id", None) is not None:
            try:
                self.after_cancel(self._render_after_id)
            except Exception:
                pass
        self._render_after_id = self.after(30, self._render_image)


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

        # mask overlay
        if self._active_series_id is not None:
            s = self.owner.series_actor._get_series(self._active_series_id)
            if s is not None:
                self._draw_mask_overlay(s)

        # # active series overlay
        # if self._active_series_id is not None:
        #     s = self.owner.series_actor._get_series(self._active_series_id)
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
            s = self.owner.series_actor._get_series(self._active_series_id)
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
            s = self.owner.series_actor._get_series(self._active_series_id)
            if s and (s.seed_px or s.extra_seeds_px):
                seeds = []
                if s.seed_px:
                    seeds.append((s.seed_px, "#00B4FF"))
                for ex in (s.extra_seeds_px or []):
                    seeds.append((ex, "#00E5FF"))
                marker_bboxes = getattr(s, "scatter_marker_bboxes_px", []) or []
                for i, bbox in enumerate(marker_bboxes):
                    color = "#00B4FF" if i == 0 else "#00E5FF"
                    mx0, my0, mx1, my1 = bbox
                    cx0, cy0 = self._to_canvas(int(mx0), int(my0))
                    cx1, cy1 = self._to_canvas(int(mx1), int(my1))
                    self.canvas.create_oval(cx0, cy0, cx1, cy1,
                                            outline="black", width=2, fill="",
                                            tags=("overlay","seed"))
                    self.canvas.create_oval(cx0+1, cy0+1, cx1-1, cy1-1,
                                            outline=color, width=2, fill="",
                                            tags=("overlay","seed"))

                for (sx, sy), color in seeds:
                    cx, cy = self._to_canvas(int(sx), int(sy))
                    r = 5
                    self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r,
                                            outline="black", width=2, fill="",
                                            tags=("overlay","seed"))
                    self.canvas.create_oval(cx-(r-1), cy-(r-1), cx+(r-1), cy+(r-1),
                                            outline=color, width=2, fill="",
                                            tags=("overlay","seed"))
        self.owner.calibrator._update_axis_pixels_label()


    def _draw_mask_overlay(self, s: Series) -> None:
        mask = getattr(s, "mask_bitmap", None)
        if mask is None:
            return
        if mask.shape[0] != self._ih or mask.shape[1] != self._iw:
            return
        alpha_val = 80
        rgb = (0, 200, 255) if not getattr(s, "mask_invert", False) else (255, 90, 90)
        alpha = (mask > 0).astype(np.uint8) * alpha_val
        overlay = Image.new("RGBA", (self._iw, self._ih), (*rgb, alpha_val))
        overlay.putalpha(Image.fromarray(alpha))
        disp_w = int(self._iw * self._scale)
        disp_h = int(self._ih * self._scale)
        disp = overlay.resize((disp_w, disp_h), Image.NEAREST)
        self._mask_photo = ImageTk.PhotoImage(disp)
        self.canvas.create_image(self._offx, self._offy, image=self._mask_photo, anchor="nw", tags=("overlay", "mask"))

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
            if self.tool_mode.get() == "mask":
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

        if mode == "mask":
            self._start_mask_draw(xpx, ypx, value=255)
            return

        # mode == none: edit points
        self._start_point_drag(xpx, ypx)



    def _start_point_drag(self, xpx: int, ypx: int):
        """Begin drag-editing a point in the active series (editseries mode only)."""
        if self.tool_mode.get() != "editseries":
            return
        if self._active_series_id is None:
            return

        s = self.owner.series_actor._get_series(self._active_series_id)
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
        if mode == "mask" and self._mask_drawing:
            self._apply_mask_brush(xpx, ypx, self._mask_draw_value)
            return
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
            s = self.owner.series_actor._get_series(self._active_series_id)
            if not s:
                return

            roi_x0, roi_y0, roi_x1, roi_y1 = self._roi_px()
            xpx = max(roi_x0, min(roi_x1, xpx))
            ypx = max(roi_y0, min(roi_y1, ypx))

            cal = self.owner.calibrator._build_calibration()

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

            self.owner.series_actor._update_tree_row(s)
            self._redraw_overlay()
            self._draw_loupe(xpx, ypx)

    def _on_release(self, event):
        if self._scatter_rb_active:
            return
        if self.tool_mode.get() == "mask":
            if self._mask_drawing:
                self._mask_drawing = False
                self._finish_mask_edit()
            return
        if self.tool_mode.get() == "roi":
            self._roi_dragging = False

        # If we were dragging a scatter point, resequence by ascending X on release.
        if self.tool_mode.get() == "editseries" and self._active_series_id is not None and self._drag_series_mode == "scatter":
            s = self.owner.series_actor._get_series(self._active_series_id)
            if s and s.points and s.px_points and len(s.points) == len(s.px_points):
                order = sorted(range(len(s.points)), key=lambda i: (s.points[i][0], s.points[i][1]))
                s.points = [s.points[i] for i in order]
                s.px_points = [s.px_points[i] for i in order]
                if s.point_enabled and len(s.point_enabled) == len(order):
                    s.point_enabled = [s.point_enabled[i] for i in order]
                self.owner.series_actor._update_tree_row(s)
                self._redraw_overlay()

        self._drag_idx = None
        self._drag_series_mode = None
        self._toggle_dragging = False
        self._toggle_seen.clear()


    def _on_scatter_rubberband_press(self, event):
        if self.tool_mode.get() == "mask":
            self._start_mask_rubberband(event, value=255)
            return
        if self.tool_mode.get() != "addseries" or self.series_mode.get() != "scatter":
            return
        xpx, ypx = self._to_image_px(event.x, event.y)
        self._scatter_rb_start = (xpx, ypx)
        self._scatter_rb_active = True
        self._scatter_rb_additive = bool(event.state & 0x0004)
        cx, cy = self._to_canvas(xpx, ypx)
        self._scatter_rb_id = self.canvas.create_rectangle(
            cx, cy, cx, cy, outline="#FFD166", width=1, tags=("overlay", "scatter_rb")
        )


    def _on_scatter_rubberband_motion(self, event):
        if self._mask_rb_active:
            self._update_mask_rubberband(event)
            return
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
        if self._mask_rb_active:
            self._finish_mask_rubberband(event)
            return
        if not self._scatter_rb_active or self._scatter_rb_start is None:
            return
        xpx, ypx = self._to_image_px(event.x, event.y)
        sx, sy = self._scatter_rb_start
        x0 = min(sx, xpx); x1 = max(sx, xpx)
        y0 = min(sy, ypx); y1 = max(sy, ypx)
        self._scatter_rb_active = False
        self._scatter_rb_start = None
        additive = self._scatter_rb_additive
        self._scatter_rb_additive = False
        if self._scatter_rb_id is not None:
            self.canvas.delete(self._scatter_rb_id)
            self._scatter_rb_id = None

        if (x1 - x0) < 3 or (y1 - y0) < 3:
            self._add_series_from_click(xpx, ypx)
            return

        if additive:
            self._add_scatter_template_from_bbox((x0, y0, x1, y1))
        else:
            self._add_scatter_series_from_bbox((x0, y0, x1, y1))


    def _on_mask_rubberband_press_right(self, event):
        if self.tool_mode.get() != "mask":
            return
        self._start_mask_rubberband(event, value=0)


    def _on_mask_rubberband_motion(self, event):
        self._on_scatter_rubberband_motion(event)


    def _on_mask_rubberband_release(self, event):
        self._on_scatter_rubberband_release(event)


    def _start_mask_rubberband(self, event, *, value: int) -> None:
        s = self.owner.series_actor._get_series(self._active_series_id) if self._active_series_id is not None else None
        if s is None:
            return
        xpx, ypx = self._to_image_px(event.x, event.y)
        self._mask_rb_start = (xpx, ypx)
        self._mask_rb_active = True
        self._mask_rb_value = int(value)
        cx, cy = self._to_canvas(xpx, ypx)
        self._mask_rb_id = self.canvas.create_rectangle(
            cx, cy, cx, cy, outline="#7AE7FF", width=1, tags=("overlay", "mask_rb")
        )


    def _update_mask_rubberband(self, event) -> None:
        if not self._mask_rb_active or self._mask_rb_start is None:
            return
        xpx, ypx = self._to_image_px(event.x, event.y)
        sx, sy = self._mask_rb_start
        x0 = min(sx, xpx); x1 = max(sx, xpx)
        y0 = min(sy, ypx); y1 = max(sy, ypx)
        cx0, cy0 = self._to_canvas(x0, y0)
        cx1, cy1 = self._to_canvas(x1, y1)
        if self._mask_rb_id is not None:
            self.canvas.coords(self._mask_rb_id, cx0, cy0, cx1, cy1)


    def _finish_mask_rubberband(self, event) -> None:
        if not self._mask_rb_active or self._mask_rb_start is None:
            return
        xpx, ypx = self._to_image_px(event.x, event.y)
        sx, sy = self._mask_rb_start
        x0 = min(sx, xpx); x1 = max(sx, xpx)
        y0 = min(sy, ypx); y1 = max(sy, ypx)
        self._mask_rb_active = False
        self._mask_rb_start = None
        if self._mask_rb_id is not None:
            self.canvas.delete(self._mask_rb_id)
            self._mask_rb_id = None

        if (x1 - x0) <= 1 and (y1 - y0) <= 1:
            self._apply_mask_brush(xpx, ypx, self._mask_rb_value)
        else:
            self._apply_mask_rect(x0, y0, x1, y1, self._mask_rb_value)
        self._finish_mask_edit()

    def _on_right_click(self, event):
        # Edit-series helpers:
        # - Right-click on point: toggle enabled/NA state
        # - Right-click away from points (scatter only): insert a new point at that location (sorted by X)
        xpx, ypx = self._to_image_px(event.x, event.y)
        if self.tool_mode.get() == "mask":
            self._start_mask_draw(xpx, ypx, value=0)
            return
        if self.tool_mode.get() != "editseries" or self._active_series_id is None:
            return

        s = self.owner.series_actor._get_series(self._active_series_id)
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

            cal = self.owner.calibrator._build_calibration()
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

            self.owner.series_actor._update_tree_row(s)
            self._redraw_overlay()
            return

        # Toggle enable on an existing point
        if not s.point_enabled:
            s.point_enabled = [True] * len(s.points)
        s.point_enabled[idx] = not s.point_enabled[idx]
        self._redraw_overlay()


    def _on_right_drag(self, event):
        if self.tool_mode.get() != "mask" or not self._mask_drawing:
            return
        xpx, ypx = self._to_image_px(event.x, event.y)
        self._apply_mask_brush(xpx, ypx, self._mask_draw_value)


    def _on_right_release(self, _event):
        if self.tool_mode.get() != "mask":
            return
        if self._mask_drawing:
            self._mask_drawing = False
            self._finish_mask_edit()


    def _on_mouse_wheel(self, event):
        if self.tool_mode.get() != "mask":
            return
        delta = 1 if event.delta > 0 else -1
        self._mask_pen_radius = max(1, min(80, int(self._mask_pen_radius) + delta))
        self._update_tip()


    def _on_shift_mouse_wheel(self, event):
        if self.tool_mode.get() != "mask":
            return
        if event.delta == 0:
            return
        self._mask_pen_shape = "square" if self._mask_pen_shape == "circle" else "circle"
        self._update_tip()


    def _start_mask_draw(self, xpx: int, ypx: int, *, value: int) -> None:
        s = self.owner.series_actor._get_series(self._active_series_id) if self._active_series_id is not None else None
        if s is None:
            return
        self._mask_drawing = True
        self._mask_draw_value = int(value)
        self._apply_mask_brush(xpx, ypx, self._mask_draw_value)


    def _apply_mask_brush(self, xpx: int, ypx: int, value: int) -> None:
        s = self.owner.series_actor._get_series(self._active_series_id) if self._active_series_id is not None else None
        if s is None:
            return
        mask = self._ensure_series_mask(s)
        r = max(1, int(self._mask_pen_radius))
        x0 = max(0, xpx - r)
        x1 = min(self._iw - 1, xpx + r)
        y0 = max(0, ypx - r)
        y1 = min(self._ih - 1, ypx + r)
        if self._mask_pen_shape == "square":
            mask[y0:y1 + 1, x0:x1 + 1] = int(value)
        else:
            ys, xs = np.ogrid[y0:y1 + 1, x0:x1 + 1]
            dist2 = (xs - xpx) ** 2 + (ys - ypx) ** 2
            region = mask[y0:y1 + 1, x0:x1 + 1]
            region[dist2 <= (r * r)] = int(value)
        self._redraw_overlay()


    def _apply_mask_rect(self, x0: int, y0: int, x1: int, y1: int, value: int) -> None:
        s = self.owner.series_actor._get_series(self._active_series_id) if self._active_series_id is not None else None
        if s is None:
            return
        mask = self._ensure_series_mask(s)
        x0 = max(0, min(self._iw - 1, int(x0)))
        x1 = max(0, min(self._iw - 1, int(x1)))
        y0 = max(0, min(self._ih - 1, int(y0)))
        y1 = max(0, min(self._ih - 1, int(y1)))
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        mask[y0:y1 + 1, x0:x1 + 1] = int(value)
        self._redraw_overlay()


    def _finish_mask_edit(self) -> None:
        s = self.owner.series_actor._get_series(self._active_series_id) if self._active_series_id is not None else None
        if s is None:
            return
        if self.var_auto_rerun.get():
            try:
                self.owner.extractor._extract_series(s)
            except Exception as e:
                self._show_error("Series extraction failed", str(e))
        self.owner.series_actor._update_tree_row(s)
        self._redraw_overlay()


    def _ensure_series_mask(self, s: Series) -> np.ndarray:
        mask = getattr(s, "mask_bitmap", None)
        if mask is None or mask.shape[0] != self._ih or mask.shape[1] != self._iw:
            mask = np.zeros((self._ih, self._iw), dtype=np.uint8)
            s.mask_bitmap = mask
        return mask


    def _mask_allows_point(self, s: Series, xpx: int, ypx: int) -> bool:
        mask = getattr(s, "mask_bitmap", None)
        if mask is None:
            return True
        if ypx < 0 or ypx >= mask.shape[0] or xpx < 0 or xpx >= mask.shape[1]:
            return False
        allowed = bool(mask[int(ypx), int(xpx)] > 0)
        if getattr(s, "mask_invert", False):
            allowed = not allowed
        return allowed


    def _apply_series_mask_to_roi(self, s: Series, roi: Tuple[int, int, int, int], mask: np.ndarray) -> np.ndarray:
        series_mask = getattr(s, "mask_bitmap", None)
        if series_mask is None:
            return mask
        x0, y0, x1, y1 = roi
        x0 = max(0, min(self._iw, int(x0)))
        x1 = max(0, min(self._iw, int(x1)))
        y0 = max(0, min(self._ih, int(y0)))
        y1 = max(0, min(self._ih, int(y1)))
        if x1 <= x0 or y1 <= y0:
            return mask
        sm = series_mask[y0:y1, x0:x1]
        if sm.shape != mask.shape:
            return mask
        if getattr(s, "mask_invert", False):
            allow = (sm == 0)
        else:
            allow = (sm > 0)
        masked = mask.copy()
        masked[~allow] = 0
        return masked


    def _sync_mask_controls(self, s: Optional[Series]) -> None:
        self._suppress_mask_change = True
        self.var_mask_invert.set(bool(getattr(s, "mask_invert", False)) if s is not None else False)
        self._suppress_mask_change = False
        state = "normal" if s is not None else "disabled"
        if hasattr(self, "chk_mask_invert"):
            self.chk_mask_invert.configure(state=state)
        if hasattr(self, "btn_clear_mask"):
            self.btn_clear_mask.configure(state=state)


    def _on_mask_invert_change(self) -> None:
        if self._suppress_mask_change:
            return
        s = self.owner.series_actor._get_series(self._active_series_id) if self._active_series_id is not None else None
        if s is None:
            return
        s.mask_invert = bool(self.var_mask_invert.get())
        self._finish_mask_edit()


    def _clear_mask(self) -> None:
        s = self.owner.series_actor._get_series(self._active_series_id) if self._active_series_id is not None else None
        if s is None:
            return
        s.mask_bitmap = None
        self._finish_mask_edit()


    def _on_ctrl_toggle_press(self, event):
        if self.tool_mode.get() == "addseries":
            if (event.state & 0x0001) and self.series_mode.get() == "scatter":
                return
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
        s = self.owner.series_actor._get_series(self._active_series_id) if (self._active_series_id is not None) else None
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
            s = self.owner.series_actor._get_series(self._active_series_id)
        if s is None and self.series:
            s = self.series[-1]
        if not s or getattr(s, "chart_kind", s.mode) not in ("line", "scatter"):
            return False

        xpx, ypx = self._to_image_px(event.x, event.y)
        if s.extra_seeds_px is None:
            s.extra_seeds_px = []
        s.extra_seeds_px.append((xpx, ypx))

        if getattr(s, "chart_kind", s.mode) == "scatter":
            if s.scatter_seed_bboxes_px is None:
                s.scatter_seed_bboxes_px = []
            if s.seed_bbox_px is None:
                r = 12
                s.seed_bbox_px = (xpx - r, ypx - r, xpx + r, ypx + r)
                s.scatter_seed_bboxes_px.append(s.seed_bbox_px)
            r = 12
            s.scatter_seed_bboxes_px.append((xpx - r, ypx - r, xpx + r, ypx + r))
            try:
                self.owner.extractor._extract_series(s)
            except Exception as e:
                self._show_error("Series extraction failed", str(e))
                return True
            self.owner.series_actor._update_tree_row(s)
            self._redraw_overlay()
            return True

        try:
            self.owner.extractor._extract_series(s)
        except Exception as e:
            self._show_error("Series extraction failed", str(e))
            return True

        self.owner.series_actor._update_tree_row(s)
        self._redraw_overlay()
        return True

    # ---------- Loupe ----------

    def _on_motion(self, event):
        xpx, ypx = self._to_image_px(event.x, event.y)
        self._draw_loupe(xpx, ypx)
        self._update_mask_cursor(event.x, event.y)


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


    def _on_canvas_leave(self, _event):
        self._clear_mask_cursor()


    def _clear_mask_cursor(self) -> None:
        self.canvas.delete("mask_cursor")


    def _update_mask_cursor(self, cx: float, cy: float) -> None:
        if self.tool_mode.get() != "mask":
            self._clear_mask_cursor()
            return
        s = self.owner.series_actor._get_series(self._active_series_id) if self._active_series_id is not None else None
        if s is None:
            self._clear_mask_cursor()
            return
        r = max(1.0, float(self._mask_pen_radius) * float(self._scale))
        x0 = float(cx) - r
        y0 = float(cy) - r
        x1 = float(cx) + r
        y1 = float(cy) + r
        self.canvas.delete("mask_cursor")
        if self._mask_pen_shape == "square":
            self.canvas.create_rectangle(
                x0, y0, x1, y1,
                outline="black", width=2,
                tags=("mask_cursor",)
            )
            self.canvas.create_rectangle(
                x0, y0, x1, y1,
                outline="#7AE7FF", width=1,
                tags=("mask_cursor",)
            )
        else:
            self.canvas.create_oval(
                x0, y0, x1, y1,
                outline="black", width=2,
                tags=("mask_cursor",)
            )
            self.canvas.create_oval(
                x0, y0, x1, y1,
                outline="#7AE7FF", width=1,
                tags=("mask_cursor",)
            )

    # ---------- Series management ----------

    def _add_series_from_click(self, xpx: int, ypx: int):
        self._add_series_from_seed(xpx, ypx, seed_bbox_px=None)


    def _add_scatter_series_from_bbox(self, bbox: Tuple[int, int, int, int]):
        x0, y0, x1, y1 = bbox
        cx = int(round((x0 + x1) / 2.0))
        cy = int(round((y0 + y1) / 2.0))
        self._add_series_from_seed(cx, cy, seed_bbox_px=bbox)


    def _add_scatter_template_from_bbox(self, bbox: Tuple[int, int, int, int]):
        s = self.owner.series_actor._get_series(self._active_series_id) if self._active_series_id is not None else None
        if s is None and self.series:
            s = self.series[-1]
        if s is None or getattr(s, "chart_kind", s.mode) != "scatter":
            self._add_scatter_series_from_bbox(bbox)
            return

        patch = self._patch_from_bbox(bbox)
        if self._should_reject_uniform_patch(patch):
            self._show_info("Scatter", "No marker detected in the selection.")
            return

        x0, y0, x1, y1 = bbox
        centroid = self._bbox_mask_centroid(bbox, s.color_bgr)
        if centroid is None:
            cx = int(round((x0 + x1) / 2.0))
            cy = int(round((y0 + y1) / 2.0))
        else:
            cx, cy = centroid
        if s.extra_seeds_px is None:
            s.extra_seeds_px = []
        s.extra_seeds_px.append((int(cx), int(cy)))
        if s.seed_bbox_px is None:
            s.seed_bbox_px = bbox
        s.scatter_seed_bboxes_px.append(bbox)

        marker_bbox = self._marker_bbox_from_bbox(bbox, s.color_bgr)
        if marker_bbox is not None:
            s.scatter_marker_bboxes_px.append(marker_bbox)

        try:
            self.owner.extractor._extract_series(s)
        except Exception as e:
            self._show_error("Series extraction failed", str(e))
            return

        self.owner.series_actor._update_tree_row(s)
        self._redraw_overlay()


    def _marker_bbox_from_bbox(
        self,
        bbox: Tuple[int, int, int, int],
        target: Tuple[int, int, int],
    ) -> Optional[Tuple[int, int, int, int]]:
        bx0, by0, bx1, by1 = bbox
        bx0 = max(0, min(self._iw - 1, int(bx0)))
        bx1 = max(0, min(self._iw - 1, int(bx1)))
        by0 = max(0, min(self._ih - 1, int(by0)))
        by1 = max(0, min(self._ih - 1, int(by1)))
        if bx1 < bx0:
            bx0, bx1 = bx1, bx0
        if by1 < by0:
            by0, by1 = by1, by0
        patch = self._bgr[by0:by1+1, bx0:bx1+1, :]
        if not patch.size:
            return None
        tol = int(self.var_tol.get())
        mask = color_distance_mask(patch, target, tol).astype(np.uint8)
        nz = np.argwhere(mask > 0)
        if not nz.size:
            return None
        y_min = int(nz[:, 0].min())
        y_max = int(nz[:, 0].max())
        x_min = int(nz[:, 1].min())
        x_max = int(nz[:, 1].max())
        h = int(y_max - y_min)
        w = int(x_max - x_min)
        cy = float(nz[:, 0].mean())
        cx = float(nz[:, 1].mean())
        x0c = int(round(cx - w / 2.0))
        y0c = int(round(cy - h / 2.0))
        x1c = int(x0c + w)
        y1c = int(y0c + h)
        x0c = max(0, min((bx1 - bx0), x0c))
        y0c = max(0, min((by1 - by0), y0c))
        x1c = max(0, min((bx1 - bx0), x1c))
        y1c = max(0, min((by1 - by0), y1c))
        return (int(bx0 + x0c), int(by0 + y0c), int(bx0 + x1c), int(by0 + y1c))


    def _patch_from_bbox(self, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        bx0, by0, bx1, by1 = bbox
        bx0 = max(0, min(self._iw - 1, int(bx0)))
        bx1 = max(0, min(self._iw - 1, int(bx1)))
        by0 = max(0, min(self._ih - 1, int(by0)))
        by1 = max(0, min(self._ih - 1, int(by1)))
        if bx1 < bx0:
            bx0, bx1 = bx1, bx0
        if by1 < by0:
            by0, by1 = by1, by0
        return self._bgr[by0:by1+1, bx0:bx1+1, :]


    def _should_reject_uniform_patch(self, patch: np.ndarray) -> bool:
        if patch is None or not patch.size:
            return True
        flat = patch.reshape(-1, 3).astype(np.float32)
        rng = flat.max(axis=0) - flat.min(axis=0)
        std = float(flat.std())
        if np.max(rng) > 2.0 or std > 1.0:
            return False
        roi = self._roi_px()
        rx0, ry0, rx1, ry1 = roi
        roi_patch = self._bgr[ry0:ry1, rx0:rx1, :]
        if not roi_patch.size:
            return True
        roi_med = np.median(roi_patch.reshape(-1, 3).astype(np.float32), axis=0)
        patch_med = np.median(flat, axis=0)
        dist = float(np.linalg.norm(patch_med - roi_med))
        tol = float(self.var_tol.get())
        return dist <= max(5.0, tol)


    def _bbox_mask_centroid(
        self,
        bbox: Tuple[int, int, int, int],
        target: Tuple[int, int, int],
    ) -> Optional[Tuple[int, int]]:
        bx0, by0, bx1, by1 = bbox
        bx0 = max(0, min(self._iw - 1, int(bx0)))
        bx1 = max(0, min(self._iw - 1, int(bx1)))
        by0 = max(0, min(self._ih - 1, int(by0)))
        by1 = max(0, min(self._ih - 1, int(by1)))
        if bx1 < bx0:
            bx0, bx1 = bx1, bx0
        if by1 < by0:
            by0, by1 = by1, by0
        patch = self._bgr[by0:by1+1, bx0:bx1+1, :]
        if not patch.size:
            return None
        tol = int(self.var_tol.get())
        mask = color_distance_mask(patch, target, tol).astype(np.uint8)
        nz = np.argwhere(mask > 0)
        if not nz.size:
            return None
        cy = float(nz[:, 0].mean())
        cx = float(nz[:, 1].mean())
        return (int(round(bx0 + cx)), int(round(by0 + cy)))


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
            if self.series_mode.get() == "scatter":
                if self._should_reject_uniform_patch(patch):
                    self._show_info("Scatter", "No marker detected in the selection.")
                    return
        else:
            b, g, r = self._bgr[ypx, xpx].tolist()
            target = (int(b), int(g), int(r))
            patch = self._bgr[max(0,ypx-3):ypx+4, max(0,xpx-3):xpx+4, :]
            if self.series_mode.get() == "scatter":
                if self._should_reject_uniform_patch(patch):
                    self._show_info("Scatter", "No marker detected in the selection.")
                    return

        sid = self._next_series_id
        self._next_series_id += 1

        name = f"Series {sid}"
        mode = self.series_mode.get()
        calibration = self.owner.calibrator._make_series_calibration_from_ui()

        s = Series(
            id=sid,
            name=name,
            color_bgr=target,
            chart_kind=mode,
            stacked=bool(self.var_stacked.get()),
            stride_mode=str(self.owner.calibrator._stride_mode_for_calibration(calibration, mode)),
            prefer_outline=bool(self.var_prefer_outline.get()),
            color_tol=int(self.var_tol.get()),
            scatter_match_thresh=float(self.var_scatter_match_thresh.get()),
            calibration=calibration,
        )
        if seed_bbox_px is not None:
            centroid = self._bbox_mask_centroid(seed_bbox_px, target)
            if centroid is not None:
                xpx, ypx = centroid
        s.seed_px = (xpx, ypx)
        if seed_bbox_px is not None:
            s.seed_bbox_px = seed_bbox_px
            s.scatter_seed_bboxes_px = [seed_bbox_px]
            marker_bbox = self._marker_bbox_from_bbox(seed_bbox_px, target)
            if marker_bbox is not None:
                s.seed_marker_bbox_px = marker_bbox
                s.scatter_marker_bboxes_px = [marker_bbox]
        self.series.append(s)
        self.owner.series_actor._insert_tree_row(s)

        # Extract immediately
        try:
            self.owner.extractor._extract_series(s)
        except Exception as e:
            self._show_error("Series extraction failed", str(e))

        if s.mode == "scatter" and not s.points:
            sample_mode = getattr(s.calibration, "sample_mode", "free")
            if s.seed_bbox_px and sample_mode in ("fixed_x", "fixed_y"):
                msg = "Template matching found no matches. Try lowering Match thresh or use Free."
            else:
                msg = "No scatter points detected for that selection."
            if self.tree.exists(str(s.id)):
                self.tree.delete(str(s.id))
            self.series = [ser for ser in self.series if ser.id != s.id]
            self._show_info("Scatter", msg)
            self._redraw_overlay()
            return

        self.owner.series_actor._select_series(sid)
        self._redraw_overlay()


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
                    "Add series (Scatter): Click a marker color to extract all points of that color. "
                    "Shift+drag draws a rectangular selection around a marker to match on shape. "
                    "Ctrl+click or Ctrl+Shift+drag add additional seed locations and/or marker shapes."
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
        elif mode == "mask":
            msg = (
                "Mask series: Select a series, then draw a mask to limit extraction. "
                "Left-drag paints, right-drag erases. Shift+drag draws a rectangle. "
                "Mouse wheel changes pen size; Shift+wheel toggles circle/square. "
                f"Use 'Invert' to switch between include/exclude behavior. "
                f"Pen: {int(self._mask_pen_radius)}px {self._mask_pen_shape}."
            )
        else:
            # fallback / none
            msg = (
                "Select a tool mode above. Region controls what pixels are scanned; X/Y ticks control how pixels map to data units."
            )

        self.tip_var.set(msg)


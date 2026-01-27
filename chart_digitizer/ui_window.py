
from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
import tkinter.font as tkfont
from typing import Callable, Optional, Tuple, List

import platform
from PIL import Image

from .cv_utils import pil_to_bgr
from .data_model import Series
from .ui_state import CalibrationOverlayState
from .calibration import AxisScale
from .ui_panel_toolbar import ToolbarPanel
from .ui_panel_calibration import CalibrationPanel, Calibrator
from .ui_panel_extraction import ExtractionPanel, Extractor
from .ui_panel_series import SeriesPanel, SeriesActor
from .ui_panel_export import ExportPanel, Exporter
from .ui_panel_canvas import CanvasPanel, CanvasActor


class ChartDigitizerWindow(tk.Toplevel):
    def __init__(self, parent: tk.Tk, *, image: Image.Image, on_append_text: Callable[[str], None]):
        super().__init__(parent)
        self.title("Chart -> CSV")
        self.geometry("1180x760")
        self.resizable(True, True)
        if platform.system().lower() != "windows":
            self.transient(parent)  # modeless: no grab_set
        self._on_append_text = on_append_text

        self._pil = image.convert("RGB")
        self._bgr = pil_to_bgr(self._pil)
        self._iw, self._ih = self._pil.size

        self.state = CalibrationOverlayState(xmin_px=0, ymin_px=0, xmax_px=self._iw, ymax_px=self._ih)
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

        self.canvas_actor = CanvasActor(self)
        self.extractor = Extractor(self)
        self.exporter = Exporter(self)
        self.series_actor = SeriesActor(self)
        self.calibrator = Calibrator(self)


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
        self.var_tol = tk.IntVar(value=60)
        self.var_sample_mode = tk.StringVar(value="Free")
        self.var_scatter_match_thresh = tk.DoubleVar(value=0.6)
        self.var_auto_rerun = tk.BooleanVar(value=False)

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
        self._scatter_rb_additive: bool = False
        self._mask_pen_radius = 8
        self._mask_pen_shape = "circle"
        self._mask_drawing = False
        self._mask_draw_value = 255
        self._mask_rb_start: Optional[Tuple[int, int]] = None
        self._mask_rb_id: Optional[int] = None
        self._mask_rb_active: bool = False
        self._mask_rb_value = 255
        self._suppress_mask_change = False
        self._suppress_extraction_change = False
        self.var_mask_invert = tk.BooleanVar(value=False)

        # axis click staging
        self._pending_axis: Optional[str] = None  # 'x0','x1','y0','y1' progress
        self._axis_drag_start: Optional[Tuple[int, int]] = None
        self._axis_dragging: bool = False
        self._axis_drag_axis: Optional[str] = None  # "x" or "y"
        self._axis_drag_active: bool = False
        self._axis_drag_threshold = 3
        self._nudge_target: Optional[str] = None  # roi_tl|roi_tr|roi_bl|roi_br|x0|x1|y0|y1
        self._axis_label_pos: dict[str, Tuple[int, int]] = {}
        self._axis_label_bboxes: dict[str, Tuple[int, int, int, int]] = {}
        self._axis_label_drag: Optional[str] = None
        self._seed_drag: Optional[Tuple[int, str, int]] = None  # (series_id, kind, index)
        self._seed_drag_start: Optional[Tuple[int, int]] = None
        self._last_mouse_canvas: Optional[Tuple[int, int]] = None
        self._roi_resize_corner: Optional[str] = None
        self._roi_resize_anchor: Optional[Tuple[int, int]] = None

        self._build_ui()
        self._on_series_mode_change()
        self.canvas_actor._render_image()
        self.canvas_actor._update_tip()

    # ---------- UI ----------
    def _build_ui(self):
        root = ttk.Frame(self, padding=8)
        root.pack(fill="both", expand=True)

        self._panes = ttk.Panedwindow(root, orient="horizontal")
        self._panes.pack(fill="both", expand=True)

        left = ttk.Frame(self._panes)
        right = ttk.Frame(self._panes, width=340)
        self._right_panel = right
        self._panes.add(left, weight=2)
        self._panes.add(right, weight=1)

        # Toolbar
        self.toolbar_panel = ToolbarPanel(
            self,
            left,
            on_tool_change=self._on_tool_change,
            on_series_mode_change=self._on_series_mode_change,
        )

        # Canvas panel
        self.canvas_panel = CanvasPanel(self, left, actor=self.canvas_actor)

        # Right panel: calibration / extraction / series / export
        self.calibration_panel = CalibrationPanel(
            self,
            right,
            on_sample_mode_change=self.calibrator._on_sample_mode_change,
            on_x_scale_change=self.calibrator._on_x_scale_change,
            on_y_scale_change=self.calibrator._on_y_scale_change,
            on_x_step_unit_change=self.calibrator._on_x_step_unit_changed,
            on_y_step_unit_change=self.calibrator._on_y_step_unit_changed,
            on_apply_calibration_active=self.calibrator._apply_calibration_to_active,
            on_apply_calibration_choose=self.calibrator._apply_calibration_to_choose,
            on_apply_calibration_all=self.calibrator._apply_calibration_to_all,
        )
        self.extraction_panel = ExtractionPanel(
            self,
            right,
            on_prefer_outline_change=self.extractor._on_prefer_outline_change,
            on_mask_invert_change=self.canvas_actor._on_mask_invert_change,
            on_clear_mask=self.canvas_actor._clear_mask,
            on_rerun_extraction=self.extractor._rerun_active_series,
        )
        self.export_panel = ExportPanel(
            self,
            right,
            on_append_csv=self.exporter._append_csv,
            on_export_csv=self.exporter._export_csv,
            on_close=self.destroy,
        )
        self.series_panel = SeriesPanel(
            self,
            right,
            on_toggle=self.series_actor._toggle_series_enabled,
            on_delete=self.series_actor._delete_series,
            on_delete_all=self.series_actor._delete_all_series,
            on_select=self.series_actor._on_tree_select,
            on_rename=self.series_actor._on_tree_double_click,
        )

        self.after(0, self._set_default_pane_ratio)
        self.date_fmt.trace_add("write", lambda *_: self.calibrator._on_date_fmt_change())
        self.var_categories.trace_add("write", lambda *_: self.calibration_panel.update_category_count())
        self.calibrator._refresh_scale_ui()
        self.canvas_actor._sync_mask_controls(None)
        self.extractor._update_extraction_controls()
        self.var_tol.trace_add("write", lambda *_: self.extractor._on_extraction_setting_change())
        self.var_scatter_match_thresh.trace_add("write", lambda *_: self.extractor._on_extraction_setting_change())
        if getattr(self, "_right_panel", None) is not None:
            self._right_panel.bind("<Configure>", self._on_right_panel_configure)
        self.after(0, self._update_series_list_height)
        self._bind_tab_order(self.calibration_panel.frame)

    def _bind_tab_order(self, container: ttk.Frame) -> None:
        widgets = self._collect_focus_widgets(container)
        for w in widgets:
            w.bind("<Tab>", self._on_focus_tab, add="+")
            w.bind("<Shift-Tab>", self._on_focus_shift_tab, add="+")

    def _collect_focus_widgets(self, container: ttk.Frame) -> List[tk.Widget]:
        widgets: List[tk.Widget] = []
        stack = list(container.winfo_children())
        while stack:
            w = stack.pop(0)
            stack.extend(w.winfo_children())
            if not w.winfo_viewable():
                continue
            state = None
            try:
                state = str(w.cget("state"))
            except Exception:
                state = None
            if state == "disabled":
                continue
            if isinstance(w, (ttk.Entry, ttk.Combobox)):
                widgets.append(w)
        return widgets

    def _ordered_focus_widgets(self, container: ttk.Frame) -> List[tk.Widget]:
        widgets = self._collect_focus_widgets(container)
        widgets.sort(key=lambda w: (w.winfo_rooty(), w.winfo_rootx()))
        return widgets

    def _on_focus_tab(self, event):
        return self._focus_next_in_container(event.widget, forward=True)

    def _on_focus_shift_tab(self, event):
        return self._focus_next_in_container(event.widget, forward=False)

    def _focus_next_in_container(self, widget: tk.Widget, *, forward: bool) -> str:
        container = widget.nametowidget(widget.winfo_parent())
        widgets = self._ordered_focus_widgets(container)
        if not widgets or widget not in widgets:
            return "break"
        idx = widgets.index(widget)
        next_idx = (idx + (1 if forward else -1)) % len(widgets)
        widgets[next_idx].focus_set()
        return "break"

    def _configure_tree_rowheight(self):
        style = ttk.Style(self)
        font_name = style.lookup("Treeview", "font") or "TkDefaultFont"
        try:
            font = tkfont.nametofont(font_name)
        except Exception:
            font = tkfont.nametofont("TkDefaultFont")
        rowheight = max(18, int(font.metrics("linespace")) + 6)
        self._tree_rowheight = int(rowheight)
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

    def _on_right_panel_configure(self, _evt=None) -> None:
        if getattr(self, "_resize_after_id", None) is not None:
            try:
                self.after_cancel(self._resize_after_id)
            except Exception:
                pass
        self._resize_after_id = self.after(50, self._update_series_list_height)

    def _update_series_list_height(self) -> None:
        if not getattr(self, "_series_frame", None):
            return
        if not getattr(self, "_right_panel", None):
            return
        if not getattr(self, "_tree_rowheight", None):
            return
        self._right_panel.update_idletasks()
        total = int(self._right_panel.winfo_height())
        if total <= 0:
            return
        fixed = 0
        for frm in (getattr(self, "_calibration_frame", None),
                    getattr(self, "_extraction_frame", None),
                    getattr(self, "_export_frame", None),
                    getattr(self, "_series_btns_frame", None)):
            if frm is None:
                continue
            h = int(frm.winfo_height())
            req = int(frm.winfo_reqheight())
            fixed += max(h, req)
        available = max(0, total - fixed - 40)
        rows = max(3, int(available / max(1, int(self._tree_rowheight))))
        self.tree.configure(height=rows)
        self._series_frame.pack_propagate(True)

    def _on_series_mode_change(self):
        if getattr(self, "_suppress_series_mode_change", False):
            return
        if self.tool_mode.get() == "editseries" and self._active_series_id is not None:
            s = self.series_actor._get_series(self._active_series_id)
            if s is not None:
                desired = getattr(s, "chart_kind", s.mode)
                if self.series_mode.get() != desired:
                    self._show_info(
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
        self.canvas_actor._update_tip()
        self.calibrator._refresh_scale_ui()
        self.calibrator._update_sample_mode_ui()
        self.extractor._update_extraction_controls()

    def _on_tool_change(self):
        mode = self.tool_mode.get()
        self._pending_axis = None
        if mode == "xaxis":
            self._pending_axis = "x0"
        elif mode == "yaxis":
            self._pending_axis = "y0"
        elif mode == "editseries" and self._active_series_id is not None:
            s = self.series_actor._get_series(self._active_series_id)
            if s is not None:
                self.series_actor._set_ui_mode_from_series(s)
                self.calibrator._apply_series_calibration_to_ui(s.calibration)
        if mode == "mask":
            s = self.series_actor._get_series(self._active_series_id) if self._active_series_id is not None else None
            self.canvas_actor._sync_mask_controls(s)
        self.canvas_actor._update_tip()
        self.canvas_actor._redraw_overlay()

    # ---------- Image rendering ----------
    def _on_fit_toggle(self):
        self.canvas_actor._render_image()

    def _show_info(self, title: str, message: str) -> None:
        messagebox.showinfo(title, message, parent=self)

    def _show_error(self, title: str, message: str) -> None:
        messagebox.showerror(title, message, parent=self)


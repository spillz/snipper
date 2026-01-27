from __future__ import annotations

import tkinter as tk
from tkinter import ttk, simpledialog
from typing import Callable, List, Optional, Tuple

from .data_model import Series


class SeriesPanel:
    def __init__(
        self,
        owner,
        parent: tk.Widget,
        *,
        on_toggle: Callable[[], None],
        on_delete: Callable[[], None],
        on_delete_all: Callable[[], None],
        on_select: Callable[[object], None],
        on_rename: Callable[[object], None],
    ) -> None:
        self.owner = owner
        frame = ttk.LabelFrame(parent, text="Series", padding=8)
        self.frame = frame
        owner._series_frame = frame
        frame.pack(side="top", fill="both", expand=True, pady=(8, 0))

        owner.tree = ttk.Treeview(
            frame,
            columns=("enabled", "name", "cal", "n"),
            show="headings",
            selectmode="browse",
            height=12,
        )
        owner.tree.heading("enabled", text="On")
        owner.tree.heading("name", text="Name")
        owner.tree.heading("cal", text="Cal")
        owner.tree.heading("n", text="Pts")
        owner.tree.column("enabled", width=34, anchor="center")
        owner.tree.column("name", width=170, anchor="w")
        owner.tree.column("cal", width=60, anchor="center")
        owner.tree.column("n", width=60, anchor="e")
        owner._configure_tree_rowheight()

        btns = ttk.Frame(frame)
        owner._series_btns_frame = btns
        btns.pack(side="bottom", fill="x", pady=(8, 0))
        ttk.Button(btns, text="Toggle On/Off", command=on_toggle).pack(side="left")
        ttk.Button(btns, text="Delete", command=on_delete).pack(side="left", padx=(8, 0))
        ttk.Button(btns, text="Delete All", command=on_delete_all).pack(side="left", padx=(8, 0))

        owner.tree.pack(side="top", fill="both", expand=True)

        owner.tree.bind("<<TreeviewSelect>>", on_select)
        owner.tree.bind("<Double-1>", on_rename)


class SeriesActor:
    def __init__(self, owner) -> None:
        object.__setattr__(self, "owner", owner)

    def __getattr__(self, name):
        return getattr(self.owner, name)

    def __setattr__(self, name, value) -> None:
        if name == "owner":
            object.__setattr__(self, name, value)
            return
        setattr(self.owner, name, value)

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
            self.owner.canvas_actor._sync_mask_controls(None)
        else:
            self._active_series_id = int(sel[0])
            s = self._get_series(self._active_series_id)
            if s is not None:
                if self.tool_mode.get() == "editseries":
                    self._set_ui_mode_from_series(s)
                self.owner.calibrator._apply_series_calibration_to_ui(s.calibration)
                self.owner.extractor._apply_series_extraction_to_ui(s)
            self.owner.canvas_actor._sync_mask_controls(s)
            self.owner.extractor._update_extraction_controls()
        self.owner.canvas_actor._redraw_overlay()


    def _select_series(self, sid: int):
        self.tree.selection_set(str(sid))
        self.tree.see(str(sid))
        self._active_series_id = sid
        self.owner.canvas_actor._sync_mask_controls(self._get_series(sid))


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
        self.owner.canvas_actor._redraw_overlay()


    def _delete_series(self):
        if self._active_series_id is None:
            return
        sid = self._active_series_id
        self.series = [s for s in self.series if s.id != sid]
        if self.tree.exists(str(sid)):
            self.tree.delete(str(sid))
        self._active_series_id = None
        self.owner.canvas_actor._sync_mask_controls(None)
        self.owner.canvas_actor._redraw_overlay()


    def _delete_all_series(self):
        if not self.series:
            return
        self.series = []
        for item in self.tree.get_children(""):
            self.tree.delete(item)
        self._active_series_id = None
        self.owner.canvas_actor._redraw_overlay()


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

    def _set_ui_mode_from_series(self, s: Series) -> None:
        kind = getattr(s, "chart_kind", getattr(s, "mode", "line"))
        self.series_mode.set(kind)
        self.var_stacked.set(bool(getattr(s, "stacked", False)))


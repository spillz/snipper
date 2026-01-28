from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable


class ToolbarPanel:
    def __init__(
        self,
        owner,
        parent: tk.Widget,
        *,
        on_tool_change: Callable[[], None],
        on_series_mode_change: Callable[[], None],
    ) -> None:
        self.owner = owner
        self.frame = ttk.Frame(parent)
        self.frame.pack(side="top", fill="x")

        ttk.Label(self.frame, text="Tool:").pack(side="left")
        for lbl, val in [
            ("Set Region", "data_region"),
            ("Set X ticks", "xaxis"),
            ("Set Y ticks", "yaxis"),
            ("Add series", "addseries"),
            ("Edit series", "editseries"),
            ("Mask series", "mask"),
        ]:
            ttk.Radiobutton(
                self.frame,
                text=lbl,
                value=val,
                variable=owner.tool_mode,
                command=on_tool_change,
            ).pack(side="left", padx=(8, 0))

        ttk.Separator(self.frame, orient="vertical").pack(side="left", fill="y", padx=10)
        ttk.Label(self.frame, text="Series type:").pack(side="left")

        series_combo = ttk.Combobox(
            self.frame,
            textvariable=owner.series_mode,
            state="readonly",
            width=10,
            values=("line", "scatter", "column", "bar", "area"),
        )
        series_combo.pack(side="left", padx=(6, 0))
        series_combo.bind("<<ComboboxSelected>>", lambda _e: on_series_mode_change())

        ttk.Checkbutton(
            self.frame,
            text="Stacked",
            variable=owner.var_stacked,
            command=on_series_mode_change,
        ).pack(side="left", padx=(10, 0))

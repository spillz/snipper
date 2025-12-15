import os
import io
import csv
import json
import time
import shutil
import statistics
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter import simpledialog

import mss
import pytesseract
from pytesseract import Output
from PIL import Image, ImageTk, ImageOps, ImageFilter

from chart_digitizer.ui_dialog import ChartDigitizerDialog

import ctypes

SPI_GETWORKAREA = 0x0030

class RECT(ctypes.Structure):
    _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                ("right", ctypes.c_long), ("bottom", ctypes.c_long)]

def get_work_area():
    r = RECT()
    ctypes.windll.user32.SystemParametersInfoW(SPI_GETWORKAREA, 0, ctypes.byref(r), 0)
    return r.left, r.top, r.right, r.bottom

def park_offscreen_workarea(win, pad=50):
    win.update_idletasks()
    w = win.winfo_width() or win.winfo_reqwidth()
    h = win.winfo_height() or win.winfo_reqheight()
    left, top, right, bottom = get_work_area()
    # park just beyond the right/bottom edge
    x = right + pad
    y = bottom + pad
    win.geometry(f"{w}x{h}+{x}+{y}")

# -----------------------------
# Tesseract discovery
# -----------------------------

import os
import platform
import shutil
from pathlib import Path
import pytesseract


def _candidate_tesseract_paths() -> list[str]:
    """
    Return a list of candidate absolute paths where tesseract might be installed.
    Uses environment variables and OS conventions to avoid brittle hard-coding.
    """
    sysname = platform.system().lower()
    candidates: list[Path] = []

    if "windows" in sysname:
        # Program Files locations via env vars (system-agnostic within Windows)
        pf = os.environ.get("ProgramFiles")
        pf86 = os.environ.get("ProgramFiles(x86)")
        lad = os.environ.get("LocalAppData")
        ad = os.environ.get("AppData")

        if pf:
            candidates.append(Path(pf) / "Tesseract-OCR" / "tesseract.exe")
        if pf86:
            candidates.append(Path(pf86) / "Tesseract-OCR" / "tesseract.exe")

        # Common per-user install locations (varies by installer/package manager)
        # Keep these conservative; they cost little to check.
        for base in [lad, ad]:
            if base:
                basep = Path(base)
                candidates += [
                    basep / "Programs" / "Tesseract-OCR" / "tesseract.exe",  # some installers
                    basep / "Tesseract-OCR" / "tesseract.exe",
                ]

        # Common package managers (optional but cheap)
        # Chocolatey
        choco = os.environ.get("ChocolateyInstall")
        if choco:
            candidates.append(Path(choco) / "bin" / "tesseract.exe")

        # Scoop (typical)
        userprofile = os.environ.get("USERPROFILE")
        if userprofile:
            candidates.append(Path(userprofile) / "scoop" / "shims" / "tesseract.exe")

    elif "darwin" in sysname:  # macOS
        candidates += [
            Path("/opt/homebrew/bin/tesseract"),   # Homebrew (Apple Silicon)
            Path("/usr/local/bin/tesseract"),      # Homebrew (Intel)
            Path("/usr/bin/tesseract"),
        ]

    else:  # Linux/other Unix
        candidates += [
            Path("/usr/bin/tesseract"),
            Path("/usr/local/bin/tesseract"),
            Path("/snap/bin/tesseract"),
        ]

    # Deduplicate while preserving order
    seen = set()
    out = []
    for p in candidates:
        s = str(p)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _resolve_executable(cmd: str) -> str | None:
    """
    Resolve a command to an absolute executable path, if possible.
    """
    if not cmd:
        return None
    # Absolute/relative path given
    p = Path(cmd)
    if p.is_file():
        return str(p.resolve())

    # Otherwise attempt PATH resolution
    found = shutil.which(cmd)
    return found


def ensure_tesseract_configured() -> None:
    """
    Configure pytesseract to find the Tesseract executable.

    Resolution order:
    1) TESSERACT_CMD env var (path or command)
    2) PATH lookup for 'tesseract'
    3) Common install locations by OS/environment variables
    """
    # 1) Explicit env var
    env_cmd = os.environ.get("TESSERACT_CMD", "").strip()
    if env_cmd:
        resolved = _resolve_executable(env_cmd)
        if resolved:
            pytesseract.pytesseract.tesseract_cmd = resolved
            return
        raise RuntimeError(f"TESSERACT_CMD is set but could not be resolved: {env_cmd}")

    # 2) PATH
    resolved = _resolve_executable("tesseract")
    if resolved:
        pytesseract.pytesseract.tesseract_cmd = resolved
        return

    # 3) Common locations
    for cand in _candidate_tesseract_paths():
        if Path(cand).is_file():
            pytesseract.pytesseract.tesseract_cmd = cand
            return

    # Not found: point to Mannheim builds
    raise RuntimeError(
        "Tesseract OCR engine not found.\n\n"
        "Install Tesseract for Windows (recommended builds):\n"
        "  https://github.com/UB-Mannheim/tesseract/wiki\n\n"
        "After installing, either add it to PATH or set:\n"
        '  setx TESSERACT_CMD "C:\\Path\\To\\tesseract.exe"'
    )


# -----------------------------
# Settings + Presets persistence
# -----------------------------

CONFIG_PATH = Path.home() / ".snip_ocr_config.json"


@dataclass
class OcrSettings:
    lang: str = "eng"
    psm: int = 6
    oem: int = 3
    preserve_interword_spaces: bool = False
    whitelist: str = ""
    blacklist: str = ""
    user_defined_dpi: str = ""          # blank or integer string
    extra_vars: dict = None             # {key: value} for -c key=value

    def __post_init__(self):
        if self.extra_vars is None:
            self.extra_vars = {}

    def build_tesseract_config(self) -> str:
        parts = [f"--oem {int(self.oem)}", f"--psm {int(self.psm)}"]
        c_opts = {}

        if self.preserve_interword_spaces:
            c_opts["preserve_interword_spaces"] = "1"
        if self.whitelist.strip():
            c_opts["tessedit_char_whitelist"] = self.whitelist.strip()
        if self.blacklist.strip():
            c_opts["tessedit_char_blacklist"] = self.blacklist.strip()
        if self.user_defined_dpi.strip():
            c_opts["user_defined_dpi"] = self.user_defined_dpi.strip()

        for k, v in (self.extra_vars or {}).items():
            k = str(k).strip()
            if k:
                c_opts[k] = str(v)

        for k, v in c_opts.items():
            v_str = str(v)
            # Quote only when needed
            if any(ch.isspace() for ch in v_str) or '"' in v_str:
                v_str = v_str.replace('"', '\\"')
                parts.append(f'-c {k}="{v_str}"')
            else:
                parts.append(f"-c {k}={v_str}")

        return " ".join(parts)


def default_presets() -> dict[str, OcrSettings]:
    """
    Presets are intentionally conservative. Users can "Save As…" and tune.
    """
    return {
        "Paragraph": OcrSettings(lang="eng", psm=6, oem=3, preserve_interword_spaces=False),
        "UI sparse": OcrSettings(lang="eng", psm=11, oem=3, preserve_interword_spaces=False),
        "Numbers only": OcrSettings(
            lang="eng",
            psm=7,
            oem=3,
            preserve_interword_spaces=False,
            whitelist="0123456789.-()%,$",
            blacklist=""
        ),
        "Table": OcrSettings(
            lang="eng",
            psm=6,
            oem=3,
            preserve_interword_spaces=True
        ),
    }


def load_config() -> tuple[str, dict[str, OcrSettings]]:
    """
    Returns (active_preset_name, presets_dict)
    """
    presets = default_presets()
    active = "Paragraph"

    if not CONFIG_PATH.exists():
        return active, presets

    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        active = data.get("active_preset", active)

        user_presets = data.get("presets", {})
        if isinstance(user_presets, dict):
            for name, sdict in user_presets.items():
                if not isinstance(sdict, dict):
                    continue
                merged = {**OcrSettings().__dict__, **sdict}
                presets[name] = OcrSettings(**merged)
    except Exception:
        # If config is corrupt, fall back without blocking app usage.
        return active, presets

    if active not in presets:
        active = next(iter(presets.keys()))
    return active, presets


def save_config(active_preset: str, presets: dict[str, OcrSettings]) -> None:
    payload = {
        "active_preset": active_preset,
        "presets": {name: asdict(s) for name, s in presets.items()}
    }
    CONFIG_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# -----------------------------
# Image preprocessing
# -----------------------------

def preprocess_for_ocr(img: Image.Image) -> Image.Image:
    g = ImageOps.grayscale(img)
    w, h = g.size
    g = g.resize((max(1, w * 2), max(1, h * 2)), Image.Resampling.LANCZOS)
    g = g.filter(ImageFilter.MedianFilter(size=3))
    g = ImageOps.autocontrast(g)
    threshold = 165
    g = g.point(lambda p: 255 if p > threshold else 0)
    return g


# -----------------------------
# OCR routines
# -----------------------------

def ocr_text(img: Image.Image, settings: OcrSettings) -> str:
    pre = preprocess_for_ocr(img)
    cfg = settings.build_tesseract_config()
    text = pytesseract.image_to_string(pre, lang=settings.lang, config=cfg)
    return text.strip()


def ocr_table_to_csv(img: Image.Image, settings: OcrSettings) -> str:
    """
    Table OCR that preserves spaces inside cells by using word bounding boxes.
    """
    pre = preprocess_for_ocr(img)
    cfg = settings.build_tesseract_config()
    data = pytesseract.image_to_data(pre, lang=settings.lang, config=cfg, output_type=Output.DICT)

    words = []
    n = len(data["text"])
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        if not txt:
            continue
        conf_raw = str(data["conf"][i]).strip()
        conf = int(float(conf_raw)) if conf_raw not in ("", "-1") else -1
        if conf >= 0 and conf < 35:
            continue
        x, y, w, h = int(data["left"][i]), int(data["top"][i]), int(data["width"][i]), int(data["height"][i])
        words.append((x, y, w, h, txt))

    if not words:
        return ""

    heights = [h for (_, _, _, h, _) in words]
    med_h = statistics.median(heights) if heights else 12
    row_tol = max(6, int(0.6 * med_h))

    words.sort(key=lambda t: (t[1], t[0]))

    # Group into rows by y proximity
    rows = []
    for x, y, w, h, txt in words:
        placed = False
        for row in rows:
            if abs(y - row["y"]) <= row_tol:
                row["items"].append((x, y, w, h, txt))
                row["y"] = int((row["y"] * 0.8) + (y * 0.2))
                placed = True
                break
        if not placed:
            rows.append({"y": y, "items": [(x, y, w, h, txt)]})

    for row in rows:
        row["items"].sort(key=lambda t: t[0])

    # Typical inter-word gaps -> column break threshold
    gaps = []
    for row in rows:
        items = row["items"]
        for a, b in zip(items, items[1:]):
            ax, _, aw, _, _ = a
            bx, _, _, _, _ = b
            gaps.append(max(0, bx - (ax + aw)))
    typical_gap = statistics.median(gaps) if gaps else 10
    col_break_gap = max(int(2.5 * typical_gap), int(1.25 * med_h))

    table = []
    max_cols = 0
    for row in rows:
        items = row["items"]
        cells = []
        cur = []
        prev_right = None

        for x, y, w, h, txt in items:
            if prev_right is not None and (x - prev_right) >= col_break_gap:
                cells.append(" ".join(cur).strip())
                cur = []
            cur.append(txt)
            prev_right = x + w

        if cur:
            cells.append(" ".join(cur).strip())

        table.append(cells)
        max_cols = max(max_cols, len(cells))

    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    for row in table:
        if len(row) < max_cols:
            row = row + [""] * (max_cols - len(row))
        writer.writerow(row)

    return buf.getvalue().rstrip()


# -----------------------------
# Snipping Overlay
# -----------------------------

class SnipOverlay(tk.Toplevel):
    def __init__(self, parent: tk.Tk, screenshot: Image.Image, on_snip):
        super().__init__(parent)
        self.parent = parent
        self.on_snip = on_snip
        self.screenshot = screenshot

        self.overrideredirect(True)
        self.attributes("-topmost", True)

        mon0 = getattr(parent, "_mss_mon0", None) or {"left": 0, "top": 0}
        self.vleft = int(mon0.get("left", 0))
        self.vtop = int(mon0.get("top", 0))

        self.width, self.height = screenshot.size
        self.geometry(f"{self.width}x{self.height}+{self.vleft}+{self.vtop}")

        self.canvas = tk.Canvas(self, width=self.width, height=self.height, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self.photo = ImageTk.PhotoImage(screenshot)
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")

        # Dim overlay via stipple
        self.dim_ids = [
            self.canvas.create_rectangle(0, 0, 0, 0, fill="black", stipple="gray50", outline="")
            for _ in range(4)
        ]

        self.start_x = None
        self.start_y = None
        self.rect_id = None

        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<ButtonPress-3>", lambda e: self.__cancel())
        self.bind("<Escape>", lambda e: self.__cancel())
        self.focus_force()

    def _update_dim(self, x1, y1, x2, y2):
        W, H = self.width, self.height
        dims = [
            (0, 0, W, y1),      # top
            (0, y2, W, H),      # bottom
            (0, y1, x1, y2),    # left
            (x2, y1, W, y2),    # right
        ]
        for rid, (a, b, c, d) in zip(self.dim_ids, dims):
            self.canvas.coords(rid, a, b, c, d)

    def on_mouse_down(self, event):
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        if self.rect_id is not None:
            self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y, outline="#00B4FF", width=2
        )

    def on_mouse_drag(self, event):
        if self.rect_id is None:
            return

        cur_x = self.canvas.canvasx(event.x)
        cur_y = self.canvas.canvasy(event.y)

        x1 = min(self.start_x, cur_x)
        y1 = min(self.start_y, cur_y)
        x2 = max(self.start_x, cur_x)
        y2 = max(self.start_y, cur_y)

        self.canvas.coords(self.rect_id, x1, y1, x2, y2)
        self._update_dim(x1, y1, x2, y2)

    def on_mouse_up(self, event):
        if self.rect_id is None:
            return
        end_x = self.canvas.canvasx(event.x)
        end_y = self.canvas.canvasy(event.y)

        x1 = int(min(self.start_x, end_x))
        y1 = int(min(self.start_y, end_y))
        x2 = int(max(self.start_x, end_x))
        y2 = int(max(self.start_y, end_y))

        if (x2 - x1) < 5 or (y2 - y1) < 5:
            self.__cancel()
            return

        cropped = self.screenshot.crop((x1, y1, x2, y2))

        # IMPORTANT: destroy the overlay (releases grab/focus) before invoking callbacks.
        cb = self.on_snip
        master = self.master
        self.destroy()
        if cb is not None and master is not None:
            master.after(1, lambda: cb(cropped))
        return

    def __cancel(self):
        self.destroy()


# -----------------------------
# Settings popup with presets
# -----------------------------

class SettingsDialog(tk.Toplevel):
    def __init__(self, parent: tk.Tk, active_name: str, presets: dict[str, OcrSettings]):
        super().__init__(parent)
        self.title("OCR Settings")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self._presets = presets
        self._active = active_name
        self._result = None

        self.var_preset = tk.StringVar(value=active_name)

        # Current editable working copy
        self._iworking = self._clone_settings(presets[active_name])

        # UI vars
        self.var_lang = tk.StringVar(value=self._iworking.lang)
        self.var_psm = tk.IntVar(value=int(self._iworking.psm))
        self.var_oem = tk.IntVar(value=int(self._iworking.oem))
        self.var_preserve = tk.BooleanVar(value=bool(self._iworking.preserve_interword_spaces))
        self.var_whitelist = tk.StringVar(value=self._iworking.whitelist or "")
        self.var_blacklist = tk.StringVar(value=self._iworking.blacklist or "")
        self.var_dpi = tk.StringVar(value=self._iworking.user_defined_dpi or "")

        self._build()

    @staticmethod
    def _clone_settings(s: OcrSettings) -> OcrSettings:
        d = asdict(s)
        return OcrSettings(**d)

    def _build(self):
        pad = {"padx": 10, "pady": 6}
        frm = ttk.Frame(self)
        frm.pack(fill="both", expand=True, padx=10, pady=10)

        # Preset selector row
        r = 0
        ttk.Label(frm, text="Preset:").grid(row=r, column=0, sticky="w", **pad)

        self.cmb = ttk.Combobox(frm, textvariable=self.var_preset, width=26, state="readonly",
                                values=sorted(self._presets.keys()))
        self.cmb.grid(row=r, column=1, sticky="w", **pad)
        self.cmb.bind("<<ComboboxSelected>>", lambda e: self._load_selected_preset())

        btn_row = ttk.Frame(frm)
        btn_row.grid(row=r, column=2, sticky="e", padx=10, pady=6)

        ttk.Button(btn_row, text="Save", command=self._save_overwrite).pack(side="left", padx=(0, 6))
        ttk.Button(btn_row, text="Save As…", command=self._save_as).pack(side="left", padx=(0, 6))
        ttk.Button(btn_row, text="Delete", command=self._delete_preset).pack(side="left")

        # Core fields
        r += 1
        ttk.Label(frm, text="Language (lang):").grid(row=r, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.var_lang, width=18).grid(row=r, column=1, sticky="w", **pad)

        r += 1
        ttk.Label(frm, text="PSM:").grid(row=r, column=0, sticky="w", **pad)
        ttk.Combobox(frm, textvariable=self.var_psm, width=16, state="readonly",
                     values=[3, 4, 6, 7, 11, 12, 13]).grid(row=r, column=1, sticky="w", **pad)

        r += 1
        ttk.Label(frm, text="OEM:").grid(row=r, column=0, sticky="w", **pad)
        ttk.Combobox(frm, textvariable=self.var_oem, width=16, state="readonly",
                     values=[0, 1, 2, 3]).grid(row=r, column=1, sticky="w", **pad)

        r += 1
        ttk.Checkbutton(frm, text="Preserve interword spaces (layout/tables)",
                        variable=self.var_preserve).grid(row=r, column=0, columnspan=3, sticky="w", **pad)

        r += 1
        ttk.Label(frm, text="Whitelist:").grid(row=r, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.var_whitelist, width=48).grid(row=r, column=1, columnspan=2, sticky="w", **pad)

        r += 1
        ttk.Label(frm, text="Blacklist:").grid(row=r, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.var_blacklist, width=48).grid(row=r, column=1, columnspan=2, sticky="w", **pad)

        r += 1
        ttk.Label(frm, text="DPI hint (optional):").grid(row=r, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.var_dpi, width=18).grid(row=r, column=1, sticky="w", **pad)

        # Advanced vars (key=value) text box
        r += 1
        ttk.Label(frm, text="Advanced (-c key=value), one per line:").grid(row=r, column=0, columnspan=3, sticky="w", **pad)

        r += 1
        self.txt_extra = tk.Text(frm, width=60, height=6)
        self.txt_extra.grid(row=r, column=0, columnspan=3, sticky="w", padx=10, pady=(0, 6))
        self._set_extra_text(self._presets[self._active].extra_vars)

        # Footer buttons
        r += 1
        footer = ttk.Frame(frm)
        footer.grid(row=r, column=0, columnspan=3, sticky="e", padx=10, pady=10)

        ttk.Button(footer, text="Preview", command=self._preview).pack(side="left", padx=(0, 8))
        ttk.Button(footer, text="Use This Preset", command=self._apply_and_close).pack(side="left", padx=(0, 8))
        ttk.Button(footer, text="Cancel", command=self._cancel).pack(side="left")

    def _set_extra_text(self, extra_vars: dict):
        self.txt_extra.delete("1.0", "end")
        if not extra_vars:
            return
        lines = [f"{k}={v}" for k, v in extra_vars.items()]
        self.txt_extra.insert("1.0", "\n".join(lines))

    def _parse_extra_vars(self) -> dict:
        raw = self.txt_extra.get("1.0", "end-1c")
        out = {}
        for line in raw.splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if "=" not in s:
                raise ValueError(f"Invalid advanced setting (expected key=value): {line}")
            k, v = s.split("=", 1)
            k = k.strip()
            v = v.strip()
            if not k:
                raise ValueError(f"Invalid key in advanced setting: {line}")
            out[k] = v
        return out

    def _collect_settings_from_ui(self) -> OcrSettings:
        lang = self.var_lang.get().strip() or "eng"
        psm = int(self.var_psm.get())
        oem = int(self.var_oem.get())
        dpi = self.var_dpi.get().strip()
        if dpi and not dpi.isdigit():
            raise ValueError("DPI hint must be blank or an integer (e.g., 300).")

        extra_vars = self._parse_extra_vars()

        return OcrSettings(
            lang=lang,
            psm=psm,
            oem=oem,
            preserve_interword_spaces=bool(self.var_preserve.get()),
            whitelist=self.var_whitelist.get(),
            blacklist=self.var_blacklist.get(),
            user_defined_dpi=dpi,
            extra_vars=extra_vars,
        )

    def _load_selected_preset(self):
        name = self.var_preset.get()
        s = self._presets[name]
        self._active = name

        self.var_lang.set(s.lang)
        self.var_psm.set(int(s.psm))
        self.var_oem.set(int(s.oem))
        self.var_preserve.set(bool(s.preserve_interword_spaces))
        self.var_whitelist.set(s.whitelist or "")
        self.var_blacklist.set(s.blacklist or "")
        self.var_dpi.set(s.user_defined_dpi or "")
        self._set_extra_text(s.extra_vars or {})

    def _preview(self):
        try:
            s = self._collect_settings_from_ui()
            messagebox.showinfo("Tesseract config preview", s.build_tesseract_config())
        except Exception as e:
            messagebox.showerror("Invalid settings", str(e))

    def _save_overwrite(self):
        """
        Overwrite the currently selected preset with UI values.
        """
        try:
            s = self._collect_settings_from_ui()
        except Exception as e:
            messagebox.showerror("Invalid settings", str(e))
            return

        name = self.var_preset.get()
        self._presets[name] = s
        messagebox.showinfo("Saved", f'Preset "{name}" updated.')

    def _save_as(self):
        """
        Save current UI values as a new preset name.
        """
        try:
            s = self._collect_settings_from_ui()
        except Exception as e:
            messagebox.showerror("Invalid settings", str(e))
            return

        name = simpledialog.askstring("Save Preset As", "Preset name:", parent=self)
        if not name:
            return
        name = name.strip()
        if not name:
            return

        if name in self._presets:
            if not messagebox.askyesno("Overwrite?", f'Preset "{name}" exists. Overwrite?', parent=self):
                return

        self._presets[name] = s
        # Refresh dropdown list
        self.cmb["values"] = sorted(self._presets.keys())
        self.var_preset.set(name)
        messagebox.showinfo("Saved", f'Preset "{name}" saved.')

    def _delete_preset(self):
        name = self.var_preset.get()
        if name in ("Paragraph", "UI sparse", "Numbers only", "Table"):
            messagebox.showwarning("Not allowed", f'Built-in preset "{name}" cannot be deleted.')
            return
        if not messagebox.askyesno("Delete", f'Delete preset "{name}"?', parent=self):
            return
        if name in self._presets:
            del self._presets[name]
        self.cmb["values"] = sorted(self._presets.keys())
        self.var_preset.set("Paragraph")
        self._load_selected_preset()

    def _apply_and_close(self):
        """
        User chooses to make the currently selected preset active and close dialog.
        """
        try:
            # Make sure any unsaved UI edits are captured in that preset if desired
            # We do NOT auto-save. "Use This Preset" activates the selected preset name.
            active_name = self.var_preset.get()
            self._result = ("apply", active_name, self._presets)
            self.destroy()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _cancel(self):
        self._result = None
        self.destroy()

    def result(self):
        return self._result

# -----------------------------
# Main App
# -----------------------------

class SnipOCRApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Data Snipper")
        self.geometry("980x720")
        self._saved_geometry = self.wm_geometry()
        self._dialogs = set()   # Track Toplevel dialogs we create

        self.active_preset, self.presets = load_config()

        self._build_ui()

        try:
            ensure_tesseract_configured()
        except Exception as e:
            messagebox.showerror("Tesseract not configured", str(e))


    def register_dialog(self, win: tk.Toplevel) -> None:
        self._dialogs.add(win)
        # Remove from set when destroyed
        win.bind("<Destroy>", lambda _e, w=win: self._dialogs.discard(w), add="+")
        if False:
            # Also handle user closing via window manager
            win.protocol("WM_DELETE_WINDOW", lambda w=win: self._safe_destroy(w))

    def _safe_destroy(self, w: tk.Toplevel) -> None:
        try:
            if w.winfo_exists():
                w.destroy()
        except Exception:
            pass

    # def close_all_dialogs(self) -> None:
    #     # Copy to avoid mutation during iteration
    #     for w in list(self._dialogs):
    #         self._safe_destroy(w)
    #     self._dialogs.clear()

    def hide_main_window(self):
        # Save exact current size+position
        self._saved_geometry = self.wm_geometry()

        # Park off-screen (no withdraw/iconify)
        park_offscreen_workarea(self)
        self.update_idletasks()


    def restore_main_window(self):
        if getattr(self, "_saved_geometry", None):
            self.wm_geometry(self._saved_geometry)

        # Optional: bring forward if you want it visible immediately
        # (Does not change WM state like deiconify does.)
        self.lift()
        self.focus_force()
        self.update_idletasks()

    def begin_new_snip(self, mode='OCR'):
        # 1) close dialogs
        for w in list(self.winfo_children()):
            if isinstance(w, tk.Toplevel):
                try: w.destroy()
                except Exception: pass

        self.hide_main_window()
        self.update()

        # 4) small delay before starting overlay/capture
        #    (lets DWM finish the fade)
        if mode=='OCR':
            self.after(200, self.start_snip)   # try 50–120ms
        elif mode=='CHART':
            self.after(200, self.start_snip_chart)   # try 50–120ms

    def current_settings(self) -> OcrSettings:
        return self.presets[self.active_preset]

    def _build_ui(self):
        toolbar = ttk.Frame(self, padding=(8, 8, 8, 4))
        toolbar.pack(side="top", fill="x")

        ttk.Button(toolbar, text="Chart Snip", command=lambda: self.begin_new_snip('CHART')).pack(side="left", padx=(8, 0))
        ttk.Button(toolbar, text="Text Snip (OCR)", command=lambda: self.begin_new_snip('OCR')).pack(side="left")
        ttk.Button(toolbar, text="OCR Settings…", command=self.open_settings).pack(side="left", padx=(8, 0))

        # Show active preset name read-only (helps users trust what’s active)
        self.active_label = tk.StringVar(value=f"Preset: {self.active_preset}")
        ttk.Label(toolbar, textvariable=self.active_label).pack(side="left", padx=(12, 0))

        self.table_mode = tk.BooleanVar(value=False)
        ttk.Checkbutton(toolbar, text="Table mode (CSV)", variable=self.table_mode).pack(side="left", padx=(12, 0))

        ttk.Button(toolbar, text="Copy All", command=self.copy_all).pack(side="left", padx=(12, 0))
        ttk.Button(toolbar, text="Clear", command=self.clear_text).pack(side="left", padx=(8, 0))
        ttk.Button(toolbar, text="Save As…", command=self.save_as).pack(side="left", padx=(8, 0))

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(toolbar, textvariable=self.status_var).pack(side="right")

        body = ttk.Frame(self, padding=(8, 4, 8, 8))
        body.pack(side="top", fill="both", expand=True)

        self.text = tk.Text(body, wrap="word", undo=True)
        self.text.pack(side="left", fill="both", expand=True)

        yscroll = ttk.Scrollbar(body, orient="vertical", command=self.text.yview)
        yscroll.pack(side="right", fill="y")
        self.text.configure(yscrollcommand=yscroll.set)

        footer = ttk.Frame(self, padding=(8, 0, 8, 8))
        footer.pack(side="bottom", fill="x")
        ttk.Label(
            footer,
            text="Tips: Drag to select region. Right-click or Esc cancels. Text appends; edit and save when ready."
        ).pack(side="left")

    def set_status(self, msg: str):
        self.status_var.set(msg)
        self.update_idletasks()

    def open_settings(self):
        dlg = SettingsDialog(self, self.active_preset, self.presets)
        self.wait_window(dlg)
        res = dlg.result()
        if res is None:
            return

        action, active_name, presets = res
        if action == "apply":
            self.presets = presets
            self.active_preset = active_name
            self.active_label.set(f"Preset: {self.active_preset}")

            try:
                save_config(self.active_preset, self.presets)
                self.set_status(f"Settings saved to {CONFIG_PATH.name}")
            except Exception as e:
                messagebox.showerror("Save failed", str(e))

    def start_snip(self):
        try:
            ensure_tesseract_configured()
        except Exception as e:
            messagebox.showerror("Tesseract not configured", str(e))
            return

        self.set_status("Capturing screen…")
        with mss.mss() as sct:
            mon0 = sct.monitors[0]
            self._mss_mon0 = mon0
            shot = sct.grab(mon0)

        img = Image.frombytes("RGB", (shot.width, shot.height), shot.rgb)

        def on_snip(cropped: Image.Image):
            self.set_status("Running OCR…")
            t0 = time.time()
            try:
                settings = self.current_settings()
                if self.table_mode.get():
                    extracted = ocr_table_to_csv(cropped, settings)
                else:
                    extracted = ocr_text(cropped, settings)
            except Exception as e:
                messagebox.showerror("OCR failed", str(e))
                return

            dt = time.time() - t0
            if extracted:
                if self.text.get("1.0", "end-1c").strip():
                    self.text.insert("end", "\n\n")
                self.text.insert("end", extracted)
                self.text.see("end")
                self.set_status(f"OCR appended ({dt:.2f}s).")
            else:
                self.set_status("No text detected.")

        overlay = SnipOverlay(self, img, on_snip)

        def restore(event=None):
            if event is not None and event.widget is not overlay:
                return
            try:
                self.restore_main_window()
            except Exception:
                pass
            if self.status_var.get().startswith("Capturing"):
                self.set_status("Ready.")

        overlay.bind("<Destroy>", restore)


    def start_snip_chart(self):
        """
        Screen snip → open modeless ChartDigitizerDialog → append CSV into notepad on demand.
        """
        self.set_status("Capturing screen…")
        with mss.mss() as sct:
            mon0 = sct.monitors[0]
            self._mss_mon0 = mon0
            shot = sct.grab(mon0)

        img = Image.frombytes("RGB", (shot.width, shot.height), shot.rgb)

        def on_snip(cropped: Image.Image):
            # Restore main window immediately (dialog is modeless)
            try:
                pass
                # self.restore_main_window()
            except Exception:
                pass

            def append_text(s: str):
                if not s:
                    return
                if self.text.get("1.0", "end-1c").strip():
                    self.text.insert("end", "\n\n")
                self.text.insert("end", s)
                self.text.see("end")
                self.set_status("Chart CSV appended.")

            try:
                dlg = ChartDigitizerDialog(self, image=cropped, on_append_text=append_text)
                self.register_dialog(dlg)
                self.set_status("Chart digitizer opened.")
            except Exception as e:
                traceback.print_exc()
                messagebox.showerror("Chart digitizer failed to open", "See console for traceback.")
                self.set_status("Ready.")
        overlay = SnipOverlay(self, img, on_snip)

        def restore(event=None):
            if event is not None and event.widget is not overlay:
                return
            try:
                self.restore_main_window()
            except Exception:
                pass
            if self.status_var.get().startswith("Capturing"):
                self.set_status("Ready.")

        overlay.bind("<Destroy>", restore)

    def copy_all(self):
        content = self.text.get("1.0", "end-1c")
        self.clipboard_clear()
        self.clipboard_append(content)
        self.set_status("Copied to clipboard.")

    def clear_text(self):
        if messagebox.askyesno("Clear", "Clear the editor?"):
            self.text.delete("1.0", "end")
            self.set_status("Cleared.")

    def save_as(self):
        content = self.text.get("1.0", "end-1c")
        if not content.strip():
            messagebox.showinfo("Save As", "Nothing to save.")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")],
            title="Save OCR text as…"
        )
        if not path:
            return

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            self.set_status(f"Saved: {Path(path).name}")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))


def main():
    app = SnipOCRApp()
    app.mainloop()


if __name__ == "__main__":
    main()

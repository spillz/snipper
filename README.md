<!-- IMAGE: Snipping overlay selection -->
![Snipping splash](images/snipper-banner.png)

# Data Snipper

A GUI application for scraping text and data from screengrabs that combines an OCR notepad and an integrated tool to convert charts to data.

Find a table, chart or text on screen that you want to capture as text or data and snip it

<!-- IMAGE: Snipping overlay selection -->
![Snipping overlay](images/snipping-overlay.png)

Text snips (including tables) get appended immediately into a text notebook, which you can hand tweak

<!-- IMAGE: Main app window -->
![Main app window](images/main-window.png)

Chart snips allow you to pick out individual series for various chart types after following some simple setup steps. You can append the data in the notepad or save to CSV.

<!-- IMAGE: OCR configuration -->
![Chart snipping](images/chart-snip.png)


## Install

Install python then in the terminal:

```bash
pip install mss pillow pytesseract
pip install opencv-python
```

Tesseract must be installed separately (pytessseract is just a binding). On Windows, Tesseract builds can be found here:
- https://github.com/UB-Mannheim/tesseract/wiki

## Run

```bash
python snipper.py
```

See additional notes for running in UV and compiling your own binaries below.

## Text snipping workflow

1. Click **Text snip** and drag-select the region of text you wish to convert from image to text.
2. The text will be appended to the end of the notebook
3. If you don't like the output, delete it and try again.
4. You can use the OCR settings panel to tune your output using tesseract configuration.

<!-- IMAGE: OCR configuration -->
![OCR configuration](images/OCR-config.png)

## Chart snipping workflow

1. Open a chart image in your viewer of choice, then in snipper, click **Chart snip** and drag-select the chart region in your viewer.
2. Setup the chart for data point detection:
   - Toolbar: **Set region**: The data region is the part of the chart that contains series data. Defaults to the full image snip.
   - Toolbar: **Set X axis**: click twice for any pair of x-axis tick marks; enter x0/x1 values. These do not restrict what is scanned, only how values are converted from pixels to x-axis units or categories.
   - Toolbar: **Set Y axis**: click twice for any pair of y-axis pixels; enter y0/y1 values. These do not restrict what is scanned, only how values are converted from pixels to y-axis units.
   - Chart type: Line/Area/Column/Bar/Scatter supported (if you have combo charts you can change this per series).
   - Stacked: check for stacked Line/Area/Column values.
   - Sample: Free/Fixed X/Fixed Y (scatter only); use Fixed X/Y for gridded scatter.
   - Span detection: checked measures the length of the column/bar/area rather than distance from the axis.
   - Calibration: specify axis scales and categories or x0/x1 bounds and step size. Dates are supported.
   - Buttons: **Update** applies the current calibration to the active series, **Apply to...** lets you choose series to update, **Apply to all** updates all series.
3. Toolbar: change tool to **Add series**: 
   - line/area/column/bar: click on the series color in the chart to add one series at a time to the Series list.
   - scatter: click a marker color to find simple solid markers. Drag a rectangle around a marker to match its shape. Ctrl+click adds more guide points; Ctrl+drag adds more shapes to the active series.
   - You can edit the parameters above between clicks.
4. Toolbar: change tool to **Edit series** to edit:
   - Click on the individual series in the Series list
   - Drag points vertically (line mode) to adjust values.
   - Right-click a point to toggle NA (disabled).
5. Keyboard and fine control:
   - Arrow keys nudge the active item in the canvas (a data region corner, axis tick, point, or seed).
   - Shift+arrow moves faster; Ctrl+arrow cycles which item is active.
   - Enter / Shift+Enter switches tools forward/backward.
   - You can drag the tick labels in **Set X axis** / **Set Y axis** to move the tick positions.
   - You can drag seed markers in **Add series** / **Edit series** to move the seed and re-extract.
6. Toolbar: change tool to **Mask series** (optional)
   - Draw a mask if detection is noisy or misses points; use Invert to exclude/include regions.
   - Re-run extraction after masking if Auto rerun is off.
7. Series list
   - Click series to change the active series
   - Double click on series to change names
   - Delete to drop the series (easy to add again via the "add series" tool)
   - "Toggle on/off" to off excludes from notebook append or CSV export
8. Export:
   - **Append CSV** appends to the notepad in the main window.
   - **Export CSV...** saves a CSV file.

Notes:
- Line charts export in **wide** format with a shared x-index.
- Scatter charts export in **long** format of stacked series with x,y pairs.

## Running with UV and Building Binaries

Use uv to replace pip, venv, and pip-tools with a single, extremely fast tool that creates reproducible Python environments (Python's answer to node.js).

```bash
uv venv
uv pip install -e .
uv run python snipper.py
```

You can create Windows builds with PyInstaller and UV using the following commands for a single-file executable:

```bash
uv pip install -e .[build]
.\build_windows.ps1 -OneFile
```

Replace -OneFile with -OneDirSfx for an installable MSI (requires WiX Toolset v3.x). Use -WixPath if WiX is not on PATH (for example `-WixPath "C:\\Program Files (x86)\\WiX Toolset v3.11\\bin"`).

Version lives in `VERSION` and is embedded into the Windows EXE. Release notes live in `CHANGELOG.md`.


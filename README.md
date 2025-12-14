# Data Snipper

A GUI application for scraping text and data from screengrabs that combines an OCR notepad and an integrated tool to convert charts to data.

Find a table, chart or text on screen that you want to capture as text or data and snip it

<!-- IMAGE: Snipping overlay selection -->
![Snipping overlay](images/snipping-overlay.png)

Text snips (including tables) get appended immediately into a text notebook, which you can hand tweak

<!-- IMAGE: Main app window -->
![Main app window](images/main-window.png)

Chart snips allow you to pick out individual series for various chart type after following some simple setup steps. You can append the data in the notepad or save to CSV.

<!-- IMAGE: OCR configuration -->
![Chart snipping](images/chart-snip.png)


## Install

```bash
pip install mss pillow pytesseract
pip install opencv-python
```

Tesseract must be installed separately. On Windows, recommended builds:
- https://github.com/UB-Mannheim/tesseract/wiki

## Run

```bash
python snipper.py
```

## Text snipping workflow

1. Click **Text snip** and drag-select the region of text you wish to convert from image to text.
2. The text will be appended to the end of the notebook
3. If you don't like the output, delete it and try again.
4. You can use the OCR settings panel to tune your output using tesseract configuation.

<!-- IMAGE: OCR configuration -->
![OCR configuration](images/OCR-config.png)

## Chart snipping workflow

1. Click **Chart snip** and drag-select the chart region.
2. Setup the chart for scanning:
   - Toolbar: **Set region**: The part of the chart that contains series data. Defaults to full snip image.
   - Toolbar: **Set X axis**: click twice for any pair of x-axis tick marks; enter x0/x1 values. These do not restrict what is scanned only how values are converted from pixels to chart units.
   - Toolbar: **Set Y axis**: click twice for any pair of y-axis pixels; enter y0/y1 values. These do not restrict what is scanned only how values are converted from pixels to chart units.
   - Calibration: You can specify X and Y axis scales and optional date unit specifier for the X-axis (UNIX format: %Y=year, %m-%Y=month-year, %m/%d/%Y=month/day/year etc.)
   - Output: **Set the x step**: The interval between series observations. This is a floating point number (default 1) in the units of the date fmt and you may specify partial units (e.g., 0.25 years = quarters)
3. Toolbar: change tool to **Add series**: 
   - click on the lines in the chart image to add one series at a time to the Series list. 
   - You can edit the parameters above between clicks.
3. Toolbar: change tool to **Edit series** to edit:
   - Click on the individual series in the Series list
   - Drag points vertically (line mode) to adjust values.
   - Right-click a point to toggle NA (disabled).
4. Series list
   - Click series to change the active series
   - Double click on series to change names
   - Delete to drop the series (easy to add again via the "add series" tool)
   - "Toggle on/off" to off excludes from notebook append or CSV export
5. Export:
   - **Append CSV** appends to the notepad in the main window.
   - **Export CSV…** saves a CSV file.

Notes:
- Line charts export in **wide** format with a shared x-index.
- Scatter charts export in **long** format of stacked series with x,y pairs.

# Snipper: Snip text and data from screengrabs into a notepad

A lightweight Windows snipping utility written in Python. It lets you drag-select regions of the screen, runs OCR via Tesseract, and appends the extracted text into an editable notepad-style window. You can take multiple snips, clean up the output, and save the final text (or CSV output from table mode) to disk.

## Screenshot Placeholders

<!-- IMAGE: Main app window -->
![Main app window](images/main-window.png)

<!-- IMAGE: Snipping overlay selection -->
![Snipping overlay](images/snipping-overlay.png)

<!-- IMAGE: OCR configuration -->
![Snipping overlay](images/OCR-config.png)

---

## Features

- Interactive region selection (snip) over the full desktop
- OCR results append into an editable text widget (combine multiple snips)
- Save output to `.txt` (and optionally `.csv`)
- Table mode emits CSV using bounding-box reconstruction (robust to spaces inside cells)
- OCR presets (Paragraph / UI sparse / Numbers only / Table)
- Custom presets via **Save As…**
- Settings persist in user home folder (JSON)

---

## Requirements

- Windows 10/11 (primary target)
- Python 3.11+ recommended (3.13 should work if your packages install cleanly)
- Tesseract OCR engine installed (separate from Python packages)

Python dependencies:

```bash
pip install mss pillow pytesseract
```

---

## Installing Tesseract (Windows)

This project uses the Tesseract **engine binary** (`tesseract.exe`) via the Python package `pytesseract`, which is only a wrapper and does not include the engine.

Recommended Windows builds:
- https://github.com/UB-Mannheim/tesseract/wiki

After installation, confirm it works:

```bash
tesseract --version
```

Snipper will attempt to find your tesseract executable in the usual places. If not found, try adding the tesseract folder to your PATH.
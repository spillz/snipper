from __future__ import annotations

import re
import struct
from pathlib import Path
from PIL import Image

SIZE_RE = re.compile(r".*_(\d+)x(\d+)\.png$", re.IGNORECASE)

def rgba_png_to_dib32(im: Image.Image) -> bytes:
    """
    Convert an RGBA Pillow image to ICO DIB (BITMAPINFOHEADER + BGRA pixels + AND mask).
    - 32-bit BGRA pixels, bottom-up
    - AND mask: 1 bit per pixel, padded to 32-bit per row
    """
    im = im.convert("RGBA")
    w, h = im.size
    if w != h:
        raise ValueError(f"ICO entries should be square; got {w}x{h}")

    # BGRA pixels, bottom-up
    px = im.tobytes()  # RGBA row-major, top-down
    # Convert RGBA -> BGRA and flip vertically
    rows = []
    stride = w * 4
    for y in range(h - 1, -1, -1):
        row = px[y * stride : (y + 1) * stride]
        # swap R and B
        bgra = bytearray(row)
        bgra[0::4], bgra[2::4] = row[2::4], row[0::4]
        rows.append(bytes(bgra))
    bgra_pixels = b"".join(rows)

    # AND mask (1 bit per pixel), rows padded to 32-bit boundary
    # In ICO, mask bit = 1 means transparent.
    # We'll set bit=1 when alpha==0, else 0.
    and_row_bytes = ((w + 31) // 32) * 4
    and_mask = bytearray(and_row_bytes * h)

    # Alpha bytes from original (top-down); map to bottom-up mask rows
    alpha = px[3::4]  # top-down
    for y in range(h):
        src_y = h - 1 - y  # because mask is also bottom-up
        for x in range(w):
            a = alpha[src_y * w + x]
            if a == 0:
                byte_index = y * and_row_bytes + (x // 8)
                bit = 7 - (x % 8)
                and_mask[byte_index] |= (1 << bit)

    # BITMAPINFOHEADER (40 bytes)
    # Note: biHeight is *2* height (includes mask)
    header = struct.pack(
        "<IIIHHIIIIII",
        40,         # biSize
        w,          # biWidth
        h * 2,      # biHeight
        1,          # biPlanes
        32,         # biBitCount
        0,          # biCompression (BI_RGB)
        w * h * 4,  # biSizeImage
        0, 0,       # biXPelsPerMeter, biYPelsPerMeter
        0, 0        # biClrUsed, biClrImportant
    )

    return header + bgra_pixels + bytes(and_mask)

def write_ico_dib32(png_paths: list[Path], out_path: Path) -> None:
    """
    Build a .ico with classic DIB frames (best compatibility for small icons).
    No resampling: each PNG is embedded at its native size.
    """
    images = []
    for p in png_paths:
        im = Image.open(p)
        w, h = im.size
        if w != h:
            raise ValueError(f"Non-square: {p} is {w}x{h}")
        dib = rgba_png_to_dib32(im)
        images.append((w, h, dib, p.name))

    # Sort ascending by size (tidy)
    images.sort(key=lambda t: t[0])

    # ICONDIR
    header = struct.pack("<HHH", 0, 1, len(images))

    # ICONDIRENTRY list
    offset = 6 + 16 * len(images)
    entries = []
    for (w, h, dib, name) in images:
        bWidth = 0 if w == 256 else w
        bHeight = 0 if h == 256 else h
        bColorCount = 0
        bReserved = 0
        wPlanes = 1
        wBitCount = 32
        dwBytesInRes = len(dib)
        dwImageOffset = offset

        entries.append(struct.pack(
            "<BBBBHHII",
            bWidth, bHeight, bColorCount, bReserved,
            wPlanes, wBitCount,
            dwBytesInRes, dwImageOffset
        ))
        offset += dwBytesInRes

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        f.write(header)
        for e in entries:
            f.write(e)
        for (_, _, dib, _) in images:
            f.write(dib)

def collect_named_pngs(images_dir: str | Path, stem: str = "snipper-icon") -> list[Path]:
    images_dir = Path(images_dir)
    candidates = list(images_dir.glob(f"{stem}_*x*.png"))

    by_size: dict[tuple[int, int], Path] = {}
    for p in candidates:
        m = SIZE_RE.match(p.name)
        if not m:
            continue
        w, h = int(m.group(1)), int(m.group(2))
        by_size[(w, h)] = p

    if not by_size:
        raise FileNotFoundError(f"No {stem}_*x*.png found in {images_dir}")

    return list(by_size.values())

if __name__ == "__main__":
    pngs = collect_named_pngs("images", stem="snipper-icon")
    write_ico_dib32(pngs, Path("images/snipper-icon.ico"))
    print("Wrote images/snipper-icon.ico")

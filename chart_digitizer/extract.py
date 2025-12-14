
from __future__ import annotations

from typing import List, Tuple, Optional, Sequence
import numpy as np
import math

from .cv_utils import require_cv2, color_distance_mask

def _fill_missing_y(y: List[Optional[float]], fallback: float) -> List[float]:
    # Fill missing with interpolation; flatline ends; if all missing -> fallback.
    n = len(y)
    if n == 0:
        return []
    if all(v is None for v in y):
        return [fallback]*n

    # forward fill for leading
    first = next((i for i,v in enumerate(y) if v is not None), None)
    assert first is not None
    for i in range(0, first):
        y[i] = y[first]

    # backward fill for trailing
    last = next((i for i in range(n-1, -1, -1) if y[i] is not None), None)
    assert last is not None
    for i in range(last+1, n):
        y[i] = y[last]

    # interpolate interior gaps
    i = 0
    while i < n:
        if y[i] is not None:
            i += 1
            continue
        j = i
        while j < n and y[j] is None:
            j += 1
        # now gap is [i, j-1], with y[i-1] and y[j] not None
        y0 = float(y[i-1])
        y1 = float(y[j])
        gap = j - (i-1)
        for k in range(i, j):
            t = (k - (i-1)) / gap
            y[k] = y0 + t*(y1-y0)
        i = j
    return [float(v) for v in y]

def extract_line_series(
    bgr: np.ndarray,
    roi: Tuple[int,int,int,int],
    target_bgr: Tuple[int,int,int],
    tol: int,
    xpx_grid: List[int],
    *,
    seed_px: Tuple[int,int],
    extra_seeds_px: Optional[Sequence[Tuple[int,int]]] = None,
    band_tight_px: int = 6,          # tight band half-width during normal tracking
    band_reacq_px: int = 40,         # max half-width during reacquire
    max_jump_px: int = 30,           # reject sudden y jumps (text/labels)
    max_gap_cols: int = 40,          # maximum consecutive missing columns before giving up (will become None)
    lookahead_px: int = 10,          # how far ahead to search in x during reacquire
) -> Tuple[List[Tuple[int,int]], List[Optional[int]]]:
    """
    Seeded continuity-first line extraction.

    1) Build a dense y_by_x trace across ROI by walking x pixel-by-pixel.
       - Prefer continuity in a tight y-band around current y.
       - Only on breaks do we use slope prediction + widening band + x-lookahead.
    2) Sample y_by_x at xpx_grid and return ypx_raw aligned to xpx_grid.

    Returns:
      px_points: list of (xpx, ypx) for points at xpx_grid (with missing filled later by caller)
      ypx_raw: aligned list with None for missing
    """
    require_cv2()
    import cv2

    x0, y0, x1, y1 = roi
    roi_img = bgr[y0:y1, x0:x1]

    # Binary mask of candidate pixels
    mask = color_distance_mask(roi_img, target_bgr, tol).astype(np.uint8) * 255

    # Light morphology: connect stroke pixels but avoid destroying thin lines
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)

    H, W = mask.shape[:2]

    # --- seed handling ---
    sx, sy = seed_px
    seeds = [(sx, sy)]
    if extra_seeds_px:
        seeds.extend(list(extra_seeds_px))

    # Convert seeds to ROI-local coordinates, keep those that fall within ROI
    seeds_local: List[Tuple[int,int]] = []
    for px, py in seeds:
        if x0 <= px < x1 and y0 <= py < y1:
            seeds_local.append((px - x0, py - y0))

    # If seed is outside ROI, fall back to midline
    if not seeds_local:
        seeds_local = [(W//2, H//2)]

    # Choose starting seed closest to the first grid x if possible, else closest to ROI center
    if xpx_grid:
        first_x = int(min(max(xpx_grid[0], x0), x1-1)) - x0
    else:
        first_x = W//2

    start_seed = min(seeds_local, key=lambda s: abs(s[0] - first_x))
    cur_x, cur_y = start_seed

    # --- helpers ---
    def clamp_int(v: int, lo: int, hi: int) -> int:
        return lo if v < lo else hi if v > hi else v

    def col_candidates(xc: int, y_center: int, half: int) -> np.ndarray:
        """Return y indices (ROI-local) where mask is on within [y_center-half, y_center+half]."""
        yl = clamp_int(y_center - half, 0, H-1)
        yh = clamp_int(y_center + half, 0, H-1)
        col = mask[yl:yh+1, xc]
        ys = np.where(col > 0)[0]
        return ys + yl  # ROI-local y

    # Robust local choice: pick nearest candidate to y_center
    def pick_nearest(ys: np.ndarray, y_center: int) -> Optional[int]:
        if ys.size == 0:
            return None
        j = int(np.argmin(np.abs(ys - y_center)))
        return int(ys[j])

    # Estimate slope from last few accepted points (simple 2-point slope)
    # Track recent accepted (x,y) in ROI-local coords
    recent: List[Tuple[int,int]] = [(cur_x, cur_y)]

    # Dense trace: y_by_x for every ROI-local x column, None if missing
    y_by_x: List[Optional[int]] = [None] * W

    # We’ll walk left->right across ROI to build full trace.
    # Start tracking from ROI-local x=0 to W-1, but we can begin at cur_x and fill both sides.
    def track_direction(x_start: int, x_end: int, step: int):
        nonlocal cur_y
        miss_run = 0

        for xc in range(x_start, x_end, step):
            # Use the last known y as center
            y_center = int(cur_y)

            # Normal tracking: tight band around current y
            ys = col_candidates(xc, y_center, band_tight_px)
            y_pick = pick_nearest(ys, y_center)

            if y_pick is None:
                # Break: attempt reacquire with prediction + widening search + x lookahead
                miss_run += 1
                if miss_run > max_gap_cols:
                    y_by_x[xc] = None
                    continue

                # Predict y from recent slope (if we have at least 2 points)
                if len(recent) >= 2:
                    (x2, y2) = recent[-1]
                    (x1r, y1r) = recent[-2]
                    dx = max(1, abs(x2 - x1r))
                    dy = (y2 - y1r) / dx
                    y_pred = int(round(y2 + dy * (xc - x2)))
                else:
                    y_pred = y_center

                best = None  # (cost, found_x, found_y)
                # search forward a bit in x (in the direction we are tracking)
                for xo in range(0, lookahead_px + 1):
                    xq = xc + xo * step
                    if xq < 0 or xq >= W:
                        break
                    # widen band proportional to miss_run (up to band_reacq_px)
                    half = int(min(band_reacq_px, band_tight_px + 4 * miss_run))
                    ys2 = col_candidates(xq, y_pred, half)
                    if ys2.size == 0:
                        continue
                    # choose nearest to prediction
                    yq = int(ys2[np.argmin(np.abs(ys2 - y_pred))])
                    cost = abs(yq - y_pred) + 0.2 * abs(xq - xc)
                    if best is None or cost < best[0]:
                        best = (cost, xq, yq)

                if best is None:
                    y_by_x[xc] = None
                    continue

                _, found_x, found_y = best
                # If we “reacquired” at a later x, fill the skipped xs as missing; tracking resumes at found_x
                # Also gate implausible jumps (same-colored text)
                if abs(found_y - cur_y) > max_jump_px:
                    y_by_x[xc] = None
                    continue

                cur_y = found_y
                y_by_x[found_x] = found_y
                recent.append((found_x, found_y))
                miss_run = 0
                continue

            # Candidate exists in tight band: continuity-first
            miss_run = 0
            if abs(y_pick - cur_y) > max_jump_px:
                # likely text; treat as missing here
                y_by_x[xc] = None
                continue

            cur_y = y_pick
            y_by_x[xc] = y_pick
            recent.append((xc, y_pick))
            if len(recent) > 6:
                recent.pop(0)

    # Track rightward from seed
    cur_y = start_seed[1]
    recent = [(start_seed[0], start_seed[1])]
    y_by_x[start_seed[0]] = start_seed[1]
    track_direction(start_seed[0], W, +1)

    # Track leftward from seed (reset cur_y/recent)
    cur_y = start_seed[1]
    recent = [(start_seed[0], start_seed[1])]
    track_direction(start_seed[0], -1, -1)

    # --- Sample dense trace at requested xpx_grid ---
    ypx_raw: List[Optional[int]] = []
    px_points: List[Tuple[int,int]] = []

    for xpx in xpx_grid:
        if xpx < x0 or xpx >= x1:
            ypx_raw.append(None)
            continue
        xc = int(xpx - x0)
        y_local = y_by_x[xc]
        if y_local is None:
            ypx_raw.append(None)
            continue
        ypx = int(y0 + y_local)
        ypx_raw.append(ypx)
        px_points.append((int(xpx), int(ypx)))

    return px_points, ypx_raw

def extract_scatter_series(
    bgr: np.ndarray,
    roi: Tuple[int,int,int,int],
    target_bgr: Tuple[int,int,int],
    tol: int,
) -> List[Tuple[int,int]]:
    require_cv2()
    import cv2

    x0,y0,x1,y1 = roi
    roi_img = bgr[y0:y1, x0:x1]
    mask = color_distance_mask(roi_img, target_bgr, tol).astype(np.uint8)*255

    # clean
    mask = cv2.medianBlur(mask, 3)
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    pts: List[Tuple[int,int]] = []
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 4:
            continue
        cx, cy = centroids[i]
        pts.append((x0 + int(round(cx)), y0 + int(round(cy))))
    return pts

def build_x_grid(xmin: float, xmax: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError("x_step must be > 0")
    xs = []
    x = xmin
    tol = step * 1e-9
    while x <= xmax + tol:
        xs.append(round(float(x), 10))
        x += step
    return xs

def build_x_grid_aligned(xmin_val: float, xmax_val: float, step: float, *, anchor: float) -> List[float]:
    if step <= 0:
        raise ValueError("x_step must be > 0")
    if xmax_val < xmin_val:
        xmin_val, xmax_val = xmax_val, xmin_val

    # First lattice point >= xmin_val
    k_start = math.ceil((xmin_val - anchor) / step - 1e-12)
    x = anchor + k_start * step

    xs: List[float] = []
    tol = step * 1e-9
    while x <= xmax_val + tol:
        xs.append(round(float(x), 10))
        x += step
    return xs

def enforce_line_grid(
    x_grid: List[float],
    y_vals: List[Optional[float]],
    fallback_y: float,
) -> List[float]:
    # y_vals already aligned to x_grid but may have None; fill with interpolation/flatline
    return _fill_missing_y(y_vals, fallback=fallback_y)

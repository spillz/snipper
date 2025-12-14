
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

def _runs_from_ys(ys: np.ndarray) -> List[Tuple[int,int,int]]:
    """
    Convert sorted ys into runs.
    Returns list of (y_lo, y_hi, length) inclusive.
    """
    if ys.size == 0:
        return []
    ys = np.sort(ys)
    runs: List[Tuple[int,int,int]] = []
    start = int(ys[0])
    prev = int(ys[0])
    for y in ys[1:]:
        y = int(y)
        if y == prev + 1:
            prev = y
            continue
        runs.append((start, prev, prev - start + 1))
        start = prev = y
    runs.append((start, prev, prev - start + 1))
    return runs

def _run_center(y_lo: int, y_hi: int) -> int:
    return int(round((y_lo + y_hi) / 2.0))

def pick_centered_run_y(
    ys: np.ndarray,
    target_y: int,
    *,
    min_run: int = 2,
    max_center_dist: Optional[int] = None,
) -> Optional[int]:
    """
    Pick the run whose center is near target_y, preferring the longest run.
    Returns the run center (not an edge pixel).
    No x-windowing; no cross-column smoothing.
    """
    runs = _runs_from_ys(ys)  # [(y_lo, y_hi, length), ...]
    if not runs:
        return None

    best = None  # (neg_len, dist, center_y)
    ty = int(target_y)

    for y_lo, y_hi, ln in runs:
        if ln < min_run:
            continue
        cy = int(round((int(y_lo) + int(y_hi)) / 2.0))
        dist = abs(cy - ty)
        if max_center_dist is not None and dist > int(max_center_dist):
            continue

        score = (-int(ln), dist, cy)  # prefer longer, then closer
        if best is None or score < best:
            best = score

    if best is None:
        # fallback: single-pixel nearest (still no smoothing)
        j = int(np.argmin(np.abs(ys - ty)))
        return int(ys[j])

    return int(best[2])

def expand_full_run_center(mask: np.ndarray, xc: int, y_seed: int) -> int:
    """
    Given a seed y where mask is ON at column xc, expand vertically to the full
    contiguous ON-run in that column and return its center y (ROI-local).
    """
    H = mask.shape[0]
    y0 = int(y_seed)
    if y0 < 0: y0 = 0
    if y0 >= H: y0 = H - 1

    # If seed isn't actually on, just return it.
    if mask[y0, xc] == 0:
        return y0

    y_lo = y0
    while y_lo > 0 and mask[y_lo - 1, xc] > 0:
        y_lo -= 1

    y_hi = y0
    while y_hi < H - 1 and mask[y_hi + 1, xc] > 0:
        y_hi += 1

    return int(round((y_lo + y_hi) / 2.0))


def extract_line_series(
    bgr: np.ndarray,
    roi: Tuple[int,int,int,int],
    target_bgr: Tuple[int,int,int],
    tol: int,
    xpx_grid: List[int],
    *,
    seed_px: Tuple[int,int],
    extra_seeds_px: Optional[Sequence[Tuple[int,int]]] = None,
    band_tight_px: int = 6,
    band_reacq_px: int = 40,
    max_jump_px: int = 30,
    max_gap_cols: int = 40,
    lookahead_px: int = 10,
) -> Tuple[List[Tuple[int,int]], List[Optional[int]]]:
    """
    Seeded continuity-first line extraction.

    - Continuity-first in a tight band.
    - On breaks: local reacquire around current y with widening band.
    - If still missing: slope prediction + widening band + x-lookahead.
    - Slope/jump only disambiguates; never hard-reject.

    Centering strategy:
      - pick candidate within band
      - then expand in the FULL column to the contiguous on-run and return its center
    """
    require_cv2()
    import cv2

    x0, y0, x1, y1 = roi
    roi_img = bgr[y0:y1, x0:x1]

    mask = color_distance_mask(roi_img, target_bgr, tol).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)

    H, W = mask.shape[:2]

    # --- seed handling ---
    sx, sy = seed_px
    seeds = [(sx, sy)]
    if extra_seeds_px:
        seeds.extend(list(extra_seeds_px))

    seeds_local: List[Tuple[int,int]] = []
    for px, py in seeds:
        if x0 <= px < x1 and y0 <= py < y1:
            seeds_local.append((px - x0, py - y0))

    if not seeds_local:
        seeds_local = [(W//2, H//2)]

    seed_x_local = int(min(max(sx - x0, 0), W - 1)) if (x0 <= sx < x1) else W // 2
    start_seed = min(seeds_local, key=lambda s: abs(s[0] - seed_x_local))
    cur_x, cur_y = start_seed

    def clamp_int(v: int, lo: int, hi: int) -> int:
        return lo if v < lo else hi if v > hi else v

    def col_candidates(xc: int, y_center: int, half: int) -> np.ndarray:
        yl = clamp_int(y_center - half, 0, H-1)
        yh = clamp_int(y_center + half, 0, H-1)
        col = mask[yl:yh+1, xc]
        ys = np.where(col > 0)[0]
        return ys + yl

    def predict_y(xc: int, recent: List[Tuple[int,int]], fallback: int) -> int:
        if len(recent) >= 2:
            (x2, y2) = recent[-1]
            (x1r, y1r) = recent[-2]
            dx = max(1, abs(x2 - x1r))
            dy = (y2 - y1r) / dx
            return int(round(y2 + dy * (xc - x2)))
        return int(fallback)

    def has_multiple_runs(ys: np.ndarray) -> bool:
        return len(_runs_from_ys(ys)) >= 2

    # recent accepted points (ROI-local)
    recent: List[Tuple[int,int]] = [(cur_x, cur_y)]
    y_by_x: List[Optional[int]] = [None] * W

    def track_direction(x_start: int, x_end: int, step: int):
        nonlocal cur_y, recent

        miss_run = 0
        xc = x_start

        def in_bounds(x: int) -> bool:
            return (x < x_end) if step > 0 else (x > x_end)

        while in_bounds(xc):
            y_center = int(cur_y)

            # 1) Normal tracking
            ys = col_candidates(xc, y_center, band_tight_px)
            y_pick = pick_centered_run_y(ys, y_center, min_run=2)
            if y_pick is not None:
                y_pick = expand_full_run_center(mask, xc, y_pick)

            if y_pick is None:
                miss_run += 1
                if miss_run > max_gap_cols:
                    y_by_x[xc] = None
                    xc += step
                    continue

                # 2) Local reacquire at SAME xc (widen around current y)
                half_local = int(min(band_reacq_px, band_tight_px + 4 * miss_run))
                ys_local = col_candidates(xc, y_center, half_local)

                if ys_local.size:
                    y_pick2 = pick_centered_run_y(ys_local, y_center, min_run=2)
                    if y_pick2 is not None:
                        y_pick2 = expand_full_run_center(mask, xc, y_pick2)

                    if y_pick2 is None:
                        y_pick2 = int(ys_local[np.argmin(np.abs(ys_local - y_center))])
                        y_pick2 = expand_full_run_center(mask, xc, y_pick2)

                    if has_multiple_runs(ys_local) and abs(int(y_pick2) - int(cur_y)) > max_jump_px:
                        y_pred = predict_y(xc, recent, fallback=y_center)
                        y_alt = pick_centered_run_y(ys_local, y_pred, min_run=2)
                        if y_alt is None:
                            y_alt = int(ys_local[np.argmin(np.abs(ys_local - y_pred))])
                        y_pick2 = expand_full_run_center(mask, xc, int(y_alt))

                    cur_y = int(y_pick2)
                    y_by_x[xc] = int(y_pick2)
                    recent.append((xc, int(y_pick2)))
                    if len(recent) > 6:
                        recent.pop(0)
                    miss_run = 0
                    xc += step
                    continue

                # 3) Full reacquire: slope prediction + x-lookahead
                y_pred = predict_y(xc, recent, fallback=y_center)

                best = None  # (cost, found_x, found_y)
                for xo in range(0, lookahead_px + 1):
                    xq = xc + xo * step
                    if xq < 0 or xq >= W:
                        break

                    half = int(min(band_reacq_px, band_tight_px + 4 * miss_run))
                    ys2 = col_candidates(xq, y_pred, half)
                    if ys2.size == 0:
                        continue

                    yq_pred = pick_centered_run_y(ys2, y_pred, min_run=2)
                    if yq_pred is None:
                        yq_pred = int(ys2[np.argmin(np.abs(ys2 - y_pred))])
                    yq_pred = expand_full_run_center(mask, xq, int(yq_pred))

                    yq_cont = pick_centered_run_y(ys2, y_center, min_run=2)
                    if yq_cont is None:
                        yq_cont = int(ys2[np.argmin(np.abs(ys2 - y_center))])
                    yq_cont = expand_full_run_center(mask, xq, int(yq_cont))

                    cand_list = [("pred", int(yq_pred))]
                    if int(yq_cont) != int(yq_pred):
                        cand_list.append(("cont", int(yq_cont)))

                    for _, yq in cand_list:
                        cost = abs(int(yq) - int(y_pred)) + 0.2 * abs(xq - xc)
                        if best is None or cost < best[0]:
                            best = (cost, xq, int(yq))

                if best is None:
                    y_by_x[xc] = None
                    xc += step
                    continue

                _, found_x, found_y = best
                cur_y = int(found_y)
                y_by_x[found_x] = int(found_y)
                recent.append((found_x, int(found_y)))
                if len(recent) > 6:
                    recent.pop(0)
                miss_run = 0

                xc = found_x + step
                continue

            # Candidate exists in tight band
            miss_run = 0

            if has_multiple_runs(ys) and abs(int(y_pick) - int(cur_y)) > max_jump_px and len(recent) >= 2:
                y_pred = predict_y(xc, recent, fallback=y_center)
                y_pick_alt = pick_centered_run_y(ys, y_pred, min_run=2)
                if y_pick_alt is not None:
                    y_pick_alt = expand_full_run_center(mask, xc, int(y_pick_alt))
                    y_pick = int(y_pick_alt)

            cur_y = int(y_pick)
            y_by_x[xc] = int(y_pick)
            recent.append((xc, int(y_pick)))
            if len(recent) > 6:
                recent.pop(0)

            xc += step

    # --- Build dense trace ---
    cur_y = start_seed[1]
    recent = [(start_seed[0], start_seed[1])]
    y_by_x[start_seed[0]] = start_seed[1]

    if start_seed[0] + 1 < W:
        cur_y = start_seed[1]
        recent = [(start_seed[0], start_seed[1])]
        track_direction(start_seed[0] + 1, W, +1)

    if start_seed[0] - 1 >= 0:
        cur_y = start_seed[1]
        recent = [(start_seed[0], start_seed[1])]
        track_direction(start_seed[0] - 1, -1, -1)

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
        ypx = int(y0 + int(y_local))
        ypx_raw.append(ypx)
        px_points.append((int(xpx), int(ypx)))

    return px_points, ypx_raw


def extract_line_series_simple(
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
    max_jump_px: int = 30,           # jump size that triggers slope-based disambiguation (not a hard reject)
    max_gap_cols: int = 40,          # maximum consecutive missing columns before giving up (will become None)
    lookahead_px: int = 10,          # how far ahead to search in x during reacquire
) -> Tuple[List[Tuple[int,int]], List[Optional[int]]]:
    """
    Seeded continuity-first line extraction.

    - Continuity-first in a tight band.
    - On breaks: try a local reacquire around current y with widening band (handles vertical/jagged).
    - If still missing: slope prediction + widening band + x-lookahead.
    - Slope/jump is used only to DISAMBIGUATE among multiple candidates, never to hard-reject.
    """
    require_cv2()
    import cv2

    x0, y0, x1, y1 = roi
    roi_img = bgr[y0:y1, x0:x1]

    mask = color_distance_mask(roi_img, target_bgr, tol).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)

    H, W = mask.shape[:2]

    # --- seed handling ---
    sx, sy = seed_px
    seeds = [(sx, sy)]
    if extra_seeds_px:
        seeds.extend(list(extra_seeds_px))

    seeds_local: List[Tuple[int,int]] = []
    for px, py in seeds:
        if x0 <= px < x1 and y0 <= py < y1:
            seeds_local.append((px - x0, py - y0))

    if not seeds_local:
        seeds_local = [(W//2, H//2)]

    # With multiple seeds, prefer the one closest to the seed click’s x (not xpx_grid[0]).
    seed_x_local = clamp_x = int(min(max(sx - x0, 0), W - 1)) if (x0 <= sx < x1) else W // 2
    start_seed = min(seeds_local, key=lambda s: abs(s[0] - seed_x_local))
    cur_x, cur_y = start_seed

    def clamp_int(v: int, lo: int, hi: int) -> int:
        return lo if v < lo else hi if v > hi else v

    def col_candidates(xc: int, y_center: int, half: int) -> np.ndarray:
        yl = clamp_int(y_center - half, 0, H-1)
        yh = clamp_int(y_center + half, 0, H-1)
        col = mask[yl:yh+1, xc]
        ys = np.where(col > 0)[0]
        return ys + yl

    def pick_nearest(ys: np.ndarray, y_center: int) -> Optional[int]:
        if ys.size == 0:
            return None
        j = int(np.argmin(np.abs(ys - y_center)))
        return int(ys[j])

    def predict_y(xc: int, recent: List[Tuple[int,int]], fallback: int) -> int:
        if len(recent) >= 2:
            (x2, y2) = recent[-1]
            (x1r, y1r) = recent[-2]
            dx = max(1, abs(x2 - x1r))
            dy = (y2 - y1r) / dx
            return int(round(y2 + dy * (xc - x2)))
        return int(fallback)

    # recent accepted points (ROI-local)
    recent: List[Tuple[int,int]] = [(cur_x, cur_y)]
    y_by_x: List[Optional[int]] = [None] * W

    def track_direction(x_start: int, x_end: int, step: int):
        nonlocal cur_y, recent

        miss_run = 0
        xc = x_start

        def in_bounds(x: int) -> bool:
            return (x < x_end) if step > 0 else (x > x_end)

        while in_bounds(xc):
            y_center = int(cur_y)

            # 1) Normal tracking: tight band around current y
            ys = col_candidates(xc, y_center, band_tight_px)
            y_pick = pick_nearest(ys, y_center)

            if y_pick is None:
                miss_run += 1
                if miss_run > max_gap_cols:
                    y_by_x[xc] = None
                    xc += step
                    continue

                # 2) Local reacquire at SAME xc, widening around current y.
                #    This is what allows near-vertical and jagged sections to survive.
                half_local = int(min(band_reacq_px, band_tight_px + 4 * miss_run))
                ys_local = col_candidates(xc, y_center, half_local)
                if ys_local.size:
                    # continuity-first: nearest to current y
                    y_pick2 = pick_nearest(ys_local, y_center)

                    # If multiple candidates and implied move is big, use slope to disambiguate.
                    if ys_local.size >= 2 and abs(int(y_pick2) - int(cur_y)) > max_jump_px:
                        y_pred = predict_y(xc, recent, fallback=y_center)
                        y_pick2 = int(ys_local[np.argmin(np.abs(ys_local - y_pred))])

                    cur_y = int(y_pick2)
                    y_by_x[xc] = int(y_pick2)
                    recent.append((xc, int(y_pick2)))
                    if len(recent) > 6:
                        recent.pop(0)
                    miss_run = 0
                    xc += step
                    continue

                # 3) Full reacquire: slope prediction + x-lookahead
                y_pred = predict_y(xc, recent, fallback=y_center)

                best = None  # (cost, found_x, found_y)
                for xo in range(0, lookahead_px + 1):
                    xq = xc + xo * step
                    if xq < 0 or xq >= W:
                        break

                    half = int(min(band_reacq_px, band_tight_px + 4 * miss_run))
                    ys2 = col_candidates(xq, y_pred, half)
                    if ys2.size == 0:
                        continue

                    # candidate closest to prediction
                    yq_pred = int(ys2[np.argmin(np.abs(ys2 - y_pred))])

                    # If there are multiple candidates in this column, allow continuity to compete too.
                    # This helps when slope prediction is poor (common when starting on the right).
                    yq_cont = pick_nearest(ys2, y_center)

                    # Pick between pred-based and continuity-based choice by a cost that prefers
                    # matching the predicted trend but does not forbid large moves.
                    cand_list = [("pred", yq_pred)]
                    if yq_cont is not None and yq_cont != yq_pred:
                        cand_list.append(("cont", int(yq_cont)))

                    for _, yq in cand_list:
                        cost = abs(int(yq) - int(y_pred)) + 0.2 * abs(xq - xc)
                        if best is None or cost < best[0]:
                            best = (cost, xq, int(yq))

                if best is None:
                    y_by_x[xc] = None
                    xc += step
                    continue

                _, found_x, found_y = best
                cur_y = int(found_y)
                y_by_x[found_x] = int(found_y)
                recent.append((found_x, int(found_y)))
                if len(recent) > 6:
                    recent.pop(0)
                miss_run = 0

                # CRITICAL: jump to found_x (prevents long flat/interpolated runs)
                xc = found_x + step
                continue

            # Candidate exists in tight band: continuity-first
            miss_run = 0

            # If multiple candidates and implied move is big, use slope to disambiguate.
            if ys.size >= 2 and abs(int(y_pick) - int(cur_y)) > max_jump_px and len(recent) >= 2:
                y_pred = predict_y(xc, recent, fallback=y_center)
                y_pick = int(ys[np.argmin(np.abs(ys - y_pred))])

            cur_y = int(y_pick)
            y_by_x[xc] = int(y_pick)
            recent.append((xc, int(y_pick)))
            if len(recent) > 6:
                recent.pop(0)

            xc += step

    # --- Build dense trace ---
    cur_y = start_seed[1]
    recent = [(start_seed[0], start_seed[1])]
    y_by_x[start_seed[0]] = start_seed[1]

    if start_seed[0] + 1 < W:
        cur_y = start_seed[1]
        recent = [(start_seed[0], start_seed[1])]
        track_direction(start_seed[0] + 1, W, +1)

    if start_seed[0] - 1 >= 0:
        cur_y = start_seed[1]
        recent = [(start_seed[0], start_seed[1])]
        track_direction(start_seed[0] - 1, -1, -1)

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
        ypx = int(y0 + int(y_local))
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

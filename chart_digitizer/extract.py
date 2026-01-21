
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

    # De-duplicate seeds by x; keep the first seen for each x.
    seen_x = set()
    dedup_seeds: List[Tuple[int,int]] = []
    for x_s, y_s in seeds_local:
        if x_s in seen_x:
            continue
        seen_x.add(x_s)
        dedup_seeds.append((int(x_s), int(y_s)))
    seeds_local = dedup_seeds or [(W // 2, H // 2)]

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
    seeds_sorted = sorted(seeds_local, key=lambda s: s[0])
    if len(seeds_sorted) == 1:
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
    else:
        for i, (sx_l, sy_l) in enumerate(seeds_sorted):
            if i == 0:
                left_limit = 0
            else:
                left_limit = int(math.floor((seeds_sorted[i - 1][0] + sx_l) / 2.0)) + 1
            if i == len(seeds_sorted) - 1:
                right_limit = W - 1
            else:
                right_limit = int(math.floor((sx_l + seeds_sorted[i + 1][0]) / 2.0))

            if left_limit < 0:
                left_limit = 0
            if right_limit > (W - 1):
                right_limit = W - 1
            if left_limit > sx_l:
                left_limit = sx_l
            if right_limit < sx_l:
                right_limit = sx_l

            cur_y = int(sy_l)
            recent = [(int(sx_l), int(sy_l))]
            y_by_x[int(sx_l)] = int(sy_l)

            if sx_l + 1 <= right_limit:
                cur_y = int(sy_l)
                recent = [(int(sx_l), int(sy_l))]
                track_direction(int(sx_l) + 1, right_limit + 1, +1)

            if sx_l - 1 >= left_limit:
                cur_y = int(sy_l)
                recent = [(int(sx_l), int(sy_l))]
                track_direction(int(sx_l) - 1, left_limit - 1, -1)

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
    *,
    seed_px: Optional[Tuple[int,int]] = None,
    template_radius_px: int = 12,
    min_area_px: int = 6,
    split_overlaps: bool = True,
    shape_filter: bool = True,
    shape_match_thresh: float = 0.30,
) -> List[Tuple[int,int]]:
    """
    Extract scatter points inside ROI using a color mask, with optional overlap-splitting and
    seed-guided shape filtering.

    Notes
    - Overlaps: connected components can merge adjacent markers. If split_overlaps=True we use a
      distance-transform watershed to separate blobs into individual markers.
    - Shapes: if seed_px is provided and falls in ROI, we build a marker template from a local
      window around the seed and filter candidate components by cv2.matchShapes() similarity.
      This improves robustness against text/line fragments and supports non-filled marker styles
      (e.g., X/O) as long as the marker stroke color is within the mask.

    Returns: list of (xpx, ypx) centroids in image pixel coordinates (sorted by x).
    """
    require_cv2()
    import cv2

    x0, y0, x1, y1 = roi
    roi_img = bgr[y0:y1, x0:x1]

    # Base color mask
    mask = color_distance_mask(roi_img, target_bgr, tol).astype(np.uint8) * 255

    # Light cleanup: remove isolated specks while keeping thin strokes.
    mask = cv2.medianBlur(mask, 3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)

    H, W = mask.shape[:2]

    # ---------------- Seed template (optional) ----------------
    tmpl_contour = None
    if seed_px is not None:
        sx, sy = seed_px
        if (x0 <= sx < x1) and (y0 <= sy < y1):
            sxl, syl = int(sx - x0), int(sy - y0)
            r = int(max(4, template_radius_px))
            xl = max(0, sxl - r); xr = min(W, sxl + r + 1)
            yl = max(0, syl - r); yr = min(H, syl + r + 1)
            win = mask[yl:yr, xl:xr]
            if win.size:
                # Keep the largest connected component in the window (seed marker should dominate).
                nlab, labs, stats, _ = cv2.connectedComponentsWithStats((win > 0).astype(np.uint8), connectivity=8)
                if nlab > 1:
                    areas = stats[1:, cv2.CC_STAT_AREA]
                    k = int(np.argmax(areas)) + 1
                    comp = (labs == k).astype(np.uint8) * 255
                    cnts, _hier = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if cnts:
                        tmpl_contour = max(cnts, key=cv2.contourArea)

    # ---------------- Overlap splitting (optional) ----------------
    labels = None
    if split_overlaps:
        # Distance transform of foreground; peaks act as marker seeds.
        fg = (mask > 0).astype(np.uint8)
        if fg.any():
            dist = cv2.distanceTransform(fg, cv2.DIST_L2, 5)

            # Peak threshold: adaptive, but not too aggressive.
            dmax = float(dist.max())
            if dmax > 0:
                peak_thresh = max(1.0, 0.45 * dmax)
                peaks = (dist >= peak_thresh).astype(np.uint8) * 255
                peaks = cv2.dilate(peaks, np.ones((3,3), np.uint8), iterations=1)

                nmk, markers = cv2.connectedComponents(peaks)
                # If we only have one marker (or none), watershed won't help—fallback to CCs below.
                if nmk > 1:
                    # sure background
                    sure_bg = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=2)
                    unknown = cv2.subtract(sure_bg, peaks)

                    markers = markers.astype(np.int32) + 1
                    markers[unknown > 0] = 0

                    # watershed expects 3-channel image
                    ws_img = roi_img.copy()
                    markers = cv2.watershed(ws_img, markers)

                    # markers: -1 boundary, 1 background, 2.. = objects
                    labels = markers.copy()

    pts: List[Tuple[int,int]] = []

    def _accept_contour(cnt) -> bool:
        if cnt is None:
            return True
        if tmpl_contour is None or not shape_filter:
            return True
        try:
            score = float(cv2.matchShapes(tmpl_contour, cnt, cv2.CONTOURS_MATCH_I1, 0.0))
        except Exception:
            return True
        return score <= float(shape_match_thresh)

    if labels is not None:
        # Parse watershed labels (2..)
        for lab in range(2, int(labels.max()) + 1):
            comp = (labels == lab).astype(np.uint8) * 255
            area = int(cv2.countNonZero(comp))
            if area < min_area_px:
                continue
            cnts, _hier = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            cnt = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(cnt) < min_area_px:
                continue
            if not _accept_contour(cnt):
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            pts.append((x0 + cx, y0 + cy))
    else:
        # Connected components fallback
        nlab, labs, stats, cents = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
        for lab in range(1, nlab):
            area = int(stats[lab, cv2.CC_STAT_AREA])
            if area < min_area_px:
                continue
            comp = (labs == lab).astype(np.uint8) * 255
            cnts, _hier = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnt = max(cnts, key=cv2.contourArea) if cnts else None
            if cnt is not None and cv2.contourArea(cnt) < min_area_px:
                continue
            if not _accept_contour(cnt):
                continue
            cx, cy = cents[lab]
            pts.append((x0 + int(round(cx)), y0 + int(round(cy))))

    # Stable order: by x then y
    pts.sort(key=lambda p: (p[0], p[1]))
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


# -----------------------------
# Column / Bar / Area extraction
# -----------------------------

def _edge_mask_4n(mask: np.ndarray) -> np.ndarray:
    """Return a thin edge/outline mask derived from a binary mask (uint8/bool)."""
    m = (mask > 0)
    # 4-neighbor erosion (interior pixels only)
    up = np.zeros_like(m); up[1:,:] = m[:-1,:]
    dn = np.zeros_like(m); dn[:-1,:] = m[1:,:]
    lf = np.zeros_like(m); lf[:,1:] = m[:,:-1]
    rt = np.zeros_like(m); rt[:,:-1] = m[:,1:]
    er = m & up & dn & lf & rt
    edge = m & (~er)
    return edge.astype(np.uint8) * 255

def _scan_column_for_edge(
    mask: np.ndarray,
    x_local: int,
    *,
    y0: int,
    y1: int,
    y_base_local: int,
    prefer_top: bool,
    max_gap_px: int = 2,
) -> Optional[int]:
    """
    Scan a single x-column in mask[y0:y1, x_local] and return the OUTER boundary y (local coords),
    scanning away from a baseline with short gap tolerance.

    This is robust to combo charts where other series (lines) overwrite a few pixels inside the bar.
    """
    # Column slice in ROI-local coordinates
    col = (mask[y0:y1, x_local] > 0)
    H = int(col.shape[0])
    if H <= 0 or (not np.any(col)):
        return None

    # Clamp baseline to this column's local slice coordinates
    yb = int(y_base_local)
    if yb < y0: yb = y0
    if yb > (y1 - 1): yb = y1 - 1
    yb_rel = yb - y0  # baseline relative to col[0]

    direction = -1 if prefer_top else +1
    y = int(yb_rel)
    y_end = 0 if direction < 0 else (H - 1)

    best: Optional[int] = None
    gap = 0

    # Walk outward; keep the farthest ON pixel, tolerate small OFF gaps.
    while True:
        if col[y]:
            best = y
            gap = 0
        else:
            if best is not None:
                gap += 1
                if gap > int(max_gap_px):
                    break

        if y == y_end:
            break
        y_next = y + direction
        if y_next < 0 or y_next >= H:
            break
        y = y_next

    if best is None:
        return None

    # Convert back to ROI-local mask coordinates (not absolute image coords)
    return int(y0 + best)


def _scan_column_for_edge_simple(
    mask: np.ndarray,
    x_local: int,
    *,
    y0: int,
    y1: int,
    prefer_top: bool,
    y_hint: Optional[int] = None,
    band: int = 8,
) -> Optional[int]:
    """
    Scan a single x-column in mask[y0:y1, x_local] and return an edge y (in local coords).
    If y_hint is provided, prefer a tight band around y_hint; fall back to full scan.
    """
    col = mask[y0:y1, x_local] > 0
    if not np.any(col):
        return None

    # Tight band first if we have a hint
    if y_hint is not None:
        lo = max(0, (y_hint - y0) - band)
        hi = min((y1 - y0), (y_hint - y0) + band + 1)
        band_col = col[lo:hi]
        if np.any(band_col):
            idx = np.where(band_col)[0]
            return int((lo + (idx[0] if prefer_top else idx[-1])))

    idx = np.where(col)[0]
    return int(idx[0] if prefer_top else idx[-1])


def extract_filled_boundary_vertical(
    bgr: np.ndarray,
    roi: Tuple[int,int,int,int],
    target_bgr: Tuple[int,int,int],
    tol: int,
    xpx_grid: List[int],
    *,
    seed_px: Tuple[int,int],
    baseline_y_px: Optional[int] = None,
    prefer_outline: bool = True,
    band_px: int = 10,
) -> Tuple[List[Tuple[int,int]], List[Optional[int]]]:
    """
    Extract the outer boundary (top for positive, bottom for negative) of a filled region
    for each xpx in xpx_grid (column/area charts).
    """
    require_cv2()
    x0, y0, x1, y1 = roi
    mask = color_distance_mask(bgr, roi, target_bgr, tol)
    if prefer_outline:
        mask = _edge_mask_4n(mask)

    sx, sy = seed_px
    if baseline_y_px is None:
        baseline_y_px = sy

    prefer_top = (sy < baseline_y_px)  # positive-ish: boundary is towards smaller y

    ypx_raw: List[Optional[int]] = []
    px_points: List[Tuple[int,int]] = []

    y_hint: Optional[int] = sy
    for xpx in xpx_grid:
        if xpx < x0 or xpx >= x1:
            ypx_raw.append(None)
            continue
        xc = int(xpx - x0)

    # Baseline in ROI-local mask coords (same coordinate system as y_local returned)
    y_base_local = int(baseline_y_px - y0)
    if y_base_local < 0: y_base_local = 0
    if y_base_local > (y1 - y0 - 1): y_base_local = (y1 - y0 - 1)

    for xpx in xpx_grid:
        if xpx < x0 or xpx >= x1:
            ypx_raw.append(None)
            continue
        xc = int(xpx - x0)

        y_local = _scan_column_for_edge(
            mask, xc,
            y0=0, y1=(y1 - y0),
            y_base_local=y_base_local,
            prefer_top=prefer_top,
            max_gap_px=2,   # <- increase to 3 if line strokes are thick/anti-aliased
        )
        if y_local is None:
            ypx_raw.append(None)
            continue

        ypx = int(y0 + y_local)
        ypx_raw.append(ypx)
        px_points.append((int(xpx), int(ypx)))
        # y_hint = ypx

    return px_points, ypx_raw


def _scan_row_for_edge(
    mask: np.ndarray,
    y_local: int,
    *,
    x0: int,
    x1: int,
    prefer_left: bool,
    x_hint: Optional[int] = None,
    band: int = 8,
) -> Optional[int]:
    row = mask[y_local, x0:x1] > 0
    if not np.any(row):
        return None

    if x_hint is not None:
        lo = max(0, (x_hint - x0) - band)
        hi = min((x1 - x0), (x_hint - x0) + band + 1)
        band_row = row[lo:hi]
        if np.any(band_row):
            idx = np.where(band_row)[0]
            return int((lo + (idx[0] if prefer_left else idx[-1])))

    idx = np.where(row)[0]
    return int(idx[0] if prefer_left else idx[-1])


def extract_filled_boundary_horizontal(
    bgr: np.ndarray,
    roi: Tuple[int,int,int,int],
    target_bgr: Tuple[int,int,int],
    tol: int,
    ypx_grid: List[int],
    *,
    seed_px: Tuple[int,int],
    baseline_x_px: Optional[int] = None,
    prefer_outline: bool = True,
    band_px: int = 10,
) -> Tuple[List[Tuple[int,int]], List[Optional[int]]]:
    """Extract the outer boundary (right for positive, left for negative) for horizontal bars."""
    require_cv2()
    x0, y0, x1, y1 = roi
    mask = color_distance_mask(bgr, roi, target_bgr, tol)
    if prefer_outline:
        mask = _edge_mask_4n(mask)

    sx, sy = seed_px
    if baseline_x_px is None:
        baseline_x_px = sx

    prefer_right = (sx > baseline_x_px)

    xpx_raw: List[Optional[int]] = []
    px_points: List[Tuple[int,int]] = []

    x_hint: Optional[int] = sx
    for ypx in ypx_grid:
        if ypx < y0 or ypx >= y1:
            xpx_raw.append(None)
            continue
        yc = int(ypx - y0)

        x_local = _scan_row_for_edge(
            mask, yc,
            x0=0, x1=(x1 - x0),
            prefer_left=(not prefer_right),
            x_hint=(x_hint - x0) if (x_hint is not None) else None,
            band=band_px,
        )
        if x_local is None:
            xpx_raw.append(None)
            continue

        xpx = int(x0 + x_local)
        xpx_raw.append(xpx)
        px_points.append((int(xpx), int(ypx)))
        x_hint = xpx

    return px_points, xpx_raw


def detect_runs_centers_along_x(mask: np.ndarray, *, min_run_px: int = 2) -> List[int]:
    """Detect contiguous x-runs where mask has any pixels, and return run centers (local x)."""
    m = (mask > 0)
    proj = np.any(m, axis=0).astype(np.uint8)
    centers: List[int] = []
    x = 0
    n = proj.shape[0]
    while x < n:
        if proj[x] == 0:
            x += 1
            continue
        x0 = x
        while x < n and proj[x] != 0:
            x += 1
        x1 = x
        if (x1 - x0) >= min_run_px:
            centers.append(int(round((x0 + x1 - 1) / 2.0)))
    return centers


def detect_runs_centers_along_y(mask: np.ndarray, *, min_run_px: int = 2) -> List[int]:
    """Detect contiguous y-runs where mask has any pixels, and return run centers (local y)."""
    m = (mask > 0)
    proj = np.any(m, axis=1).astype(np.uint8)
    centers: List[int] = []
    y = 0
    n = proj.shape[0]
    while y < n:
        if proj[y] == 0:
            y += 1
            continue
        y0 = y
        while y < n and proj[y] != 0:
            y += 1
        y1 = y
        if (y1 - y0) >= min_run_px:
            centers.append(int(round((y0 + y1 - 1) / 2.0)))
    return centers

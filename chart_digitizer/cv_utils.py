
from __future__ import annotations

from typing import Tuple
import numpy as np

def require_cv2():
    try:
        import cv2  # noqa
    except Exception as e:
        raise RuntimeError(
            "OpenCV (cv2) is required for chart digitizing."
            "Install with:"
            "  pip install opencv-python"
        ) from e

def pil_to_bgr(pil_img) -> np.ndarray:
    """
    Convert a PIL RGB image to an OpenCV-style BGR uint8 ndarray.
    This must be unambiguous: PIL->numpy is RGB; OpenCV convention is BGR.
    """
    arr = np.array(pil_img.convert("RGB"), dtype=np.uint8)  # (H,W,3) RGB
    # swap channels to BGR
    bgr = arr[..., ::-1].copy()
    return bgr

def bgr_to_rgb(bgr: "np.ndarray") -> "np.ndarray":
    return bgr[:, :, ::-1]

def color_distance_mask(bgr: np.ndarray, target_bgr: Tuple[int,int,int], tol: int) -> np.ndarray:
    """
    Return a boolean mask where pixels are within tol (0..255) of target_bgr
    using per-channel max absolute difference:

        max(|ΔB|,|ΔG|,|ΔR|) <= tol

    This is deterministic and tol is easy to interpret.
    """
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("bgr must be HxWx3")

    tol = int(tol)
    if tol < 0:
        tol = 0

    # CRITICAL: convert to signed type before subtracting to avoid uint8 wraparound.
    img = bgr.astype(np.int16, copy=False)
    tb, tg, tr = (int(target_bgr[0]), int(target_bgr[1]), int(target_bgr[2]))
    target = np.array([tb, tg, tr], dtype=np.int16)

    diff = np.abs(img - target)                # HxWx3 int16
    d = diff.max(axis=2)                       # HxW
    return d <= tol
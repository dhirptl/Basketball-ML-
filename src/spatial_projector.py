"""Map image foot points to court coordinates (feet)."""

from __future__ import annotations

import cv2
import numpy as np

from . import config


def bbox_bottom_center(xyxy: np.ndarray) -> tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return float((x1 + x2) / 2.0), float(y2)


def image_to_court_ft(
    H_image_to_court: np.ndarray, u: float, v: float
) -> tuple[float, float] | None:
    """Apply homography mapping image (u,v) to court plane (x_ft, y_ft)."""
    pts = np.array([[[u, v]]], dtype=np.float64)
    mapped = cv2.perspectiveTransform(pts, H_image_to_court.astype(np.float64))
    x, y = float(mapped[0, 0, 0]), float(mapped[0, 0, 1])
    if not np.isfinite(x) or not np.isfinite(y):
        return None
    return x, y


def maybe_canonicalize_xy(x: float, y: float) -> tuple[float, float]:
    if not config.CANONICALIZE_COURT_DIRECTION:
        return x, y
    # Example: always map so x increases toward far baseline from home end
    return float(config.COURT_LENGTH_FT - x), float(config.COURT_WIDTH_FT - y)

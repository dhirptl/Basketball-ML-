"""Broadcast-frame helpers: HUD zero-mask, normalized court ROI, foot-point tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


def apply_hud_mask_bottom(frame_bgr: np.ndarray, bottom_fraction: float) -> None:
    """Zero-fill the bottom ``bottom_fraction`` of the image (height axis). In-place."""
    if bottom_fraction <= 0.0:
        return
    h, _w = frame_bgr.shape[:2]
    y0 = int(round(h * (1.0 - float(bottom_fraction))))
    y0 = max(0, min(h, y0))
    frame_bgr[y0:h, :, :] = 0


def normalized_polygon_to_pixels(
    polygon_norm: Sequence[tuple[float, float]],
    width: int,
    height: int,
) -> np.ndarray:
    """Map normalized (0–1) vertices to pixel coordinates. Shape (N, 1, 2) int32 for OpenCV."""
    pts: list[list[int]] = []
    for xn, yn in polygon_norm:
        u = int(round(float(xn) * width))
        v = int(round(float(yn) * height))
        pts.append([u, v])
    return np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)


def foot_inside_polygon(xyxy: np.ndarray, polygon_px: np.ndarray) -> bool:
    """True if bbox bottom-center lies inside ``polygon_px`` (OpenCV contour)."""
    x1, y1, x2, y2 = (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]))
    fx = 0.5 * (x1 + x2)
    fy = y2
    r = cv2.pointPolygonTest(polygon_px, (fx, fy), measureDist=False)
    return r >= 0.0


def bbox_iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """IoU for axis-aligned boxes (xyxy)."""
    ax1, ay1, ax2, ay2 = map(float, a.tolist())
    bx1, by1, bx2, by2 = map(float, b.tolist())
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ba = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + ba - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def load_polygon_norm_from_json(path: Path | str) -> list[tuple[float, float]]:
    """Load ``polygon_norm`` from a JSON file. Schema: ``{\"polygon_norm\": [[x,y], ...]}``."""
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    poly = raw.get("polygon_norm") or raw.get("polygon")
    if not poly or len(poly) < 3:
        return []
    out: list[tuple[float, float]] = []
    for pair in poly:
        if len(pair) < 2:
            continue
        out.append((float(pair[0]), float(pair[1])))
    return out

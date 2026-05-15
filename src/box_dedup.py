"""Post-NMS foot-point deduplication for overlapping same-player hypotheses."""

from __future__ import annotations

import numpy as np

from . import config
from .spatial_projector import bbox_bottom_center


def foot_point(xyxy: np.ndarray) -> np.ndarray:
    fx, fy = bbox_bottom_center(xyxy)
    return np.array([fx, fy], dtype=np.float64)


def dedup_by_foot(
    boxes_xyxy: np.ndarray,
    confs: np.ndarray,
    foot_dist_thresh_px: float | None = None,
) -> np.ndarray:
    if foot_dist_thresh_px is None:
        foot_dist_thresh_px = float(config.FOOT_DEDUP_DIST_PX)
    n = len(boxes_xyxy)
    if n <= 1:
        return np.ones(n, dtype=bool)
    order = np.argsort(-confs.astype(np.float64))
    keep_mask = np.ones(n, dtype=bool)
    for i in range(n):
        idx_i = int(order[i])
        if not keep_mask[idx_i]:
            continue
        fi = foot_point(boxes_xyxy[idx_i])
        for j in range(i + 1, n):
            idx_j = int(order[j])
            if not keep_mask[idx_j]:
                continue
            fj = foot_point(boxes_xyxy[idx_j])
            if float(np.linalg.norm(fi - fj)) < foot_dist_thresh_px:
                keep_mask[idx_j] = False
    return keep_mask


def filter_by_foot_dedup(
    xyxy: np.ndarray,
    confs: np.ndarray,
    ids: np.ndarray | None,
    foot_dist_thresh_px: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if len(xyxy) < 2:
        return xyxy, confs, ids
    keep = dedup_by_foot(xyxy, confs, foot_dist_thresh_px)
    ids_f = ids[keep] if ids is not None else None
    return xyxy[keep], confs[keep], ids_f


if __name__ == "__main__":
    boxes = np.array([[100.0, 200.0, 150.0, 350.0], [105.0, 205.0, 155.0, 360.0]], dtype=np.float64)
    confs = np.array([0.90, 0.75], dtype=np.float64)
    keep = dedup_by_foot(boxes, confs, foot_dist_thresh_px=30.0)
    assert int(keep.sum()) == 1 and bool(keep[0])
    boxes2 = np.array([[100.0, 200.0, 150.0, 350.0], [200.0, 200.0, 250.0, 350.0]], dtype=np.float64)
    confs2 = np.array([0.85, 0.88], dtype=np.float64)
    assert int(dedup_by_foot(boxes2, confs2, 30.0).sum()) == 2
    print("box_dedup tests passed.")

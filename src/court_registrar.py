"""Court registration: YOLO pose -> filtered image keypoints."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from ultralytics import YOLO

from . import config


@dataclass
class CourtKeypoints:
    """Pixel-space court keypoints (image size)."""

    xy: np.ndarray  # (K, 2) float32, NaN if missing
    conf: np.ndarray  # (K,) float32
    visible_mask: np.ndarray  # (K,) bool — passes confidence threshold


class CourtRegistrar:
    def __init__(self, weights_path: str, device: str) -> None:
        self.model = YOLO(weights_path)
        self.device = device

    @torch.inference_mode()
    def infer(self, frame_bgr: np.ndarray) -> CourtKeypoints:
        h, w = frame_bgr.shape[:2]
        results = self.model(frame_bgr, verbose=False, device=self.device)
        k = config.NUM_COURT_KEYPOINTS
        xy = np.full((k, 2), np.nan, dtype=np.float32)
        conf = np.zeros((k,), dtype=np.float32)

        if not results:
            return CourtKeypoints(xy=xy, conf=conf, visible_mask=np.zeros(k, dtype=bool))

        r0 = results[0]
        if r0.keypoints is None or r0.keypoints.data is None:
            return CourtKeypoints(xy=xy, conf=conf, visible_mask=np.zeros(k, dtype=bool))

        kpts = r0.keypoints
        if kpts.xy is None or kpts.xy.numel() == 0:
            return CourtKeypoints(xy=xy, conf=conf, visible_mask=np.zeros(k, dtype=bool))

        xy_all = kpts.xy.detach().cpu().numpy().astype(np.float32)
        n_det = xy_all.shape[0]
        if n_det == 0:
            return CourtKeypoints(xy=xy, conf=conf, visible_mask=np.zeros(k, dtype=bool))

        if kpts.conf is not None and kpts.conf.numel() > 0:
            cf_all = kpts.conf.detach().cpu().numpy().astype(np.float32)
            pick = int(np.argmax(np.mean(cf_all, axis=1)))
        else:
            pick = 0

        xy_t = xy_all[pick]
        if xy_t.shape[0] < k:
            return CourtKeypoints(xy=xy, conf=conf, visible_mask=np.zeros(k, dtype=bool))

        xy[: xy_t.shape[0]] = xy_t[:k]
        if kpts.conf is not None and kpts.conf.numel() > 0:
            cf = kpts.conf[pick].detach().cpu().numpy().astype(np.float32)
            conf[: cf.shape[0]] = cf[:k]

        inside = (
            np.isfinite(xy[:, 0])
            & np.isfinite(xy[:, 1])
            & (xy[:, 0] >= 0)
            & (xy[:, 0] < w)
            & (xy[:, 1] >= 0)
            & (xy[:, 1] < h)
        )
        vis = inside & (conf >= config.KPT_CONF_THRESHOLD)
        return CourtKeypoints(xy=xy, conf=conf, visible_mask=vis)

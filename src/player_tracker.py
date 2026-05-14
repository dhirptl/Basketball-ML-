"""Player detection + ByteTrack via Ultralytics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from . import broadcast_preprocess as bp
from . import config


@dataclass
class TrackedPlayer:
    track_id: int
    xyxy: np.ndarray  # (4,) float32
    conf: float


def _greedy_merge_by_iou(players: list[TrackedPlayer], merge_iou: float) -> list[TrackedPlayer]:
    if merge_iou <= 0.0 or len(players) <= 1:
        return list(players)
    n = len(players)
    order = sorted(range(n), key=lambda i: -players[i].conf)
    suppressed = [False] * n
    out: list[TrackedPlayer] = []
    for i in order:
        if suppressed[i]:
            continue
        out.append(players[i])
        for j in range(n):
            if j == i or suppressed[j]:
                continue
            if bp.bbox_iou_xyxy(players[i].xyxy, players[j].xyxy) >= merge_iou:
                suppressed[j] = True
    return out


def _dedup_different_track_ids(players: list[TrackedPlayer], dedup_iou: float) -> list[TrackedPlayer]:
    if dedup_iou <= 0.0 or len(players) <= 1:
        return list(players)
    pl_sorted = sorted(players, key=lambda p: -p.conf)
    out: list[TrackedPlayer] = []
    for p in pl_sorted:
        drop = False
        for q in out:
            if p.track_id < 0 or q.track_id < 0 or p.track_id == q.track_id:
                continue
            if bp.bbox_iou_xyxy(p.xyxy, q.xyxy) >= dedup_iou:
                drop = True
                break
        if not drop:
            out.append(p)
    return out


class PlayerTracker:
    def __init__(self, weights_path: str, device: str) -> None:
        self.model = YOLO(weights_path)
        self.device = device
        self._roi_poly_norm = self._resolve_roi_polygon_norm()

    def _resolve_roi_polygon_norm(self) -> list[tuple[float, float]]:
        jp = config.COURT_ROI_JSON_PATH
        if jp:
            p = Path(jp)
            if p.is_file():
                loaded = bp.load_polygon_norm_from_json(p)
                if len(loaded) >= 3:
                    return loaded
        return list(config.COURT_ROI_POLYGON_NORM)

    @torch.inference_mode()
    def track(
        self,
        frame_bgr: np.ndarray,
        *,
        persist: bool = True,
        reset_ids: bool = False,
    ) -> list[TrackedPlayer]:
        if reset_ids:
            persist = False

        h, w = frame_bgr.shape[:2]
        frame_for = frame_bgr.copy()
        if config.HUD_MASK_BOTTOM_PCT > 0.0:
            bp.apply_hud_mask_bottom(frame_for, config.HUD_MASK_BOTTOM_PCT)

        polygon_px = None
        if len(self._roi_poly_norm) >= 3:
            polygon_px = bp.normalized_polygon_to_pixels(self._roi_poly_norm, w, h)

        results = self.model.track(
            frame_for,
            persist=persist,
            verbose=False,
            device=self.device,
            tracker=config.TRACKER_CFG,
            classes=[config.PLAYER_CLASS_ID],
            conf=config.PLAYER_PREDICT_CONF,
            iou=config.PLAYER_PREDICT_IOU,
            max_det=config.PLAYER_PREDICT_MAX_DET,
        )
        out: list[TrackedPlayer] = []
        if not results:
            return out
        r0 = results[0]
        if r0.boxes is None or r0.boxes.xyxy is None:
            return out
        xyxy = r0.boxes.xyxy.detach().cpu().numpy()
        confs = r0.boxes.conf.detach().cpu().numpy() if r0.boxes.conf is not None else np.ones(len(xyxy))
        ids = r0.boxes.id
        if ids is None:
            for i in range(len(xyxy)):
                out.append(
                    TrackedPlayer(
                        track_id=-1,
                        xyxy=xyxy[i].astype(np.float32),
                        conf=float(confs[i]),
                    )
                )
        else:
            id_arr = ids.detach().cpu().numpy().astype(np.int64)
            for i in range(len(xyxy)):
                out.append(
                    TrackedPlayer(
                        track_id=int(id_arr[i]),
                        xyxy=xyxy[i].astype(np.float32),
                        conf=float(confs[i]),
                    )
                )

        if polygon_px is not None:
            out = [p for p in out if bp.foot_inside_polygon(p.xyxy, polygon_px)]

        if config.PLAYER_GREEDY_MERGE_IOU > 0.0:
            out = _greedy_merge_by_iou(out, config.PLAYER_GREEDY_MERGE_IOU)
        if config.PLAYER_TRACK_FRAME_DEDUP_IOU > 0.0:
            out = _dedup_different_track_ids(out, config.PLAYER_TRACK_FRAME_DEDUP_IOU)

        return out

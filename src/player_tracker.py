"""Player detection + ByteTrack via Ultralytics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from ultralytics import YOLO

from . import config


@dataclass
class TrackedPlayer:
    track_id: int
    xyxy: np.ndarray  # (4,) float32
    conf: float


class PlayerTracker:
    def __init__(self, weights_path: str, device: str) -> None:
        self.model = YOLO(weights_path)
        self.device = device

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
        results = self.model.track(
            frame_bgr,
            persist=persist,
            verbose=False,
            device=self.device,
            tracker=config.TRACKER_CFG,
            classes=[config.PLAYER_CLASS_ID],
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
            return out
        id_arr = ids.detach().cpu().numpy().astype(np.int64)
        for i in range(len(xyxy)):
            out.append(
                TrackedPlayer(
                    track_id=int(id_arr[i]),
                    xyxy=xyxy[i].astype(np.float32),
                    conf=float(confs[i]),
                )
            )
        return out

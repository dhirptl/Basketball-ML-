#!/usr/bin/env python3
"""ByteTrack video export: model.track() + clean boxes (no labels). Optional per-track EMA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from . import config
from .device_utils import resolve_device


def _color_for_track(tid: int) -> tuple[int, int, int]:
    rng = np.random.RandomState(tid % 10000 + (tid // 10000) * 17)
    return int(rng.randint(64, 256)), int(rng.randint(64, 256)), int(rng.randint(64, 256))


def _clamp_ema_alpha(a: float) -> float:
    if a >= 1.0:
        return 1.0
    return max(0.2, min(0.999, a))


def _draw_boxes_manual(
    frame_bgr: np.ndarray,
    tracks: list[tuple[int, np.ndarray]],
    line_width: int,
) -> np.ndarray:
    vis = frame_bgr.copy()
    for tid, box in tracks:
        x1, y1, x2, y2 = map(int, box.tolist())
        col = _color_for_track(tid)
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, line_width)
    return vis


@torch.inference_mode()
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Export tracked video: ByteTrack (model.track) + no label text; optional EMA on boxes"
    )
    p.add_argument("--model", type=str, required=True, help="Weights .pt path")
    p.add_argument("--source", type=str, required=True, help="Input video path")
    p.add_argument(
        "--out",
        type=str,
        default="tracked_output.mp4",
        help="Output video path (default: tracked_output.mp4 in cwd)",
    )
    p.add_argument("--device", type=str, default="auto", help="auto | mps | cuda | cpu")
    p.add_argument("--conf", type=float, default=None, help=f"Override conf (default {config.PLAYER_PREDICT_CONF})")
    p.add_argument("--iou", type=float, default=None, help=f"Override NMS iou (default {config.PLAYER_PREDICT_IOU})")
    p.add_argument("--max-det", type=int, default=None, help=f"Override max_det (default {config.PLAYER_PREDICT_MAX_DET})")
    p.add_argument(
        "--ema-alpha",
        type=float,
        default=None,
        help=f"Override PLAYER_BOX_EMA_ALPHA (default {config.PLAYER_BOX_EMA_ALPHA}; 1.0 = off)",
    )
    args = p.parse_args(argv)

    model_path = Path(args.model)
    if not model_path.is_file():
        print(f"Model not found: {model_path}", file=sys.stderr)
        return 1
    src_path = Path(args.source)
    if not src_path.is_file():
        print(f"Source video not found: {src_path}", file=sys.stderr)
        return 1

    device = resolve_device(args.device)
    conf = float(config.PLAYER_PREDICT_CONF if args.conf is None else args.conf)
    iou = float(config.PLAYER_PREDICT_IOU if args.iou is None else args.iou)
    max_det = int(config.PLAYER_PREDICT_MAX_DET if args.max_det is None else args.max_det)
    ema_alpha = float(config.PLAYER_BOX_EMA_ALPHA if args.ema_alpha is None else args.ema_alpha)
    ema_alpha = _clamp_ema_alpha(ema_alpha)
    use_ema = ema_alpha < 1.0
    lw = int(config.PLAYER_BOX_LINE_WIDTH)
    stale_n = int(config.PLAYER_BOX_EMA_STALE_FRAMES)

    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        print(f"Could not open video: {src_path}", file=sys.stderr)
        return 1

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if w <= 0 or h <= 0:
        ok, probe = cap.read()
        if not ok or probe is None:
            print("Could not read video dimensions", file=sys.stderr)
            cap.release()
            return 1
        h, w = probe.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    model = YOLO(str(model_path))
    ema_prev: dict[int, np.ndarray] = {}
    last_seen: dict[int, int] = {}
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        results = model.track(
            frame,
            persist=True,
            verbose=False,
            device=device,
            tracker=config.TRACKER_CFG,
            classes=[config.PLAYER_CLASS_ID],
            conf=conf,
            iou=iou,
            max_det=max_det,
        )
        if not results:
            writer.write(frame)
            frame_idx += 1
            continue

        r0 = results[0]
        if not use_ema or r0.boxes is None or r0.boxes.xyxy is None or r0.boxes.id is None:
            annotated = r0.plot(
                conf=False,
                labels=False,
                boxes=True,
                line_width=int(lw),
                img=frame,
            )
            writer.write(annotated)
        else:
            xyxy = r0.boxes.xyxy.detach().cpu().numpy()
            ids = r0.boxes.id.detach().cpu().numpy().astype(np.int64)
            seen_this: set[int] = set()
            tracks_draw: list[tuple[int, np.ndarray]] = []
            for i in range(len(xyxy)):
                tid = int(ids[i])
                raw = xyxy[i].astype(np.float64)
                seen_this.add(tid)
                prev = ema_prev.get(tid)
                if prev is None:
                    sm = raw.copy()
                else:
                    sm = ema_alpha * raw + (1.0 - ema_alpha) * prev
                ema_prev[tid] = sm
                last_seen[tid] = frame_idx
                tracks_draw.append((tid, sm.astype(np.float32)))

            for tid in list(ema_prev.keys()):
                if tid not in seen_this and frame_idx - last_seen.get(tid, -10**9) > stale_n:
                    del ema_prev[tid]
                    last_seen.pop(tid, None)

            vis = _draw_boxes_manual(frame, tracks_draw, max(1, lw))
            writer.write(vis)

        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Wrote {out_path.resolve()} ({frame_idx} frames, ema={'on' if use_ema else 'off'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""ByteTrack export: foot dedup, v3.2 team colours (ROI + quality warm-up), optional EMA and debug overlays."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from . import box_dedup
from . import config
from .device_utils import resolve_device
from .team_classifier import TeamClassifier, draw_four_slice_debug_lines


def _color_for_track(tid: int) -> tuple[int, int, int]:
    rng = np.random.RandomState(tid % 10000 + (tid // 10000) * 17)
    return int(rng.randint(64, 256)), int(rng.randint(64, 256)), int(rng.randint(64, 256))


def _clamp_ema_alpha(a: float) -> float:
    if a >= 1.0:
        return 1.0
    return max(0.2, min(0.999, a))


def _draw_warmup_overlay(vis: np.ndarray, team_clf: TeamClassifier) -> None:
    remaining = team_clf.warmup_frames_remaining
    cv2.putText(
        vis,
        f"Calibrating... {remaining} frames remaining",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        tuple(int(c) for c in config.UNKNOWN_TEAM_COLOUR),
        2,
        cv2.LINE_AA,
    )


def _draw_boxes_manual(
    frame_bgr: np.ndarray,
    tracks: list[tuple[int, np.ndarray]],
    line_width: int,
    team_clf: TeamClassifier | None = None,
    debug_slices: bool = False,
) -> np.ndarray:
    vis = frame_bgr.copy()
    for tid, box in tracks:
        x1, y1, x2, y2 = map(int, box.tolist())
        if team_clf is not None:
            col = team_clf.get_colour(tid)
        else:
            col = _color_for_track(tid)
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, line_width)
        if debug_slices:
            draw_four_slice_debug_lines(vis, x1, y1, x2, y2)
        if team_clf is not None and bool(config.TEAM_DRAW_TRACK_ID):
            cv2.putText(
                vis,
                str(tid),
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                col,
                1,
                cv2.LINE_AA,
            )
    if team_clf is not None and not team_clf.is_locked:
        _draw_warmup_overlay(vis, team_clf)
    return vis


@torch.inference_mode()
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="ByteTrack video export with warm-up team-colour boxes")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--source", type=str, required=True)
    p.add_argument("--out", type=str, default="tracked_output.mp4")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--conf", type=float, default=None)
    p.add_argument("--iou", type=float, default=None)
    p.add_argument("--max-det", type=int, default=None)
    p.add_argument("--ema-alpha", type=float, default=None)
    p.add_argument("--debug-slices", action="store_true")
    p.add_argument(
        "--debug-team",
        action="store_true",
        help="Print phase and per-track centroid distances every 30 frames",
    )
    p.add_argument(
        "--warmup-frames",
        type=int,
        default=None,
        help=f"Override WARMUP_FRAMES (default {config.WARMUP_FRAMES})",
    )
    p.add_argument(
        "--debug-team-crops",
        type=str,
        default=None,
        metavar="DIR",
        help="Save sample torso crops during warm-up to this directory",
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
    ema_alpha = _clamp_ema_alpha(
        float(config.PLAYER_BOX_EMA_ALPHA if args.ema_alpha is None else args.ema_alpha)
    )
    use_ema = ema_alpha < 1.0
    lw = int(config.PLAYER_BOX_LINE_WIDTH)
    stale_n = int(config.PLAYER_BOX_EMA_STALE_FRAMES)
    imgsz = int(config.PLAYER_IMGSZ)
    foot_active = bool(config.PLAYER_FOOT_DEDUP_ENABLE and float(config.FOOT_DEDUP_DIST_PX) > 0.0)
    team_enabled = bool(config.TEAM_CLASSIFIER_ENABLE)
    crops_dir = Path(args.debug_team_crops) if args.debug_team_crops else None
    team_clf = (
        TeamClassifier(warmup_frames=args.warmup_frames, debug_crops_dir=crops_dir)
        if team_enabled
        else None
    )
    debug_slices = bool(args.debug_slices)
    debug_team = bool(args.debug_team)

    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        print(f"Could not open video: {src_path}", file=sys.stderr)
        return 1

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if w <= 0 or h <= 0:
        ok, probe = cap.read()
        if not ok or probe is None:
            cap.release()
            print("Could not read video dimensions", file=sys.stderr)
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
            imgsz=imgsz,
        )
        if not results:
            writer.write(frame)
            frame_idx += 1
            continue

        r0 = results[0]
        if r0.boxes is None or r0.boxes.xyxy is None:
            writer.write(frame)
            frame_idx += 1
            continue

        xyxy = r0.boxes.xyxy.detach().cpu().numpy()
        confs = r0.boxes.conf.detach().cpu().numpy() if r0.boxes.conf is not None else np.ones(len(xyxy))
        id_arr: np.ndarray | None = None
        if r0.boxes.id is not None:
            id_arr = r0.boxes.id.detach().cpu().numpy().astype(np.int64)

        if foot_active and len(xyxy) >= 2:
            xyxy, confs, id_arr = box_dedup.filter_by_foot_dedup(
                xyxy, confs, id_arr, float(config.FOOT_DEDUP_DIST_PX)
            )

        if team_clf is not None and id_arr is not None and len(xyxy) > 0:
            team_clf.update(frame, xyxy, id_arr, confs)
            if debug_team and frame_idx % 30 == 0:
                cent = team_clf.centroids
                cent_s = (
                    f"team0=({cent[0][0]:.0f},{cent[0][1]:.0f}) team1=({cent[1][0]:.0f},{cent[1][1]:.0f})"
                    if cent is not None
                    else "n/a"
                )
                print(f"frame={frame_idx} phase={team_clf.phase.name} centroids={cent_s}")
                for i in range(len(xyxy)):
                    tid = int(id_arr[i])
                    dbg = team_clf.get_debug_info(tid)
                    if dbg:
                        print(
                            f"  id={tid} H={dbg.get('H')} S={dbg.get('S')} "
                            f"da={dbg.get('dist_a')} db={dbg.get('dist_b')} "
                            f"lbl={dbg.get('frame_label')} conf={dbg.get('confident')} team={dbg.get('team')}"
                        )

        use_manual = foot_active or (use_ema and id_arr is not None) or team_enabled

        if not use_manual:
            writer.write(
                r0.plot(conf=False, labels=False, boxes=True, line_width=int(lw), img=frame)
            )
        elif use_ema and id_arr is not None:
            seen_this: set[int] = set()
            tracks_draw: list[tuple[int, np.ndarray]] = []
            for i in range(len(xyxy)):
                tid = int(id_arr[i])
                raw = xyxy[i].astype(np.float64)
                seen_this.add(tid)
                prev = ema_prev.get(tid)
                sm = raw.copy() if prev is None else ema_alpha * raw + (1.0 - ema_alpha) * prev
                ema_prev[tid] = sm
                last_seen[tid] = frame_idx
                tracks_draw.append((tid, sm.astype(np.float32)))
            for tid in list(ema_prev.keys()):
                if tid not in seen_this and frame_idx - last_seen.get(tid, -10**9) > stale_n:
                    del ema_prev[tid]
                    last_seen.pop(tid, None)
            vis = _draw_boxes_manual(
                frame, tracks_draw, max(1, lw), team_clf=team_clf, debug_slices=debug_slices
            )
            writer.write(vis)
        else:
            tracks = [
                (int(id_arr[i]) if id_arr is not None else i, xyxy[i].astype(np.float32))
                for i in range(len(xyxy))
            ]
            vis = _draw_boxes_manual(
                frame, tracks, max(1, lw), team_clf=team_clf, debug_slices=debug_slices
            )
            writer.write(vis)

        frame_idx += 1

    cap.release()
    writer.release()
    locked = team_clf.is_locked if team_clf else False
    print(
        f"Wrote {out_path.resolve()} ({frame_idx} frames, ema={'on' if use_ema else 'off'}, "
        f"foot_dedup={foot_active}, team_locked={locked})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

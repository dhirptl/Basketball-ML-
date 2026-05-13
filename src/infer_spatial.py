#!/usr/bin/env python3
"""Orchestrate dual-model inference: court pose + player track + homography + CSV."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

from . import config
from .court_registrar import CourtRegistrar
from .device_utils import resolve_device
from .homography_state import HomographyStateMachine
from .player_tracker import PlayerTracker
from .spatial_projector import bbox_bottom_center, image_to_court_ft, maybe_canonicalize_xy


def _draw_overlay(
    frame: np.ndarray,
    H: np.ndarray | None,
    kpts_xy: np.ndarray,
    kpts_vis: np.ndarray,
    players: list,
    status: str,
) -> np.ndarray:
    vis = frame.copy()
    h, w = vis.shape[:2]
    for i in range(kpts_xy.shape[0]):
        if not kpts_vis[i] or not np.isfinite(kpts_xy[i, 0]):
            continue
        u, v = int(kpts_xy[i, 0]), int(kpts_xy[i, 1])
        cv2.circle(vis, (u, v), 6, (0, 255, 255), 2)
        cv2.putText(vis, str(i), (u + 4, v - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    if H is not None:
        try:
            Hi = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            Hi = None
        if Hi is not None:
            xs = [0.0, config.COURT_LENGTH_FT / 2, config.COURT_LENGTH_FT]
            ys = [0.0, config.COURT_WIDTH_FT / 2, config.COURT_WIDTH_FT]
            for xv in xs:
                pts = []
                for yv in np.linspace(0, config.COURT_WIDTH_FT, 40):
                    p = np.array([[[xv, yv]]], dtype=np.float64)
                    q = cv2.perspectiveTransform(p, Hi)
                    uu, vv = int(q[0, 0, 0]), int(q[0, 0, 1])
                    if 0 <= uu < w and 0 <= vv < h:
                        pts.append((uu, vv))
                for a, b in zip(pts, pts[1:]):
                    cv2.line(vis, a, b, (0, 180, 0), 1)
            for yv in ys:
                pts = []
                for xv in np.linspace(0, config.COURT_LENGTH_FT, 60):
                    p = np.array([[[xv, yv]]], dtype=np.float64)
                    q = cv2.perspectiveTransform(p, Hi)
                    uu, vv = int(q[0, 0, 0]), int(q[0, 0, 1])
                    if 0 <= uu < w and 0 <= vv < h:
                        pts.append((uu, vv))
                for a, b in zip(pts, pts[1:]):
                    cv2.line(vis, a, b, (0, 180, 0), 1)
    for pl in players:
        x1, y1, x2, y2 = pl.xyxy
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (255, 128, 0), 2)
        fu, fv = bbox_bottom_center(pl.xyxy)
        cv2.circle(vis, (int(fu), int(fv)), 4, (0, 0, 255), -1)
        cv2.putText(
            vis,
            f"id{pl.track_id}",
            (int(x1), int(y1) - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )
    cv2.putText(vis, status[:120], (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
    return vis


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Dual-model spatial tracker inference")
    p.add_argument("--video", type=str, required=True, help="Path to input .mp4")
    p.add_argument("--player-weights", type=str, required=True, help="YOLO bbox weights (best.pt)")
    p.add_argument("--court-weights", type=str, required=True, help="YOLO pose weights for court (best.pt)")
    p.add_argument("--out-csv", type=str, default="", help="Optional CSV output path")
    p.add_argument("--device", type=str, default="auto", help="auto | mps | cuda | cpu")
    p.add_argument("--debug", action="store_true", help="Enable debug overlay (overrides config)")
    p.add_argument("--debug-out", type=str, default="", help="Write debug video to this path")
    args = p.parse_args(argv)

    device = resolve_device(args.device)
    debug = bool(config.DEBUG_OVERLAY or args.debug)
    video_path = Path(args.video)
    if not video_path.is_file():
        print(f"Video not found: {video_path}", file=sys.stderr)
        return 1

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Could not open video: {video_path}", file=sys.stderr)
        return 1

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if w <= 0 or h <= 0:
        ok, probe = cap.read()
        if not ok or probe is None:
            print("Could not read video dimensions", file=sys.stderr)
            return 1
        h, w = probe.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    court = CourtRegistrar(args.court_weights, device)
    players = PlayerTracker(args.player_weights, device)
    homo = HomographyStateMachine((w, h))

    writer: cv2.VideoWriter | None = None
    if args.debug_out:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.debug_out, fourcc, cap.get(cv2.CAP_PROP_FPS) or 25.0, (w, h))

    csv_file = None
    csv_writer = None
    if args.out_csv:
        csv_file = open(args.out_csv, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            [
                "frame_idx",
                "track_temp_id",
                "x_ft",
                "y_ft",
                "player_conf",
                "homography_conf",
                "num_visible_kpts",
            ]
        )

    frame_idx = 0
    persist = True
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        kpts = court.infer(frame)
        hf = homo.step(frame, kpts)
        reset_ids = homo.should_reset_tracker(hf)
        if reset_ids:
            persist = False
        trks = players.track(frame, persist=persist, reset_ids=reset_ids)
        persist = True

        status = (
            f"f={frame_idx} vis={hf.num_visible_kpts} reg={hf.registration_ok} "
            f"cut={hf.scene_cut} hconf={hf.homography_conf:.2f} H={'ok' if hf.H_image_to_court is not None else 'none'}"
        )

        if hf.H_image_to_court is not None:
            for pl in trks:
                fu, fv = bbox_bottom_center(pl.xyxy)
                xy_c = image_to_court_ft(hf.H_image_to_court, fu, fv)
                if xy_c is None:
                    continue
                x_ft, y_ft = maybe_canonicalize_xy(xy_c[0], xy_c[1])
                tid = pl.track_id if pl.track_id >= 0 else -1
                if csv_writer:
                    csv_writer.writerow(
                        [
                            frame_idx,
                            tid,
                            f"{x_ft:.4f}",
                            f"{y_ft:.4f}",
                            f"{pl.conf:.4f}",
                            f"{hf.homography_conf:.4f}",
                            hf.num_visible_kpts,
                        ]
                    )
                print(
                    f"frame={frame_idx} id={tid} x_ft={x_ft:.2f} y_ft={y_ft:.2f} "
                    f"pconf={pl.conf:.2f} hconf={hf.homography_conf:.2f} kpts={hf.num_visible_kpts}"
                )

        if debug or writer is not None:
            ov = _draw_overlay(
                frame,
                hf.H_image_to_court,
                kpts.xy,
                kpts.visible_mask,
                trks,
                status,
            )
            if writer is not None:
                writer.write(ov)
            elif debug:
                cv2.imshow("spatial_tracker", ov)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    if csv_file is not None:
        csv_file.close()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

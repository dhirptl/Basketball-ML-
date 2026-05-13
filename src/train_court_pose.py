#!/usr/bin/env python3
"""Train YOLOv11 pose model for court keypoints."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from .device_utils import resolve_device


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train court registration (YOLO pose)")
    p.add_argument("--data", type=str, required=True, help="Pose data.yaml (kpt_shape [8,3])")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--project", type=str, default="spatialTracker")
    p.add_argument("--name", type=str, default="courtWeights")
    p.add_argument("--model", type=str, default="yolo11n-pose.pt")
    args = p.parse_args(argv)

    device = resolve_device(args.device)
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=args.project,
        name=args.name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

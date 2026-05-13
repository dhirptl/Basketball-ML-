#!/usr/bin/env python3
"""Train YOLOv11 bbox model on the Basketball Players dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from .device_utils import resolve_device


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train player detection (YOLO bbox)")
    p.add_argument(
        "--data",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "Basketball Players" / "dataset.yaml"),
        help="Ultralytics data.yaml path",
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--project", type=str, default="spatialTracker")
    p.add_argument("--name", type=str, default="playerWeights")
    p.add_argument("--model", type=str, default="yolo11n.pt")
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

#!/usr/bin/env python3
"""YOLO predict with Player-only defaults (classes, conf, iou, max_det from config)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

from . import config
from .device_utils import resolve_device


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="YOLO predict: Player class only with tuned conf/iou")
    p.add_argument("--model", type=str, required=True, help="Weights .pt path")
    p.add_argument("--source", type=str, required=True, help="Video, image, or directory")
    p.add_argument("--device", type=str, default="auto", help="auto | mps | cuda | cpu")
    p.add_argument("--conf", type=float, default=None, help=f"Override conf (default {config.PLAYER_PREDICT_CONF})")
    p.add_argument("--iou", type=float, default=None, help=f"Override NMS iou (default {config.PLAYER_PREDICT_IOU})")
    p.add_argument("--max-det", type=int, default=None, help=f"Override max_det (default {config.PLAYER_PREDICT_MAX_DET})")
    p.add_argument("--save", action="store_true", help="Save annotated outputs (Ultralytics runs/detect)")
    args = p.parse_args(argv)

    model_path = Path(args.model)
    if not model_path.is_file():
        print(f"Model not found: {model_path}", file=sys.stderr)
        return 1

    device = resolve_device(args.device)
    conf = float(config.PLAYER_PREDICT_CONF if args.conf is None else args.conf)
    iou = float(config.PLAYER_PREDICT_IOU if args.iou is None else args.iou)
    max_det = int(config.PLAYER_PREDICT_MAX_DET if args.max_det is None else args.max_det)

    model = YOLO(str(model_path))
    model.predict(
        source=args.source,
        classes=[config.PLAYER_CLASS_ID],
        conf=conf,
        iou=iou,
        max_det=max_det,
        imgsz=int(config.PLAYER_IMGSZ),
        device=device,
        save=args.save,
        verbose=True,
        show_labels=False,
        show_conf=False,
        line_width=int(config.PLAYER_BOX_LINE_WIDTH),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

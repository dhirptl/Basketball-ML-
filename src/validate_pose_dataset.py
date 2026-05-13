#!/usr/bin/env python3
"""Validate YOLO pose labels for court keypoints (8 points, strict schema)."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from . import config


def _parse_pose_line(parts: list[str]) -> tuple[int, list[tuple[float, float, int]]]:
    # YOLO pose: class cx cy w h then K*(x y v) normalized
    bbox_n = 4
    kpt_n = 3 * config.NUM_COURT_KEYPOINTS
    need_vals = bbox_n + kpt_n
    if len(parts) < 1 + need_vals:
        raise ValueError(f"expected at least {1 + need_vals} tokens, got {len(parts)}")
    cls = int(parts[0])
    vals = [float(x) for x in parts[1 : 1 + need_vals]]
    if len(vals) < need_vals:
        raise ValueError(f"expected {need_vals} numbers after class, got {len(vals)}")
    kpts: list[tuple[float, float, int]] = []
    off = bbox_n
    for i in range(config.NUM_COURT_KEYPOINTS):
        x, y, v = vals[off + 3 * i], vals[off + 3 * i + 1], int(round(vals[off + 3 * i + 2]))
        kpts.append((x, y, v))
    return cls, kpts


def validate_label_file(path: Path) -> list[str]:
    errors: list[str] = []
    text = path.read_text().strip().splitlines()
    for li, line in enumerate(text, 1):
        parts = line.strip().split()
        if not parts:
            continue
        try:
            cls, kpts = _parse_pose_line(parts)
        except Exception as e:
            errors.append(f"{path}:{li}: parse error: {e}")
            continue
        if cls != 0:
            errors.append(f"{path}:{li}: expected single class id 0 (Court), got {cls}")
        vis = [v > 0 for _, _, v in kpts]
        for i, (x, y, v) in enumerate(kpts):
            if v == 0:
                continue
            if not (-1e-3 <= x <= 1.0 + 1e-3 and -1e-3 <= y <= 1.0 + 1e-3):
                errors.append(f"{path}:{li}: kpt {i} out of [0,1]: ({x},{y}) v={v}")
            if v not in (0, 1, 2):
                errors.append(f"{path}:{li}: kpt {i} invalid visibility {v}")
        # duplicate visible keypoints (very close)
        vis_pts = [(i, x, y) for i, (x, y, v) in enumerate(kpts) if v > 0]
        for a in range(len(vis_pts)):
            for b in range(a + 1, len(vis_pts)):
                ia, xa, ya = vis_pts[a]
                ib, xb, yb = vis_pts[b]
                d = math.hypot(xa - xb, ya - yb)
                if d < 1e-4:
                    errors.append(f"{path}:{li}: duplicate/near-duplicate kpts {ia} and {ib}")
    return errors


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Validate court pose YOLO labels")
    p.add_argument("--labels-dir", type=str, required=True)
    args = p.parse_args(argv)

    root = Path(args.labels_dir)
    if not root.is_dir():
        print(f"Not a directory: {root}")
        return 1
    all_errs: list[str] = []
    for lf in sorted(root.glob("*.txt")):
        all_errs.extend(validate_label_file(lf))
    if all_errs:
        print("\n".join(all_errs[:200]))
        if len(all_errs) > 200:
            print(f"... and {len(all_errs) - 200} more")
        return 1
    print("OK: no issues found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

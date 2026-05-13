#!/usr/bin/env python3
"""Randomly sample training images for court keypoint annotation."""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Sample frames/images for pose annotation")
    p.add_argument(
        "--images-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "Basketball Players" / "train" / "images"),
        help="Directory containing source images",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory (created) for copied images",
    )
    p.add_argument("--count", type=int, default=500, help="Number of images to sample")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--manifest", type=str, default="", help="Optional CSV manifest path")
    args = p.parse_args(argv)

    src = Path(args.images_dir)
    if not src.is_dir():
        raise SystemExit(f"images dir not found: {src}")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    all_imgs = [p for p in src.iterdir() if p.suffix.lower() in exts]
    if not all_imgs:
        raise SystemExit(f"No images found in {src}")

    rnd = random.Random(args.seed)
    n = min(args.count, len(all_imgs))
    chosen = rnd.sample(all_imgs, n)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[tuple[str, str]] = []
    for i, pth in enumerate(chosen):
        dest = out / pth.name
        shutil.copy2(pth, dest)
        manifest_rows.append((str(pth), str(dest)))

    if args.manifest:
        mp = Path(args.manifest)
        mp.parent.mkdir(parents=True, exist_ok=True)
        with mp.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["source_path", "dest_path"])
            w.writerows(manifest_rows)

    print(f"Copied {n} images to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

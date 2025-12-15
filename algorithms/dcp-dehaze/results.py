#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterable

import cv2
from numpy.typing import NDArray
from tqdm import tqdm

from dcp_dehaze import dcp_dehaze


def resize(img: NDArray, max_size: int) -> NDArray:
    M = max(img.shape[:2])
    ratio = float(max_size) / float(M)
    if M > max_size:
        img = cv2.resize(img, (0, 0), fx=ratio, fy=ratio)  # type: ignore
    return img


def iter_images(input_dir: Path, exts: set[str]) -> Iterable[Path]:
    for p in sorted(input_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run DCP dehaze on all images in a directory, saving outputs with the same filenames."
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input images.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to write output images.")
    parser.add_argument(
        "--max_size",
        type=int,
        default=0,
        help="If > 0, resize so max(H,W) <= max_size before dehazing.",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default="jpg,jpeg,png,bmp,tif,tiff,webp",
        help="Comma-separated list of extensions to process.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"input_dir does not exist or is not a directory: {input_dir}")

    exts = {"." + e.strip().lower().lstrip(".") for e in args.ext.split(",") if e.strip()}
    paths = list(iter_images(input_dir, exts))
    if not paths:
        print(f"No images found in {input_dir} with extensions: {sorted(exts)}")
        return 0

    total_start = time.time()
    failures = 0

    for in_path in tqdm(paths, desc="Dehazing", unit="img"):
        # Preserve only the filename (not subfolders) as requested
        out_path = output_dir / in_path.name

        img = cv2.imread(str(in_path))
        if img is None:
            failures += 1
            tqdm.write(f"[WARN] Could not read: {in_path}")
            continue

        if args.max_size and args.max_size > 0:
            img = resize(img, args.max_size)

        try:
            dehazed = dcp_dehaze(img)
        except Exception as e:
            failures += 1
            tqdm.write(f"[WARN] Failed on {in_path}: {e}")
            continue

        ok = cv2.imwrite(str(out_path), dehazed)
        if not ok:
            failures += 1
            tqdm.write(f"[WARN] Could not write: {out_path}")

    total_end = time.time()
    print(f"\nDone. Processed {len(paths)} images in {total_end - total_start:.2f}s. Failures: {failures}")
    print(f"Outputs saved to: {output_dir}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np
import pandas as pd
import skvideo.measure
import scipy.misc
from skimage.transform import resize
from tqdm import tqdm

if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]

def _imresize(img, scale, interp='bicubic', mode='F'):
    order = 3 if interp == 'bicubic' else 1
    out = resize(
        img,
        (int(img.shape[0] * scale), int(img.shape[1] * scale)),
        order=order,
        preserve_range=True,
        anti_aliasing=True
    )
    return out.astype(img.dtype)

if not hasattr(scipy.misc, "imresize"):
    scipy.misc.imresize = _imresize

def resize_max_side(img: np.ndarray, max_side: int) -> np.ndarray:
    if max_side <= 0:
        return img
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def to_gray_u8(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr.ndim == 2:
        g = img_bgr
        return g if g.dtype == np.uint8 else np.clip(g, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


# Optional super-fast Levenshtein if installed
try:
    import Levenshtein  # type: ignore

    def char_accuracy(pred: str, gt: str) -> float:
        if not pred or not gt:
            return 0.0
        dist = Levenshtein.distance(pred, gt)
        max_len = max(len(pred), len(gt))
        return 1.0 - (dist / max_len) if max_len else 1.0

except Exception:
    def char_accuracy(pred: str, gt: str) -> float:
        if not pred or not gt:
            return 0.0
        a, b = pred, gt
        m, n = len(a), len(b)
        if m == 0 and n == 0:
            return 1.0
        if m == 0 or n == 0:
            return 0.0

        # Ensure n is smaller for slightly better cache behavior
        if n > m:
            a, b = b, a
            m, n = n, m

        prev = list(range(n + 1))
        curr = [0] * (n + 1)
        for i in range(1, m + 1):
            curr[0] = i
            ai = a[i - 1]
            for j in range(1, n + 1):
                cost = 0 if ai == b[j - 1] else 1
                curr[j] = min(
                    prev[j] + 1,        # deletion
                    curr[j - 1] + 1,    # insertion
                    prev[j - 1] + cost  # substitution
                )
            prev, curr = curr, prev

        dist = prev[n]
        max_len = max(m, n)
        return 1.0 - (dist / max_len) if max_len else 1.0


def niqe_score(img_bgr_or_gray: np.ndarray) -> float:
    gray = to_gray_u8(img_bgr_or_gray)
    gray_f = gray.astype(np.float32)
    return float(skvideo.measure.niqe(gray_f))


class BrisqueComputer:
    def __init__(self, model_path: str, range_path: str):
        self.model_path = model_path
        self.range_path = range_path
        self.obj = None
        if hasattr(cv2, "quality") and hasattr(cv2.quality, "QualityBRISQUE_create"):
            try:
                self.obj = cv2.quality.QualityBRISQUE_create(model_path, range_path)
            except Exception:
                self.obj = None

    def __call__(self, img_bgr: np.ndarray) -> float:
        if self.obj is not None:
            val = self.obj.compute(img_bgr)
            # OpenCV returns [[score]] sometimes; normalize to float
            if isinstance(val, (list, tuple, np.ndarray)):
                return float(np.array(val).ravel()[0])
            return float(val)
        # Fallback: loads/parses models each call (slower)
        val = cv2.quality.QualityBRISQUE_compute(img_bgr, self.model_path, self.range_path)
        return float(np.array(val).ravel()[0])


def compute_basic_metrics(img_bgr: np.ndarray) -> Tuple[float, float]:
    # brightness: mean of all channels
    brightness = float(img_bgr.mean())

    gray = to_gray_u8(img_bgr)
    # Use CV_32F to avoid CV_64F overhead (usually enough)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    sharpness = float(lap.var())
    return brightness, sharpness

def main():
    parser = argparse.ArgumentParser(description="Evaluate processing on license plate images")
    parser.add_argument("--raw_dir", type=str, required=True, help="Directory containing raw foggy images")
    parser.add_argument("--processed_dir", type=str, required=True, help="Directory containing processed images")
    parser.add_argument("--labels_csv", type=str, required=True, help="CSV file with image_path and label columns")
    parser.add_argument("--results_csv", type=str, required=True, help="Output CSV file for results")

    # Speed knobs
    parser.add_argument("--max_metric_side", type=int, default=512,
                        help="Downscale images so max(H,W)<=this before computing BRISQUE/NIQE/sharpness/brightness. 0 disables.")
    parser.add_argument("--max_alpr_side", type=int, default=0,
                        help="Optional downscale before ALPR for speed. 0 disables (uses full-res).")

    parser.add_argument("--brisque_model", type=str, default="brisque_model_live.yml")
    parser.add_argument("--brisque_range", type=str, default="brisque_range_live.yml")

    args = parser.parse_args()

    # Import here so the script still loads even if your wrapper deps are heavy.
    from scripts.fast_alpr_wrapper import run_fast_alpr

    raw_dir = Path(args.raw_dir)
    proc_dir = Path(args.processed_dir)

    df_labels = pd.read_csv(args.labels_csv)
    if "image_path" not in df_labels.columns or "label" not in df_labels.columns:
        raise ValueError("labels_csv must have 'image_path' and 'label' columns")

    brisque = BrisqueComputer(args.brisque_model, args.brisque_range)

    results = []
    # itertuples is much faster than iterrows
    for row in tqdm(df_labels.itertuples(index=False), total=len(df_labels), desc="Evaluating images"):
        filename = getattr(row, "image_path")
        gt_plate = str(getattr(row, "label")).strip()

        img_name = Path(filename).name
        raw_path = raw_dir / img_name
        proc_path = proc_dir / img_name

        print(f"Processing: {img_name}", end="\r")

        if not raw_path.exists():
            print(f"\nWarning: Image not found: {raw_path}, skipping...")
            continue
        if not proc_path.exists():
            print(f"\nWarning: Processed image not found: {proc_path}, skipping...")
            continue

        img_raw = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
        img_processed = cv2.imread(str(proc_path), cv2.IMREAD_COLOR)
        if img_raw is None:
            print(f"\nWarning: Failed to load raw image: {raw_path}, skipping...")
            continue
        if img_processed is None:
            print(f"\nWarning: Failed to load processed image: {proc_path}, skipping...")
            continue

        img_raw_m = resize_max_side(img_raw, args.max_metric_side)
        img_proc_m = resize_max_side(img_processed, args.max_metric_side)

        brightness_raw, sharp_raw = compute_basic_metrics(img_raw_m)
        brightness_processed, sharp_processed = compute_basic_metrics(img_proc_m)

        brisque_raw = brisque(img_raw_m)
        brisque_processed = brisque(img_proc_m)

        niqe_raw = niqe_score(img_raw_m)
        niqe_processed = niqe_score(img_proc_m)

        img_raw_a = resize_max_side(img_raw, args.max_alpr_side)
        img_proc_a = resize_max_side(img_processed, args.max_alpr_side)

        pred_raw, conf_raw, x1_raw, y1_raw, x2_raw, y2_raw = run_fast_alpr(img_raw_a)
        pred_processed, conf_processed, x1_processed, y1_processed, x2_processed, y2_processed = run_fast_alpr(img_proc_a)

        if "," in gt_plate:
            parts = [p.strip() for p in gt_plate.split(",") if p.strip()]
            acc_raw = max((char_accuracy(pred_raw, p) for p in parts), default=0.0)
            acc_processed = max((char_accuracy(pred_processed, p) for p in parts), default=0.0)
        else:
            acc_raw = char_accuracy(pred_raw, gt_plate)
            acc_processed = char_accuracy(pred_processed, gt_plate)

        results.append({
            "filename": filename,
            "gt_plate": gt_plate,
            "pred_raw": pred_raw,
            "pred_processed": pred_processed,
            "acc_raw": acc_raw,
            "acc_processed": acc_processed,
            "conf_raw": conf_raw,
            "conf_processed": conf_processed,
            "brightness_raw": brightness_raw,
            "brightness_processed": brightness_processed,
            "sharp_raw": sharp_raw,
            "sharp_processed": sharp_processed,
            "brisque_raw": brisque_raw,
            "brisque_processed": brisque_processed,
            "niqe_raw": niqe_raw,
            "niqe_processed": niqe_processed,
        })

        # Show a visual of the original image with bbox and text and confidence, also for processed
        cv2.rectangle(img_raw_a, (x1_raw, y1_raw), (x2_raw, y2_raw), (0, 255, 0), 2)
        cv2.putText(img_raw_a, f"{pred_raw} ({conf_raw:.4f})", (x1_raw, max(y1_raw - 10, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 255, 0), 3)

        cv2.rectangle(img_proc_a, (x1_processed, y1_processed), (x2_processed, y2_processed), (0, 255, 0), 2)
        cv2.putText(img_proc_a, f"{pred_processed} ({conf_processed:.4f})", (x1_processed, max(y1_processed - 10, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 255, 0), 3)
        combined = np.hstack((img_raw_a, img_proc_a))
        cv2.imshow("Raw (left) vs Processed (right). Ground Truth: " + gt_plate, combined)
        key = cv2.waitKey(0)
        if key == 27:  # ESC to quit early
            break

    print()  # newline after progress

    df_results = pd.DataFrame(results)
    df_results.to_csv(args.results_csv, index=False)

    if len(results) == 0:
        print("No images were processed successfully.")
        return

    # Summary (vectorized via pandas)
    print("\n=== Summary Statistics ===")
    for k in [
        "acc_raw", "acc_processed",
        "conf_raw", "conf_processed",
        "brightness_raw", "brightness_processed",
        "sharp_raw", "sharp_processed",
        "brisque_raw", "brisque_processed",
        "niqe_raw", "niqe_processed",
    ]:
        print(f"Mean {k}: {df_results[k].mean():.4f}")


if __name__ == "__main__":
    main()

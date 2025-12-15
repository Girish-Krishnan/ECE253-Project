
import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from distortion_classifier import classify_distortion, compute_distortion_metrics
from scripts.fast_alpr_wrapper import run_fast_alpr
import skvideo.measure
from skimage.transform import resize
import scipy

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

# Monkey-patch scipy.misc.imresize
if not hasattr(scipy.misc, "imresize"):
    scipy.misc.imresize = _imresize

def niqe_score(img) -> float:
    # scikit-video NIQE expects grayscale (C must be 1), so convert if needed.
    if img.ndim == 3:
        # RGB -> luminance (BT.601-ish)
        img = (0.299*img[...,0] + 0.587*img[...,1] + 0.114*img[...,2]).astype(img.dtype)

    # skvideo returns a scalar for a single image
    return float(skvideo.measure.niqe(img))

def apply_fog(image_bgr, fog_intensity=0.5):
    """Apply fog effect to image."""
    img = image_bgr.astype(np.float32) / 255.0
    h, w = img.shape[:2]
    
    A = 0.9
    center_x, center_y = w // 2, h // 2
    y_coords, x_coords = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    max_dist = np.sqrt(center_x**2 + center_y**2)
    depth = 1.0 - (dist_from_center / (max_dist + 1)) * fog_intensity
    
    beta = fog_intensity * 2.0
    t = np.exp(-beta * depth)
    t = np.clip(t, 0.3, 1.0)
    
    img_foggy = img * t[..., None] + (1 - t[..., None]) * A
    
    noise = np.random.normal(0, 0.02, img_foggy.shape).astype(np.float32)
    img_foggy = np.clip(img_foggy + noise, 0, 1)
    
    return (img_foggy * 255).astype(np.uint8)


def defog_dcp_clahe(image_bgr):
    I = image_bgr.astype(np.float32) / 255.0

    patch = 100
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))
    dark = cv2.erode(np.min(I, axis=2), kernel)

    num_pixels = I.shape[0] * I.shape[1]
    top_k = max(1, num_pixels // 2000)
    indices = np.argpartition(dark.flatten(), -top_k)[-top_k:]
    A = np.mean(I.reshape(-1,3)[indices], axis=0)

    omega = 0.7
    norm_I = I / (A + 1e-6)
    dark_norm = cv2.erode(np.min(norm_I, axis=2), kernel)
    t = 1 - omega * dark_norm
    t0 = 0.2
    t = np.clip(t, t0, 1)

    t = cv2.GaussianBlur(t, (7,7), 10)

    J = (I - A) / t[..., None] + A
    J = np.clip(J, 0, 1)

    blend_alpha = 0.8
    J = blend_alpha * J + (1 - blend_alpha) * I

    hsv = cv2.cvtColor((J*255).astype(np.uint8), cv2.COLOR_BGR2HSV)
    clahe = cv2.createCLAHE(clipLimit=0.8, tileGridSize=(8,8))
    hsv[:,:,2] = clahe.apply(hsv[:,:,2])
    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return out


def char_accuracy(pred: str, gt: str) -> float:
    """Levenshtein similarity between pred and gt."""
    if not pred or not gt:
        return 0.0

    m, n = len(pred), len(gt)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred[i - 1] == gt[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + 1)

    distance = dp[m][n]
    max_len = max(m, n)
    return 1.0 - (distance / max_len) if max_len > 0 else 1.0


def main():
    parser = argparse.ArgumentParser(description="Evaluate defogging on license plate images")
    parser.add_argument("--raw_dir", type=str, required=True, help="Directory containing raw foggy images")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save defogged images")
    parser.add_argument("--labels_csv", type=str, required=True, help="CSV file with filename and plate columns")
    parser.add_argument("--results_csv", type=str, required=True, help="Output CSV file for results")

    args = parser.parse_args()

    # Read labels CSV
    df_labels = pd.read_csv(args.labels_csv)
    if "image_path" not in df_labels.columns or "label" not in df_labels.columns:
        raise ValueError("labels_csv must have 'image_path' and 'label' columns")

    # Make sure out_dir exists
    os.makedirs(args.out_dir, exist_ok=True)

    # Process each image
    results = []
    for _, row in df_labels.iterrows():
        print("Processing:", row["image_path"], end='\r')
        filename = row["image_path"]
        gt_plate = str(row["label"]).strip()

        # Load raw image
        img_path = os.path.join(args.raw_dir, filename)
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}, skipping...")
            continue

        img_original = cv2.imread(img_path)
        if img_original is None:
            print(f"Warning: Failed to load image: {img_path}, skipping...")
            continue

        # Classify original image distortion type
        # metrics = compute_distortion_metrics(img_original)
        # predicted = classify_distortion(img_original)
        # print(f"{filename}: class={predicted}, metrics={metrics}")

        # Apply fog to make image foggy
        # img_raw = apply_fog(img_original, fog_intensity=0.6)
        img_raw = img_original

        # Run defogging and save
        img_defog = defog_dcp_clahe(img_raw)
        filename = Path(filename).name  # Just the filename
        out_path = os.path.join(args.out_dir, filename)
        cv2.imwrite(out_path, img_defog)

        # Compute brightness (mean)
        brightness_raw = float(img_raw.mean())
        brightness_defog = float(img_defog.mean())

        # Compute sharpness (Laplacian variance)
        gray_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
        gray_defog = cv2.cvtColor(img_defog, cv2.COLOR_BGR2GRAY)
        sharp_raw = float(cv2.Laplacian(gray_raw, cv2.CV_64F).var())
        sharp_defog = float(cv2.Laplacian(gray_defog, cv2.CV_64F).var())
        brisque_raw = cv2.quality.QualityBRISQUE_compute(img_raw, "brisque_model_live.yml", "brisque_range_live.yml")
        brisque_raw = brisque_raw[0]
        brisque_defog = cv2.quality.QualityBRISQUE_compute(img_defog, "brisque_model_live.yml", "brisque_range_live.yml")
        brisque_defog = brisque_defog[0]
        niqe_raw = niqe_score(img_raw)
        niqe_defog = niqe_score(img_defog)

        # Run ALPR
        pred_raw, conf_raw = run_fast_alpr(img_raw)
        pred_defog, conf_defog = run_fast_alpr(img_defog)

        # Compute accuracy
        if ',' in gt_plate:
            gt_plate_parts = [part.strip() for part in gt_plate.split(',')]
            acc_raw = max(char_accuracy(pred_raw, part) for part in gt_plate_parts)
            acc_defog = max(char_accuracy(pred_defog, part) for part in gt_plate_parts)
        else:
            acc_raw = char_accuracy(pred_raw, gt_plate)
            acc_defog = char_accuracy(pred_defog, gt_plate)

        # Append results
        results.append({
            "filename": filename,
            "gt_plate": gt_plate,
            "pred_raw": pred_raw,
            "pred_defog": pred_defog,
            "acc_raw": acc_raw,
            "acc_defog": acc_defog,
            "conf_raw": conf_raw,
            "conf_defog": conf_defog,
            "brightness_raw": brightness_raw,
            "brightness_defog": brightness_defog,
            "sharp_raw": sharp_raw,
            "sharp_defog": sharp_defog,
            "brisque_raw": brisque_raw,
            "brisque_defog": brisque_defog,
            "niqe_raw": niqe_raw,
            "niqe_defog": niqe_defog,
        })

        # print(f"Processed: {filename}")

    # Convert to DataFrame and save
    df_results = pd.DataFrame(results)
    df_results.to_csv(args.results_csv, index=False)

    # Print summary statistics
    if len(results) > 0:
        mean_acc_raw = df_results["acc_raw"].mean()
        mean_acc_defog = df_results["acc_defog"].mean()
        mean_conf_raw = df_results["conf_raw"].mean()
        mean_conf_defog = df_results["conf_defog"].mean()
        mean_brightness_raw = df_results["brightness_raw"].mean()
        mean_brightness_defog = df_results["brightness_defog"].mean()
        mean_sharp_raw = df_results["sharp_raw"].mean()
        mean_sharp_defog = df_results["sharp_defog"].mean()
        mean_brisque_raw = df_results["brisque_raw"].mean()
        mean_brisque_defog = df_results["brisque_defog"].mean()
        mean_niqe_raw = df_results["niqe_raw"].mean()
        mean_niqe_defog = df_results["niqe_defog"].mean()

        print("\n=== Summary Statistics ===")
        print(f"Mean accuracy (raw): {mean_acc_raw:.4f}")
        print(f"Mean accuracy (defogged): {mean_acc_defog:.4f}")
        print(f"Mean confidence (raw): {mean_conf_raw:.4f}")
        print(f"Mean confidence (defogged): {mean_conf_defog:.4f}")
        print(f"Mean brightness (raw): {mean_brightness_raw:.2f}")
        print(f"Mean brightness (defogged): {mean_brightness_defog:.2f}")
        print(f"Mean sharpness (raw): {mean_sharp_raw:.2f}")
        print(f"Mean sharpness (defogged): {mean_sharp_defog:.2f}")
        print(f"Mean BRISQUE (raw): {mean_brisque_raw:.2f}")
        print(f"Mean BRISQUE (defogged): {mean_brisque_defog:.2f}")
        print(f"Mean NIQE (raw): {mean_niqe_raw:.2f}")
        print(f"Mean NIQE (defogged): {mean_niqe_defog:.2f}")
    else:
        print("No images were processed successfully.")


if __name__ == "__main__":
    main()


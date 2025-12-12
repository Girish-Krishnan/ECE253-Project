import os
import json
import cv2
import numpy as np
from typing import List, Tuple, Dict

INPUT_DIR = r"dip_algorithms\data\blurred"
OUTPUT_DIR = r"dip_algorithms\data\deblurred"

# -------------------------------------------------------------
# Motion PSF
# -------------------------------------------------------------
def motion_psf(length: int, angle_deg: float) -> np.ndarray:
    length = int(length)
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((length / 2, length / 2), angle_deg, 1.0)
    k = cv2.warpAffine(k, M, (length, length))
    k = np.clip(k, 0, None)
    k /= (k.sum() + 1e-6)
    return k

# -------------------------------------------------------------
# Grayscale RL
# -------------------------------------------------------------
def rl_gray_fast(
    y: np.ndarray,
    psf: np.ndarray,
    iters: int = 3,
    clip_low: float = 0.8,
    clip_high: float = 3.0,
):
    eps = 1e-6
    psf = psf / (psf.sum() + eps)
    psf_flip = psf[::-1, ::-1]

    x = y.copy()

    for _ in range(iters):
        est = cv2.filter2D(x, -1, psf, borderType=cv2.BORDER_REPLICATE)
        est = np.clip(est, eps, 1.0)
        ratio = np.clip(y / est, clip_low, clip_high)
        corr = cv2.filter2D(ratio, -1, psf_flip, borderType=cv2.BORDER_REPLICATE)
        x *= corr
        x = np.clip(x, 0.0, 1.0)

    return x

# -------------------------------------------------------------
# SHARPNESS SCORE
# -------------------------------------------------------------
def laplacian_sharpness_gray(img_gray: np.ndarray) -> float:
    lap = cv2.Laplacian(img_gray, cv2.CV_32F, ksize=3)
    return float(lap.var())

def estimate_psf_grid_fast(
    blurred_bgr: np.ndarray,
    lengths: List[int],
    angles_deg: List[float],
    iters_eval: int = 6,
    scale: float = 0.6,
):
    small = cv2.resize(
        blurred_bgr, (0, 0),
        fx=scale, fy=scale,
        interpolation=cv2.INTER_AREA
    )

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    best_score = -np.inf
    best_psf = None
    best_params = (None, None)

    for L in lengths:
        for theta in angles_deg:
            psf = motion_psf(L, theta)
            x_est = rl_gray_fast(gray, psf, iters=iters_eval)
            S_i = laplacian_sharpness_gray(x_est)

            if S_i >= best_score:
                best_score = S_i
                best_psf = psf
                best_params = (L, theta)

    return best_psf, best_params, best_score

def rl_color_fast(
    img_bgr: np.ndarray,
    psf: np.ndarray,
    iters: int = 25,
    pad_px: int = 40,
):
    h0, w0 = img_bgr.shape[:2]

    img_pad = cv2.copyMakeBorder(
        img_bgr, pad_px, pad_px, pad_px, pad_px,
        borderType=cv2.BORDER_REPLICATE,
    )

    img_pad = img_pad.astype(np.float32) / 255.0
    psf = psf / (psf.sum() + 1e-8)
    psf_flip = psf[::-1, ::-1]
    eps = 1e-6

    out = np.zeros_like(img_pad, np.float32)

    for c in range(3):
        y = img_pad[..., c]
        x = y.copy()

        for _ in range(iters):
            est = cv2.filter2D(x, -1, psf, borderType=cv2.BORDER_REPLICATE)
            est = np.clip(est, eps, 1.0)
            ratio = np.clip(y / est, 0.6, 2.5)
            corr = cv2.filter2D(ratio, -1, psf_flip, borderType=cv2.BORDER_REPLICATE)
            x *= corr
            x = np.clip(x, 0.0, 1.0)

        out[..., c] = x

    out = np.clip(out, 0.0, 1.0)
    out = (out * 255).astype(np.uint8)

    return out[pad_px : pad_px + h0, pad_px : pad_px + w0]

def grid_rl_deblur_fast(blurred_bgr: np.ndarray):

    lengths = [7, 8, 10]
    angles_deg = [15, -30]

    best_psf, (L_star, theta_star), best_score = estimate_psf_grid_fast(
        blurred_bgr,
        lengths=lengths,
        angles_deg=angles_deg,
        iters_eval=3,
        scale=1,
    )

    pad_px = int(1.2 * L_star)
    restored = rl_color_fast(
        blurred_bgr,
        best_psf,
        iters=25,
        pad_px=pad_px,
    )

    info = dict(
        L_star=int(L_star),
        theta_star=float(theta_star),
        sharpness=float(best_score),
    )

    return restored, info

def process_folder():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    valid_exts = (".jpg", ".jpeg", ".png")
    files = sorted([
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(valid_exts)
    ])

    print(f"[INFO] Found {len(files)} images in '{INPUT_DIR}/'")

    for fname in files:
        in_path = os.path.join(INPUT_DIR, fname)
        out_name = f"deblurred_{os.path.splitext(fname)[0]}.png"
        out_path = os.path.join(OUTPUT_DIR, out_name)

        img = cv2.imread(in_path)
        restored, info = grid_rl_deblur_fast(img)

        cv2.imwrite(out_path, restored)

        with open(out_path + ".json", "w") as f:
            json.dump(info, f, indent=2)

        print(f"[OK] {fname} -> L*={info['L_star']} theta*={info['theta_star']}")


if __name__ == "__main__":
    process_folder()
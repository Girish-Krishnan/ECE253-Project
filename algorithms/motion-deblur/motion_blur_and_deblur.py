import os
import cv2
import numpy as np

# ===========================
# CONFIG
# ===========================
RAW_DIR = r"algorithms\motion-deblur\raw"
BLURRED_DIR = r"algorithms\motion-deblur\blurred"
DEBLURRED_DIR = r"algorithms\motion-deblur\deblurred"

PSF_LENGTH = 9
PSF_ANGLE = -15
RL_ITERS = 15  

# ===========================
# MOTION BLUR PSF
# ===========================
def motion_psf(length: int, angle_deg: float) -> np.ndarray:
    length = int(length)
    psf = np.zeros((length, length), dtype=np.float32)
    psf[length // 2, :] = 1.0

    M = cv2.getRotationMatrix2D((length / 2, length / 2), angle_deg, 1.0)
    psf = cv2.warpAffine(psf, M, (length, length))

    psf = np.clip(psf, 0, None)
    psf /= (psf.sum() + 1e-6)
    return psf

def motion_psf_for_image(
    img: np.ndarray,
    base_frac: float = 0.03,
    angle_deg: float = 25.0,
    min_len: int = 9,
    max_len: int = 61
    ) -> np.ndarray:
    """
    Create a motion PSF whose length scales with image resolution.
    base_frac: fraction of min(H, W) to use as blur length.
    """
    h, w = img.shape[:2]
    length = int(base_frac * min(h, w))
    length = max(min_len, min(max_len, length))  # clamp to reasonable range

    # make length odd so kernel is centered
    if length % 2 == 0:
        length += 1

    psf = np.zeros((length, length), dtype=np.float32)
    psf[length // 2, :] = 1.0

    M = cv2.getRotationMatrix2D((length / 2, length / 2), angle_deg, 1.0)
    psf = cv2.warpAffine(psf, M, (length, length))
    psf = np.clip(psf, 0, None)
    psf /= (psf.sum() + 1e-8)
    return psf


# ===========================
# APPLY MOTION BLUR
# ===========================
def apply_motion_blur(img: np.ndarray, psf: np.ndarray) -> np.ndarray:
    img_f = img.astype(np.float32) / 255.0
    blurred = np.zeros_like(img_f)

    for c in range(3):
        blurred[..., c] = cv2.filter2D(img_f[..., c], -1, psf)

    return np.clip(blurred * 255, 0, 255).astype(np.uint8)

# ===========================
# RICHARDSON-LUCY DECONVOLUTION
# ===========================
def rl_deconvolution(
    img: np.ndarray, 
    psf: np.ndarray, 
    iters: int = 30,
    gauss_ksize: int = 5, 
    gauss_sigma: float = 1
    ) -> np.ndarray:
    """
    Richardson-Lucy deconvolution with Gaussian smoothing per iteration.
    Works for COLOR images.
    """

    eps = 1e-8
    psf_flip = psf[::-1, ::-1]

    img_f = img.astype(np.float32) / 255.0
    out = img_f.copy()

    for c in range(3):
        channel = out[..., c]
        y = img_f[..., c]

        for _ in range(iters):
            # Forward blur estimate
            est = cv2.filter2D(channel, -1, psf) + eps

            # RL ratio
            ratio = np.clip(y / est, 0.6, 2.5)

            # Backprojection
            channel *= cv2.filter2D(ratio, -1, psf_flip)

            channel = cv2.GaussianBlur(
                channel,
                (gauss_ksize, gauss_ksize),
                gauss_sigma,
                borderType=cv2.BORDER_REPLICATE
            )

        out[..., c] = channel

    return np.clip(out * 255, 0, 255).astype(np.uint8)

def main():
    os.makedirs(BLURRED_DIR, exist_ok=True)
    os.makedirs(DEBLURRED_DIR, exist_ok=True)

    psf = motion_psf(PSF_LENGTH, PSF_ANGLE)

    files = sorted(os.listdir(RAW_DIR))

    for filename in files:
        if not filename.lower().endswith((".png")):
            continue

        raw_path = os.path.join(RAW_DIR, filename)
        blurred_path = os.path.join(BLURRED_DIR, "blurred" + filename[filename.index("_"):])
        deblurred_path = os.path.join(DEBLURRED_DIR, "deblurred" +  filename[filename.index("_"):])

        print(f"Processing: {filename}")

        # Load 
        img = cv2.imread(raw_path)
        if img is None:
            print(f"Skipping invalid image: {filename}")
            continue

        # Blur
        img = cv2.imread(raw_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Skipping invalid image: {filename}")
            continue

        # PSF scaled to this image's resolution
        psf = motion_psf_for_image(img, base_frac=0.03, angle_deg=PSF_ANGLE)

        blurred = apply_motion_blur(img, psf)
        cv2.imwrite(blurred_path, blurred)

        deblurred = rl_deconvolution(blurred, psf, RL_ITERS)
        cv2.imwrite(deblurred_path, deblurred)

    print("\nAll images processed successfully.")

# ===========================
# RUN
# ===========================
if __name__ == "__main__":
    main()

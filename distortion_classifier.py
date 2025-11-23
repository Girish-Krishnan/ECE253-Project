import os
from pathlib import Path

import cv2
import numpy as np

# thresholds - tune these as needed
BRIGHT_DARK_THRESH = 80
BLUR_LAP_VAR_THRESH = 60
FOG_BRIGHT_THRESH = 150
FOG_CONTRAST_MAX = 40


def compute_distortion_metrics(img_bgr: np.ndarray) -> dict:
    """Returns brightness, contrast, lap_var, fog_index."""
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    brightness = float(img_gray.mean())
    contrast = float(img_gray.std())
    
    lap = cv2.Laplacian(img_gray, cv2.CV_64F)
    lap_var = float(lap.var())
    
    fog_index = brightness - contrast
    
    return {
        "brightness": brightness,
        "contrast": contrast,
        "lap_var": lap_var,
        "fog_index": fog_index,
    }


def classify_distortion(img_bgr: np.ndarray) -> str:
    """Returns 'fog', 'dark', 'blur', or 'clean'."""
    metrics = compute_distortion_metrics(img_bgr)
    brightness = metrics["brightness"]
    contrast = metrics["contrast"]
    lap_var = metrics["lap_var"]

    if brightness < BRIGHT_DARK_THRESH:
        return "dark"
    
    if lap_var < BLUR_LAP_VAR_THRESH:
        return "blur"
    
    if brightness > FOG_BRIGHT_THRESH and contrast < FOG_CONTRAST_MAX:
        return "fog"
    
    return "clean"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--img", required=True)
    args = parser.parse_args()

    if os.path.isdir(args.img):
        img_dir = Path(args.img)
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        files = [f for f in img_dir.iterdir() if f.suffix.lower() in exts]
        
        if not files:
            print(f"No images in {args.img}")
            exit(1)

        counts = {"fog": 0, "dark": 0, "blur": 0, "clean": 0}
        print("filename, predicted_class, brightness, contrast, lap_var, fog_index")
        
        for img_path in sorted(files):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            pred = classify_distortion(img)
            m = compute_distortion_metrics(img)
            counts[pred] += 1
            
            print(f"{img_path.name}, {pred}, {m['brightness']:.2f}, "
                  f"{m['contrast']:.2f}, {m['lap_var']:.2f}, {m['fog_index']:.2f}")

        print("\n=== Summary ===")
        for cls, count in counts.items():
            print(f"{cls}: {count}")
    else:
        img = cv2.imread(args.img)
        if img is None:
            print(f"Failed to load {args.img}")
            exit(1)

        pred = classify_distortion(img)
        m = compute_distortion_metrics(img)
        
        print(f"Image: {args.img}")
        print(f"Class: {pred}")
        print(f"brightness: {m['brightness']:.2f}")
        print(f"contrast: {m['contrast']:.2f}")
        print(f"lap_var: {m['lap_var']:.2f}")
        print(f"fog_index: {m['fog_index']:.2f}")


import os
import random
import logging
import cv2

import config
from motion_blur import motion_blur_strong

def process_folder(cfg, script_dir):
    input_dir = os.path.join(script_dir, cfg.input_dir)
    output_dir = os.path.join(script_dir, cfg.output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    supported = (".jpg", ".jpeg", ".png", ".bmp")

    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(supported):
            continue

        path = os.path.join(input_dir, fname)
        img = cv2.imread(path)
        if img is None:
            logging.warning(f"Skipping invalid: {fname}")
            continue

        degree = random.randint(cfg.blur.min_degree, cfg.blur.max_degree)
        angle = random.uniform(cfg.blur.min_angle, cfg.blur.max_angle)
        logging.info(f"Processing {fname} — degree={degree}, angle={angle:.1f}")

        blurred = motion_blur_strong(img, degree, angle, cfg.blur)
        out_path = os.path.join(output_dir, f"{os.path.splitext(fname)[0]}_blurred.jpg")
        cv2.imwrite(out_path, blurred)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "blur_config.yaml")
    cfg: config.Config = config.load_config(config_path)
    process_folder(cfg, script_dir)

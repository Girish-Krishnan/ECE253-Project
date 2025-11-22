"""
Wrapper module for fast-alpr library.

This module provides a simple interface to run license plate recognition
on OpenCV BGR images.
"""

import os
from pathlib import Path

import cv2
import numpy as np

from fast_alpr import ALPR

# Initialize models once at module import (global variables)
_alpr = ALPR(
    detector_model="yolo-v9-t-384-license-plate-end2end",
    ocr_model="cct-xs-v1-global-model",
)


def run_fast_alpr(img_bgr: np.ndarray) -> str:
    """
    Input: OpenCV BGR uint8 image.
    Output: predicted license plate text string (empty string if nothing detected).

    Args:
        img_bgr: OpenCV BGR image as numpy array (uint8).

    Returns:
        Predicted license plate text string. Returns empty string if no plate is detected
        or if OCR fails.
    """
    # Run detector to get plate bounding boxes
    plate_detections = _alpr.detector.predict(img_bgr)

    # If no plate is found, return empty string
    if not plate_detections:
        return ""

    # Pick the highest-confidence plate box
    best_detection = max(plate_detections, key=lambda d: d.confidence)

    # Crop the plate region
    bbox = best_detection.bounding_box
    x1, y1 = max(bbox.x1, 0), max(bbox.y1, 0)
    x2, y2 = min(bbox.x2, img_bgr.shape[1]), min(bbox.y2, img_bgr.shape[0])
    cropped_plate = img_bgr[y1:y2, x1:x2]

    # Feed cropped plate to OCR model
    ocr_result = _alpr.ocr.predict(cropped_plate)

    # Return OCR text string (empty if OCR fails)
    if ocr_result is None or not ocr_result.text:
        return ""

    return ocr_result.text


if __name__ == "__main__":
    # Test block: read test.jpg if it exists, run run_fast_alpr, print result
    test_image_path = Path("test.jpg")
    if test_image_path.exists():
        img = cv2.imread(str(test_image_path))
        if img is not None:
            predicted_text = run_fast_alpr(img)
            print(f"Predicted license plate text: {predicted_text}")
        else:
            print(f"Failed to load image from {test_image_path}")
    else:
        print(f"Test image {test_image_path} not found. Skipping test.")


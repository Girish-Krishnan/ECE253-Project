import cv2
import numpy as np
import random

import config

def generate_motion_kernel(size, curved=True, intensity_factor=1.5):
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    x, y = center, center

    if curved:
        # Create random 2D curved trajectory
        for _ in range(int(size * intensity_factor)):
            dx = np.random.randn() * 0.8
            dy = np.random.randn() * 0.8
            x = np.clip(x + dx, 0, size - 1)
            y = np.clip(y + dy, 0, size - 1)
            kernel[int(y), int(x)] += 1
    else:
        kernel[center, :] = 1

    kernel /= np.sum(kernel)
    return kernel

def motion_blur_strong(
    image: np.ndarray, 
    degree: int, 
    angle: float, 
    config: config.BlurSettings
) -> np.ndarray:
    """Simulate strong, realistic motion blur."""
    # Ensure input image is float32 for processing
    image_float = image.astype(np.float32)
    
    kernel = generate_motion_kernel(degree, curved=config.curved_motion,
                                    intensity_factor=config.intensity_factor)

    # Rotate kernel
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    kernel = cv2.warpAffine(kernel, M, (degree, degree))

    def apply_single_channel(img_channel):
        return cv2.filter2D(img_channel, -1, kernel)

    if config.color_shift:
        # Slightly different angles per channel for realism
        b, g, r = cv2.split(image_float)
        b_blur = apply_single_channel(b)
        
        # Create rotated kernels for g and r channels
        kernel_g = cv2.warpAffine(kernel, cv2.getRotationMatrix2D((degree / 2, degree / 2), 3, 1), (degree, degree))
        kernel_r = cv2.warpAffine(kernel, cv2.getRotationMatrix2D((degree / 2, degree / 2), -3, 1), (degree, degree))
        
        g_blur = cv2.filter2D(g, -1, kernel_g)
        r_blur = cv2.filter2D(r, -1, kernel_r)
        blurred = cv2.merge([b_blur, g_blur, r_blur])
    else:
        blurred = cv2.filter2D(image_float, -1, kernel)

    # sensor noise
    noise = np.random.normal(0, config.noise_stddev, blurred.shape).astype(np.float32)
    blurred = np.clip(blurred + noise, 0, 255)

    # Blend with original to retain realism - ensure both arrays are same type
    alpha = config.blend_original
    blended = cv2.addWeighted(blurred.astype(np.float32), 1 - alpha, image_float, alpha, 0)
    return np.clip(blended, 0, 255).astype(np.uint8)

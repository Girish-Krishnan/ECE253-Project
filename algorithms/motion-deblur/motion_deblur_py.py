"""
Classical blind motion deblurring using Richardson-Lucy deconvolution
with grid search over motion kernels, plus Wiener filter as a baseline.
"""

import cv2
import numpy as np
from scipy import ndimage
from scipy.signal import convolve2d


def create_motion_psf(length, angle, size=31):
    """Create a linear motion blur PSF kernel.
    
    The PSF is a line segment parameterized by length and angle.
    Such a PSF is specified by two parameters: LEN is the length of the blur 
    and THETA is the angle of motion.
    
    Args:
        length: Length of the blur in pixels (LEN)
        angle: Angle of motion in degrees (THETA), 0 = horizontal, 90 = vertical
        size: Size of the kernel (should be odd, >= length)
    
    Returns:
        Normalized PSF kernel as float32 array
    """
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    
    angle_rad = np.deg2rad(angle)
    
    # Draw line segment
    for i in range(length):
        x = int(center + (i - length/2) * np.cos(angle_rad))
        y = int(center + (i - length/2) * np.sin(angle_rad))
        if 0 <= x < size and 0 <= y < size:
            kernel[y, x] = 1.0
    
    # Normalize so sum = 1
    s = kernel.sum()
    if s > 0:
        kernel /= s
    
    return kernel


def richardson_lucy_deconv(image, psf, num_iter=30, sigma=0.3, pad=15):
    """Richardson-Lucy deconvolution with Gaussian smoothing and boundary padding.
    
    Implements the RL update:
        I^(t+1) = I^(t) * [(B / (I^(t) * k + eps)) * k*]
    
    where k* is the flipped PSF. After each iteration, Gaussian smoothing
    is applied to suppress noise and ringing.
    
    Args:
        image: Blurred image (float [0,1])
        psf: Point spread function kernel
        num_iter: Number of RL iterations
        sigma: Gaussian smoothing sigma (0 to disable)
        pad: Border extension size to prevent boundary artifacts
    
    Returns:
        Deblurred image (float [0,1])
    """
    # Extend image with border replication to prevent Gibbs oscillation
    image_ext = np.pad(image, pad, mode='edge')
    estimate = image_ext.copy()
    
    # Flip PSF for correlation (k*)
    psf_flipped = np.flip(np.flip(psf, 0), 1)
    eps = 1e-10
    
    # Ratio clipping to prevent instability
    r_min, r_max = 0.01, 0.99
    
    for _ in range(num_iter):
        # Convolve estimate with PSF: I^(t) * k
        blurred_est = convolve2d(estimate, psf, mode='same', boundary='symm')
        
        # Compute ratio: B / (I^(t) * k + eps)
        ratio = image_ext / (blurred_est + eps)
        ratio = np.clip(ratio, r_min, r_max)
        
        # Correlate with flipped PSF: ratio * k*
        correction = convolve2d(ratio, psf_flipped, mode='same', boundary='symm')
        
        # Update estimate: I^(t+1) = I^(t) * correction
        estimate = estimate * correction
        
        # Gaussian smoothing to suppress noise and ringing
        if sigma > 0:
            estimate = ndimage.gaussian_filter(estimate, sigma=sigma)
        
        # Clip to valid range
        estimate = np.clip(estimate, 0, 1)
    
    # Crop padding
    estimate = estimate[pad:-pad, pad:-pad]
    return estimate


def wiener_deconv(image, psf, nsr=0.01):
    """Wiener filter deconvolution in frequency domain.
    
    Implements:
        I(u,v) = [H*(u,v) / (|H(u,v)|^2 + NSR)] * B(u,v)
    
    where H is the PSF spectrum, H* is its conjugate, and NSR is the
    noise-to-signal ratio. NSR is the noise-to-signal power ratio of the additive noise.
    
    Args:
        image: Blurred image (float [0,1])
        psf: Point spread function kernel
        nsr: Noise-to-signal ratio (regularization parameter)
    
    Returns:
        Deblurred image (float [0,1])
    """
    h, w = image.shape
    psf_h, psf_w = psf.shape
    
    # Pad PSF to image size
    psf_padded = np.zeros((h, w), dtype=np.float32)
    psf_padded[:psf_h, :psf_w] = psf
    
    # Shift PSF to center for FFT
    psf_padded = np.fft.fftshift(psf_padded)
    
    # FFT of image and PSF
    image_fft = np.fft.fft2(image)
    psf_fft = np.fft.fft2(psf_padded)
    
    # Wiener filter: H* / (|H|^2 + NSR)
    psf_conj = np.conj(psf_fft)
    psf_mag_sq = np.abs(psf_fft) ** 2
    wiener = psf_conj / (psf_mag_sq + nsr)
    
    # Apply filter
    restored_fft = image_fft * wiener
    restored = np.real(np.fft.ifft2(restored_fft))
    
    # Shift back
    restored = np.fft.fftshift(restored)
    
    return np.clip(restored, 0, 1)


def edge_taper(image, gamma=5.0, beta=0.2):
    """Apply edge tapering window to reduce ringing artifacts.
    
    Uses a smooth windowing function that attenuates image borders:
        w(x) = 0.5 * [tanh((x + gamma/2)/beta) - tanh((x - gamma/2)/beta)]
    
    This ensures continuity near image edges before frequency-domain filtering.
    
    Args:
        image: Input image (float [0,1])
        gamma: Window width parameter
        beta: Smoothness parameter
    
    Returns:
        Tapered image (float [0,1])
    """
    h, w = image.shape
    Nx, Ny = w, h
    
    # Create 1D windows
    x = np.linspace(-np.pi, np.pi, Nx)
    w1 = 0.5 * (np.tanh((x + gamma/2) / beta) - np.tanh((x - gamma/2) / beta))
    
    y = np.linspace(-np.pi, np.pi, Ny)
    w2 = 0.5 * (np.tanh((y + gamma/2) / beta) - np.tanh((y - gamma/2) / beta))
    
    # 2D window
    w_2d = np.outer(w2, w1)
    
    return image * w_2d


def deblur_rl_grid_search(image_bgr, method='rl'):
    """Deblur image using grid search over motion kernels.
    
    This implements a grid-based blind motion deblurring algorithm:
    1. Extend image with border replication
    2. Grid search over PSF parameter space (length, angle)
    3. For each kernel, run a few RL iterations on downsampled image
    4. Score each candidate using Laplacian variance (sharpness)
    5. Select best kernel and run full RL deconvolution
    
    Args:
        image_bgr: Input BGR image (uint8)
        method: 'rl' for Richardson-Lucy, 'wiener' for Wiener filter
    
    Returns:
        Deblurred BGR image (uint8)
    """
    h, w = image_bgr.shape[:2]
    
    # Downsample for faster grid search evaluation
    # Use 1/4 scale to speed up kernel evaluation
    scale = 0.25
    h_small = int(h * scale)
    w_small = int(w * scale)
    img_small = cv2.resize(image_bgr, (w_small, h_small))
    
    # Convert to float [0,1] and split channels
    img_small_float = img_small.astype(np.float32) / 255.0
    b_small, g_small, r_small = cv2.split(img_small_float)
    
    # Grid search parameters
    # Typical motion blur lengths: 10-40 pixels
    # Angles: 0-180 degrees (symmetric, so we only need 0-180)
    lengths = [10, 15, 20, 25, 30, 35]
    angles = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165]
    
    best_psf = None
    best_score = -1
    
    # Evaluation iterations (fewer for speed during grid search)
    T_eval = 5
    
    # Grid search: evaluate each kernel
    for length in lengths:
        for angle in angles:
            psf = create_motion_psf(length, angle, size=31)
            if psf.sum() == 0:
                continue
            
            # Quick deconvolution on downsampled image
            if method == 'rl':
                # Run RL on each channel
                b_est = richardson_lucy_deconv(b_small, psf, num_iter=T_eval, sigma=0.3)
                g_est = richardson_lucy_deconv(g_small, psf, num_iter=T_eval, sigma=0.3)
                r_est = richardson_lucy_deconv(r_small, psf, num_iter=T_eval, sigma=0.3)
            else:
                # Wiener filter
                b_est = wiener_deconv(b_small, psf, nsr=0.01)
                g_est = wiener_deconv(g_small, psf, nsr=0.01)
                r_est = wiener_deconv(r_small, psf, nsr=0.01)
            
            # Score: mean Laplacian variance across channels
            # Higher variance = sharper edges = better deblurring
            b_lap = cv2.Laplacian((b_est * 255).astype(np.uint8), cv2.CV_64F)
            g_lap = cv2.Laplacian((g_est * 255).astype(np.uint8), cv2.CV_64F)
            r_lap = cv2.Laplacian((r_est * 255).astype(np.uint8), cv2.CV_64F)
            
            score = (b_lap.var() + g_lap.var() + r_lap.var()) / 3.0
            
            if score > best_score:
                best_score = score
                best_psf = psf
    
    if best_psf is None:
        return image_bgr
    
    # Final deconvolution on full-resolution image
    img_float = image_bgr.astype(np.float32) / 255.0
    b, g, r = cv2.split(img_float)
    
    # More iterations for final result
    T_final = 30
    
    if method == 'rl':
        # Process each channel separately
        b_deblur = richardson_lucy_deconv(b, best_psf, num_iter=T_final, sigma=0.3)
        g_deblur = richardson_lucy_deconv(g, best_psf, num_iter=T_final, sigma=0.3)
        r_deblur = richardson_lucy_deconv(r, best_psf, num_iter=T_final, sigma=0.3)
    else:
        # Wiener filter with edge tapering
        b_tapered = edge_taper(b, gamma=5.0, beta=0.2)
        g_tapered = edge_taper(g, gamma=5.0, beta=0.2)
        r_tapered = edge_taper(r, gamma=5.0, beta=0.2)
        
        b_deblur = wiener_deconv(b_tapered, best_psf, nsr=0.01)
        g_deblur = wiener_deconv(g_tapered, best_psf, nsr=0.01)
        r_deblur = wiener_deconv(r_tapered, best_psf, nsr=0.01)
    
    # Merge channels and convert back to uint8
    result = cv2.merge([b_deblur, g_deblur, r_deblur])
    result = np.clip(result, 0, 1) * 255.0
    return result.astype(np.uint8)

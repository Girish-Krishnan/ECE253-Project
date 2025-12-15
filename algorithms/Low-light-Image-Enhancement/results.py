#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
from os import makedirs
from os.path import join, exists, basename, splitext

import cv2
import numpy as np
from tqdm import tqdm

from scipy.ndimage import convolve
from scipy.sparse import coo_matrix, diags, csr_matrix
from scipy.sparse.linalg import spsolve

def create_spatial_affinity_kernel(spatial_sigma: float, size: int = 15) -> np.ndarray:
    """
    Vectorized Gaussian kernel centered at (size//2, size//2).
    """
    c = size // 2
    yy, xx = np.mgrid[0:size, 0:size]
    d2 = (yy - c) ** 2 + (xx - c) ** 2
    kernel = np.exp(-0.5 * d2 / (spatial_sigma ** 2))
    return kernel.astype(np.float64)


def compute_smoothness_weights(L: np.ndarray, x: int, kernel: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """
    Computes direction-dependent smoothness weights.
    x=1 -> horizontal, x=0 -> vertical (matches your original intent).
    """
    # Sobel derivative: dx if x==1 else dy
    Lp = cv2.Sobel(L, cv2.CV_64F, int(x == 1), int(x == 0), ksize=1)

    # spatially weighted normalization
    T = convolve(np.ones_like(L, dtype=np.float64), kernel, mode="constant")
    denom = np.abs(convolve(Lp, kernel, mode="constant")) + eps
    T = T / denom

    return T / (np.abs(Lp) + eps)


def build_inhomogeneous_laplacian(wx: np.ndarray, wy: np.ndarray) -> csr_matrix:
    """
    Build the 5-point spatially inhomogeneous Laplacian F (size N x N) in a fully vectorized way,
    replicating the *directed-weight* behavior in your original loop:

      For pixel p=(i,j), neighbor q=(k,l):
        offdiag F[p,q] = -weight where weight uses wx/wy at (k,l) (the neighbor location).
        diag F[p,p] = sum(outgoing weights from p.

    This matches your code that did:
      weight = wx[k,l] if x else wy[k,l]
      data.append(-weight)
      diag += weight

    Returns CSR for fast solving.
    """
    n, m = wx.shape
    N = n * m

    # Flattened index map
    idx = np.arange(N, dtype=np.int64).reshape(n, m)

    rows = []
    cols = []
    data = []

    diag_accum = np.zeros(N, dtype=np.float64)

    # --- Up neighbor (i-1, j), uses wy at neighbor (i-1,j)
    if n > 1:
        p = idx[1:, :]          # pixels that have an up neighbor
        q = idx[:-1, :]         # their up neighbors
        w = wy[:-1, :].ravel()  # weight at neighbor
        p_flat = p.ravel()
        q_flat = q.ravel()
        rows.append(p_flat); cols.append(q_flat); data.append(-w)
        np.add.at(diag_accum, p_flat, w)

    # --- Down neighbor (i+1, j), uses wy at neighbor (i+1,j)
    if n > 1:
        p = idx[:-1, :]
        q = idx[1:, :]
        w = wy[1:, :].ravel()
        p_flat = p.ravel()
        q_flat = q.ravel()
        rows.append(p_flat); cols.append(q_flat); data.append(-w)
        np.add.at(diag_accum, p_flat, w)

    # --- Left neighbor (i, j-1), uses wx at neighbor (i, j-1)
    if m > 1:
        p = idx[:, 1:]
        q = idx[:, :-1]
        w = wx[:, :-1].ravel()
        p_flat = p.ravel()
        q_flat = q.ravel()
        rows.append(p_flat); cols.append(q_flat); data.append(-w)
        np.add.at(diag_accum, p_flat, w)

    # --- Right neighbor (i, j+1), uses wx at neighbor (i, j+1)
    if m > 1:
        p = idx[:, :-1]
        q = idx[:, 1:]
        w = wx[:, 1:].ravel()
        p_flat = p.ravel()
        q_flat = q.ravel()
        rows.append(p_flat); cols.append(q_flat); data.append(-w)
        np.add.at(diag_accum, p_flat, w)

    # Stack COO entries
    if rows:
        row = np.concatenate(rows)
        col = np.concatenate(cols)
        dat = np.concatenate(data)
    else:
        row = np.array([], dtype=np.int64)
        col = np.array([], dtype=np.int64)
        dat = np.array([], dtype=np.float64)

    # Add diagonal
    diag_row = np.arange(N, dtype=np.int64)
    diag_col = diag_row.copy()
    row = np.concatenate([row, diag_row])
    col = np.concatenate([col, diag_col])
    dat = np.concatenate([dat, diag_accum])

    F = coo_matrix((dat, (row, col)), shape=(N, N)).tocsr()
    return F


def refine_illumination_map_linear(L: np.ndarray, gamma: float, lambda_: float, kernel: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """
    Refine illumination map by solving:
      (I + lambda * F) L_refined = L
    then apply gamma correction.

    This uses vectorized sparse assembly for F.
    """
    wx = compute_smoothness_weights(L, x=1, kernel=kernel, eps=eps)
    wy = compute_smoothness_weights(L, x=0, kernel=kernel, eps=eps)

    n, m = L.shape
    N = n * m
    L_1d = L.reshape(-1).astype(np.float64, copy=False)

    F = build_inhomogeneous_laplacian(wx, wy)
    A = diags(np.ones(N, dtype=np.float64), 0, format="csr") + (lambda_ * F)

    # Solve
    L_refined = spsolve(A, L_1d).reshape(n, m)

    # Gamma correction
    L_refined = np.clip(L_refined, eps, 1.0) ** gamma
    return L_refined


def correct_underexposure(im01: np.ndarray, gamma: float, lambda_: float, kernel: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """
    Retinex-style correction: estimate illumination L = max(im), refine it, then divide.
    Input im01 is float image in [0,1].
    """
    L = np.max(im01, axis=-1)
    L_refined = refine_illumination_map_linear(L, gamma, lambda_, kernel, eps)

    L3 = L_refined[..., None]
    return im01 / np.clip(L3, eps, 1.0)


def fuse_multi_exposure_images(im01: np.ndarray, under: np.ndarray, over: np.ndarray,
                              bc: float = 1, bs: float = 1, be: float = 1) -> np.ndarray:
    """
    Exposure fusion via OpenCV MergeMertens.
    """
    merge_mertens = cv2.createMergeMertens(bc, bs, be)
    images = [np.clip(x * 255.0, 0, 255).astype(np.uint8) for x in (im01, under, over)]
    fused = merge_mertens.process(images)  # float32 [0,1]
    return fused.astype(np.float64, copy=False)


def enhance_image_exposure(im_bgr: np.ndarray,
                           gamma: float,
                           lambda_: float,
                           dual: bool = True,
                           sigma: float = 3,
                           kernel_size: int = 15,
                           bc: float = 1,
                           bs: float = 1,
                           be: float = 1,
                           eps: float = 1e-3,
                           max_size: int | None = None) -> np.ndarray:
    """
    Main API: enhance BGR uint8 image.
    If max_size is set and image is larger, we downscale for the optimization step and upscale result.
    """
    if im_bgr is None or im_bgr.size == 0:
        raise ValueError("Empty image passed to enhance_image_exposure().")

    orig_h, orig_w = im_bgr.shape[:2]

    # Optional downscale for speed
    scale = 1.0
    if max_size is not None:
        max_dim = max(orig_h, orig_w)
        if max_dim > max_size:
            scale = max_size / float(max_dim)
            new_w = max(1, int(round(orig_w * scale)))
            new_h = max(1, int(round(orig_h * scale)))
            im_proc = cv2.resize(im_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            im_proc = im_bgr
    else:
        im_proc = im_bgr

    kernel = create_spatial_affinity_kernel(spatial_sigma=float(sigma), size=int(kernel_size))

    im01 = im_proc.astype(np.float64) / 255.0
    under = correct_underexposure(im01, gamma, lambda_, kernel, eps)

    if dual:
        inv = 1.0 - im01
        over = 1.0 - correct_underexposure(inv, gamma, lambda_, kernel, eps)
        out01 = fuse_multi_exposure_images(im01, under, over, bc, bs, be)
    else:
        out01 = under

    out_u8 = np.clip(out01 * 255.0, 0, 255).astype(np.uint8)

    # Upscale back if we downscaled
    if out_u8.shape[0] != orig_h or out_u8.shape[1] != orig_w:
        out_u8 = cv2.resize(out_u8, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    return out_u8


# -----------------------------
# CLI / batch processing
# -----------------------------

def list_images(folder: str, recursive: bool, exts: tuple[str, ...]) -> list[str]:
    pattern = "**/*" if recursive else "*"
    files: list[str] = []
    for e in exts:
        files.extend(glob.glob(join(folder, f"{pattern}.{e}"), recursive=recursive))
        files.extend(glob.glob(join(folder, f"{pattern}.{e.upper()}"), recursive=recursive))
    files.sort()
    return files


def main():
    parser = argparse.ArgumentParser(
        description="Fast LIME/DUAL low-light enhancement (single-file optimized)."
    )
    parser.add_argument("-f", "--folder", default="./demo/", type=str, help="Folder containing input images.")
    parser.add_argument("-o", "--out", default=None, type=str, help="Output folder (default: <folder>/enhanced).")
    parser.add_argument("--recursive", action="store_true", help="Recursively search for images.")
    parser.add_argument("--exts", default="png,jpg,jpeg,bmp", type=str, help="Comma-separated extensions.")

    parser.add_argument("-g", "--gamma", default=0.6, type=float, help="Gamma correction parameter.")
    parser.add_argument("-l", "--lambda_", dest="lambda_", default=0.15, type=float, help="Lambda for refinement.")
    parser.add_argument("--lime", action="store_true", help="Use LIME only (no dual fusion). Default is DUAL.")
    parser.add_argument("-s", "--sigma", default=3, type=float, help="Spatial sigma for affinity kernel.")
    parser.add_argument("--kernel-size", default=15, type=int, help="Affinity kernel size (odd recommended).")

    parser.add_argument("-bc", default=1.0, type=float, help="Mertens contrast weight.")
    parser.add_argument("-bs", default=1.0, type=float, help="Mertens saturation weight.")
    parser.add_argument("-be", default=1.0, type=float, help="Mertens well-exposedness weight.")
    parser.add_argument("-eps", default=1e-3, type=float, help="Stability epsilon.")

    parser.add_argument("--max-size", default=None, type=int,
                        help="If set, downscale large images so max(H,W)=max-size for the optimization step (faster).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite outputs if they already exist.")

    args = parser.parse_args()

    in_dir = args.folder
    out_dir = args.out or join(in_dir, "enhanced")
    if not exists(out_dir):
        makedirs(out_dir)

    exts = tuple(x.strip() for x in args.exts.split(",") if x.strip())
    files = list_images(in_dir, args.recursive, exts)
    if not files:
        raise SystemExit(f"No images found in: {in_dir} (exts={exts}, recursive={args.recursive})")

    dual = not args.lime
    method = "DUAL" if dual else "LIME"

    for path in tqdm(files, desc=f"Enhancing images ({method})"):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            continue

        out_img = enhance_image_exposure(
            img,
            gamma=args.gamma,
            lambda_=args.lambda_,
            dual=dual,
            sigma=args.sigma,
            kernel_size=args.kernel_size,
            bc=args.bc,
            bs=args.bs,
            be=args.be,
            eps=args.eps,
            max_size=args.max_size,
        )

        base = basename(path)
        name, ext = splitext(base)
        out_path = join(out_dir, f"{name}{ext}")

        if (not args.overwrite) and exists(out_path):
            continue

        cv2.imwrite(out_path, out_img)


if __name__ == "__main__":
    main()

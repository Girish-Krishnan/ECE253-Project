#!/usr/bin/env python3
"""
DehazeNet batch dehazing (fast + MPS + optional downscale+upscale)

- Uses your pretrained DehazeNet weights unchanged.
- Vectorized patch processing via unfold/fold (much faster than per-patch loops).
- Uses Apple Silicon MPS if available, otherwise CPU.
- Optional: downscale large images to a max_side for speed/memory, then upscale back.
- Preserves aspect ratio; output is resized back to original size (if enabled).

Usage:
  python dehazenet_batch.py \
      --weights defog4_noaug.pth \
      --input_dir "../../license_plate_restoration/Foggy Images/" \
      --output_dir "../../license_plate_restoration/Foggy Images DehazeNet/" \
      --max_side 1536

Notes:
- Like your original, the *working* image is cropped to a multiple of patch_size (default 16).
  If you upscale back to original size, you won't notice the crop, but edge behavior may differ.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF
from tqdm import tqdm


# -----------------------------
# Model (unchanged architecture)
# -----------------------------

class BRelu(nn.Hardtanh):
    def __init__(self, inplace: bool = False):
        super().__init__(0.0, 1.0, inplace)


class DehazeNet(nn.Module):
    def __init__(self, input: int = 16, groups: int = 4):
        super().__init__()
        self.input = input
        self.groups = groups
        self.conv1 = nn.Conv2d(3, self.input, kernel_size=5)
        self.conv2 = nn.Conv2d(4, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(4, 16, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(4, 16, kernel_size=7, padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=7, stride=1)
        self.conv5 = nn.Conv2d(48, 1, kernel_size=6)
        self.brelu = BRelu()

        # Keep init (won't matter after loading pretrained weights)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def Maxout(self, x: torch.Tensor, groups: int) -> torch.Tensor:
        n, c, h, w = x.shape
        x = x.view(n, groups, c // groups, h, w)
        x, _ = torch.max(x, dim=2)
        return x  # [N, groups, H, W]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.Maxout(out, self.groups)
        out1 = self.conv2(out)
        out2 = self.conv3(out)
        out3 = self.conv4(out)
        y = torch.cat((out1, out2, out3), dim=1)
        y = self.maxpool(y)
        y = self.conv5(y)
        y = self.brelu(y)
        return y.view(y.shape[0], -1)  # [N, 1]


# -----------------------------
# Fast defog (vectorized + MPS)
# -----------------------------

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def pick_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resize_to_max_side(img: Image.Image, max_side: int | None) -> Tuple[Image.Image, Tuple[int, int]]:
    """Resize keeping aspect ratio so max(w,h) <= max_side. Returns resized image + original size (w,h)."""
    orig_w, orig_h = img.size
    if max_side is None:
        return img, (orig_w, orig_h)
    if max(orig_w, orig_h) <= max_side:
        return img, (orig_w, orig_h)

    scale = max_side / float(max(orig_w, orig_h))
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))
    img_rs = img.resize((new_w, new_h), resample=Image.BICUBIC)
    return img_rs, (orig_w, orig_h)


@torch.inference_mode()
def defog_fast(
    img_path: str | Path,
    net: nn.Module,
    out_path: str | Path,
    *,
    device: torch.device,
    patch_size: int = 16,
    top_percent: float = 0.01,
    eps: float = 1e-4,
    max_side: int | None = None,
    upscale_back: bool = True,
    max_batch_patches: int = 65536,  # batches only the net forward, not unfold storage
) -> Path:
    img_path = Path(img_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    img = Image.open(img_path).convert("RGB")

    # Optional downscale for speed/memory
    img_work, (orig_w, orig_h) = resize_to_max_side(img, max_side)

    # To tensor [1,3,H,W] in 0..1
    img2 = TF.to_tensor(img_work).unsqueeze(0)
    _, _, H, W = img2.shape

    # Crop to multiple of patch_size (same as your original integer division behavior)
    Hc = (H // patch_size) * patch_size
    Wc = (W // patch_size) * patch_size
    if Hc <= 0 or Wc <= 0:
        raise ValueError(f"Image too small after cropping to patch_size={patch_size}: {img_path} ({H}x{W})")
    img2 = img2[:, :, :Hc, :Wc]

    # Normalize for net
    img1 = (img2 - IMAGENET_MEAN) / IMAGENET_STD

    img1 = img1.to(device)
    img2d = img2.to(device)

    ps = patch_size

    # Patchify: [1, 3*ps*ps, L]
    patches1 = F.unfold(img1, kernel_size=ps, stride=ps)
    patches2 = F.unfold(img2d, kernel_size=ps, stride=ps)
    L = patches1.shape[-1]

    # Net expects [N,3,ps,ps]
    patches1_nchw = patches1.squeeze(0).transpose(0, 1).contiguous().view(L, 3, ps, ps)

    # Forward in batches to avoid huge activation memory
    t_out = []
    for start in range(0, L, max_batch_patches):
        t_out.append(net(patches1_nchw[start : start + max_batch_patches]))
    t = torch.cat(t_out, dim=0).squeeze(1)  # [L]

    # smallest k transmissions
    k = max(1, int(L * top_percent))
    _, idx = torch.topk(t, k=k, largest=False)

    # a0 among selected patches in original (unnormalized) space
    a0 = patches2[:, :, idx].max()

    # Apply reconstruction: (patch - a0*(1-t))/t
    t_clamped = t.clamp_min(eps).view(1, 1, L)
    patches2_adj = (patches2 - a0 * (1.0 - t_clamped)) / t_clamped
    patches2_adj = patches2_adj.clamp(0.0, 1.0)

    # Fold back
    out = F.fold(patches2_adj, output_size=(Hc, Wc), kernel_size=ps, stride=ps)
    out_img = TF.to_pil_image(out.squeeze(0).cpu())

    # Optionally upscale back to original image size
    if upscale_back:
        out_img = out_img.resize((orig_w, orig_h), resample=Image.BICUBIC)

    out_img.save(out_path)
    return out_path


def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True, help="Path to defog4_noaug.pth")
    ap.add_argument("--input_dir", type=str, required=True, help="Directory containing images")
    ap.add_argument("--output_dir", type=str, required=True, help="Directory to write outputs")
    ap.add_argument("--patch_size", type=int, default=16)
    ap.add_argument("--top_percent", type=float, default=0.01)
    ap.add_argument("--eps", type=float, default=1e-4)
    ap.add_argument("--max_side", type=int, default=1536,
                    help="Downscale working image so max(w,h)<=max_side. Use 0 to disable.")
    ap.add_argument("--no_upscale_back", action="store_true",
                    help="If set, output stays at working resolution (possibly downscaled).")
    ap.add_argument("--force_cpu", action="store_true", help="Disable MPS even if available.")
    ap.add_argument("--max_batch_patches", type=int, default=65536, help="Batch size for patch forward pass.")
    args = ap.parse_args()

    max_side = None if args.max_side <= 0 else args.max_side
    upscale_back = not args.no_upscale_back

    device = pick_device(force_cpu=args.force_cpu)
    print(f"Device: {device}")

    net = DehazeNet()
    state = torch.load(args.weights, map_location="cpu")
    net.load_state_dict(state)
    net.to(device).eval()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in in_dir.iterdir() if p.is_file() and is_image_file(p)])
    if not images:
        raise SystemExit(f"No images found in: {in_dir}")

    for p in tqdm(images, desc="Dehazing", unit="image"):
        out_path = out_dir / p.name
        try:
            saved = defog_fast(
                p, net, out_path,
                device=device,
                patch_size=args.patch_size,
                top_percent=args.top_percent,
                eps=args.eps,
                max_side=max_side,
                upscale_back=upscale_back,
                max_batch_patches=args.max_batch_patches,
            )
        except Exception as e:
            print(f"[ERROR] {p}: {e}")
            continue


if __name__ == "__main__":
    main()

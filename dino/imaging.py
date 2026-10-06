"""Image helpers: the foreground-preprocessing of the official TRELLIS.2 pipeline *without* a background-removal model.

``preprocess_with_alpha`` is a faithful port of the ``has_alpha`` branch of ``Trellis2ImageTo3DPipeline.preprocess_image``
(microsoft/TRELLIS.2, MIT): downscale to <= 1024 px, crop a square around the foreground (alpha > 0.8), then premultiply the
colour by alpha so the background becomes black.  The mask can come from *any* background-removal node (use a permissively
licensed one: the pipeline's own default, BRIA RMBG-2.0, is CC BY-NC).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image


def apply_mask(image: Image.Image, mask: Optional[np.ndarray]) -> Image.Image:
    """Return an RGBA image whose alpha is ``mask`` (float or uint8, [H,W], resized to the image if needed)."""
    rgb = image.convert("RGB")
    if mask is None:
        return image if image.mode == "RGBA" else rgb.convert("RGBA")
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[..., 0]
    m = (np.clip(m.astype(np.float32), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8) if m.dtype != np.uint8 else m
    if m.shape != (rgb.height, rgb.width):
        m = np.asarray(Image.fromarray(m).resize(rgb.size, Image.Resampling.BILINEAR))
    out = rgb.convert("RGBA")
    out.putalpha(Image.fromarray(m))
    return out


def main_object_ratio(alpha: np.ndarray) -> float:
    """Size of the largest connected foreground region's box relative to the box of all foreground pixels (1.0 = one object)."""
    from scipy import ndimage
    fg = alpha > 0.8 * 255
    if not fg.any():
        return 1.0
    lab, n = ndimage.label(fg)
    if n <= 1:
        return 1.0
    biggest = np.argmax(np.bincount(lab.ravel())[1:]) + 1
    ys, xs = np.nonzero(fg)
    by, bx = np.nonzero(lab == biggest)
    full = max(xs.max() - xs.min(), ys.max() - ys.min()) + 1
    main = max(bx.max() - bx.min(), by.max() - by.min()) + 1
    return float(main) / float(full)


def preprocess_with_alpha(image: Image.Image, largest_object_only: bool = False) -> Image.Image:
    """Port of the ``has_alpha`` path of the official ``preprocess_image``. ``image`` must be RGBA with a real alpha.

    ``largest_object_only``: crop around the largest connected foreground region instead of every foreground pixel (stray mask
    regions such as particles otherwise enlarge the square crop and shrink the object)."""
    if image.mode != "RGBA":
        raise ValueError("preprocess_with_alpha needs an RGBA image (connect a MASK or use an image with alpha)")
    scale = min(1, 1024 / max(image.size))
    if scale < 1:
        image = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS)
    out = np.array(image)
    alpha = out[:, :, 3]
    fg = alpha > 0.8 * 255
    if largest_object_only and fg.any():
        from scipy import ndimage
        lab, n = ndimage.label(fg)
        if n > 1:
            fg = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    ys, xs = np.nonzero(fg)
    if ys.size == 0:
        raise ValueError("The mask is empty (no pixel with alpha > 0.8): nothing to texture from")
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    size = int(max(x1 - x0, y1 - y0))
    box = (cx - size // 2, cy - size // 2, cx + size // 2, cy + size // 2)
    cropped = np.array(image.crop(box)).astype(np.float32) / 255
    rgb = cropped[:, :, :3] * cropped[:, :, 3:4]
    return Image.fromarray((rgb * 255).astype(np.uint8))

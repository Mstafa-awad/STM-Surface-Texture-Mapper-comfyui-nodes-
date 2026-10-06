"""Stage 1 - DINOv3 patch tokens of the reference image(s)."""
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

from ..core.types import DinoFeatures
from ..runtime.stage import log, stage_scope
from ..runtime.vram import current_plan
from .imaging import apply_mask, main_object_ratio, preprocess_with_alpha


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    t = image.detach().cpu()
    if t.ndim == 4:
        if t.shape[0] != 1:
            raise ValueError(f"expected a batch of 1, got {t.shape[0]}")
        t = t[0]
    return Image.fromarray((t.numpy() * 255.0).clip(0, 255).astype(np.uint8))


def _mask_for_view(mask: Optional[torch.Tensor], i: int) -> Optional[np.ndarray]:
    if mask is None:
        return None
    m = mask.detach().cpu().float()
    if m.ndim == 2:
        return m.numpy()
    if m.ndim == 4:
        m = m[:, 0] if m.shape[1] == 1 else m[..., 0]
    return m[min(i, m.shape[0] - 1)].numpy()


def pad_to_square(img: Image.Image) -> Image.Image:
    """Centre a non-square image on a square canvas filled with its own border colour (DINOv3 would otherwise squash it)."""
    w, h = img.size
    if w == h:
        return img
    rgb = np.asarray(img.convert("RGB"))
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    fill = tuple(int(c) for c in np.median(border, axis=0))
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(img.convert("RGB"), ((side - w) // 2, (side - h) // 2))
    return canvas


def images_from_comfy(images: torch.Tensor, mask: Optional[torch.Tensor] = None, preprocess: bool = False,
                      max_views: int = 4, largest_object_only: bool = False) -> List[Image.Image]:
    """IMAGE batch (+ optional MASK) -> PIL views; with a mask / ``preprocess`` the official crop + black background is applied."""
    if not isinstance(images, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor for IMAGE, got {type(images)}")
    if images.ndim == 3:
        images = images.unsqueeze(0)
    if images.ndim != 4:
        raise ValueError(f"Unsupported IMAGE tensor shape: {tuple(images.shape)}")
    views = [tensor_to_pil(images[i:i + 1]) for i in range(min(int(images.shape[0]), int(max_views)))]
    if mask is None and not preprocess:
        return [pad_to_square(v) for v in views]
    out = []
    for i, img in enumerate(views):
        rgba = apply_mask(img, _mask_for_view(mask, i)) if (mask is not None or img.mode != "RGBA") else img
        if rgba.mode == "RGBA" and not (np.asarray(rgba)[:, :, 3] < 255).any():
            raise ValueError("preprocess_image needs transparency: connect a MASK or use an image with a real alpha channel "
                             "(no background-removal model is bundled; use a permissively licensed rembg node).")
        ratio = main_object_ratio(np.asarray(rgba)[:, :, 3])
        if ratio < 0.7 and not largest_object_only:
            log(f"WARNING: view {i + 1}: the mask has separate regions far from the main object (main object = {ratio:.0%} of the crop box); "
                "they make the object small in the crop. Enable 'crop_to_main_object' or clean the mask.")
        out.append(preprocess_with_alpha(rgba, largest_object_only))
    return out


def preview_images(views: List[Image.Image], image_size: int) -> torch.Tensor:
    """The views exactly as the DINOv3 model receives them (resized to ``image_size`` squared) as an IMAGE batch [V,S,S,3]."""
    size = int(image_size)
    arrays = [np.asarray(v.convert("RGB").resize((size, size), Image.LANCZOS), dtype=np.uint8) for v in views]
    return torch.from_numpy(np.stack(arrays)).to(torch.float32).div_(255.0)


@torch.no_grad()
def extract_dino_features(models, images: List[Image.Image], image_size: int = 1024) -> DinoFeatures:
    device = models.device
    plan = current_plan(device)
    with stage_scope("DINOv3", device):
        extractor = models.acquire_dino(plan, bf16=int(image_size) >= 2048)          # bf16 keeps the fused attention kernels usable at 16k+ tokens
        try:
            extractor.image_size = int(image_size)
            extractor.to(device)
            feats = extractor(images)
            extractor.cpu()
        finally:
            models.release("dino", extractor, plan)
    tokens = feats.float().cpu()
    if tokens.ndim == 2:
        tokens = tokens.unsqueeze(0)
    if tokens.shape[0] > 1:                                  # several views: one token sequence
        tokens = tokens.reshape(1, -1, tokens.shape[-1])
    return DinoFeatures(tokens, int(image_size), [np.asarray(i.convert("RGB")) for i in images])

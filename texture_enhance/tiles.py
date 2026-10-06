"""Overlapping tiles with cosine feathering: a view of any size goes through an image model in fixed-size tiles and is put back
together without visible seams.  Tiles are spread evenly (as few as the overlap allows); tiles with no object pixels are not
emitted (no model time spent on background).

``cancel_lighting``: every tile is emitted twice, the second time turned upside down (180 deg).  Image models light what they
paint from the top of the picture; turned back, the second pass is lit from the opposite side, so averaging the two cancels
the painted light and relief shading while the detail both passes agree on stays.

``keep_colors`` (merge): the merged views are the original views plus what the model really added - new detail and new
coloured marks (letters, symbols, ornaments from a reference image) - while its greying, tinting or re-lighting of what was
already there is taken out.
"""
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class TileLayout:
    shape: Tuple[int, int, int, int]                 # N, H, W, C of the original batch
    tile: int
    overlap: int
    boxes: List[Tuple[int, int, int, int, int, int]]  # (image index, y0, x0, height, width, rotated) per emitted tile, list order
    original: torch.Tensor                           # [N,H,W,C] CPU, fills the regions of skipped tiles


def tile_starts(length: int, tile: int, overlap: int) -> List[int]:
    """Evenly spread start positions covering ``length`` with tiles of ``tile`` that overlap by at least ``overlap``."""
    if length <= tile:
        return [0]
    n = max(2, math.ceil((length - overlap) / max(1, tile - overlap)))
    return sorted({round(i * (length - tile) / (n - 1)) for i in range(n)})


def _window(h: int, w: int, y0: int, x0: int, H: int, W: int, ov_y: int, ov_x: int) -> torch.Tensor:
    """Feathering weights: cosine ramps on edges shared with a neighbour, flat on image borders."""
    def ramp(n, k, at_start, at_end):
        r = torch.ones(n)
        k = min(k, n // 2)
        if k > 0:
            t = 0.5 - 0.5 * torch.cos(torch.linspace(0, torch.pi, k + 2)[1:-1])
            if not at_start:
                r[:k] = t
            if not at_end:
                r[n - k:] = t.flip(0)
        return r
    wy = ramp(h, ov_y, y0 == 0, y0 + h >= H)
    wx = ramp(w, ov_x, x0 == 0, x0 + w >= W)
    return (wy[:, None] * wx[None, :]).clamp(min=1e-4)


def _overlap(starts: List[int], tile: int) -> int:
    """Smallest overlap between neighbouring tiles (the feathering width)."""
    if len(starts) < 2:
        return 0
    return min(tile - (b - a) for a, b in zip(starts, starts[1:]))


def _grid(H, W, tile, overlap):
    th, tw = min(tile, H), min(tile, W)
    ys, xs = tile_starts(H, th, overlap), tile_starts(W, tw, overlap)
    return th, tw, ys, xs, _overlap(ys, th), _overlap(xs, tw)


def split_tiles(images: torch.Tensor, tile: int = 1024, overlap: int = 128, masks: Optional[torch.Tensor] = None,
                min_coverage: float = 0.001, cancel_lighting: bool = False) -> Tuple[List[torch.Tensor], TileLayout]:
    imgs = images.detach().cpu().float()
    N, H, W, C = imgs.shape
    th, tw, ys, xs, _, _ = _grid(H, W, tile, overlap)
    boxes, tiles = [], []
    for n in range(N):
        for y0 in ys:
            for x0 in xs:
                if masks is not None and float(masks[n, y0:y0 + th, x0:x0 + tw].float().mean()) < min_coverage:
                    continue
                t = imgs[n:n + 1, y0:y0 + th, x0:x0 + tw].contiguous()
                boxes.append((n, y0, x0, th, tw, 0))
                tiles.append(t)
                if cancel_lighting:
                    boxes.append((n, y0, x0, th, tw, 1))
                    tiles.append(torch.rot90(t, 2, dims=(1, 2)).contiguous())
    return tiles, TileLayout((N, H, W, C), int(tile), int(overlap), boxes, imgs)


_LUMA = (0.299, 0.587, 0.114)                         # BT.601 Y'


def restore_colors(refined: torch.Tensor, original: torch.Tensor, detail_size: float = 16.0) -> torch.Tensor:
    """[N,H,W,3] refined, [N,H,W,3] original -> the original image (its colours, brightness and edges) plus what the refiner
    really added: new detail, sharper edges, more contrast in the structure, and new marks, letters and symbols in their
    own colours.

    The refiner's change is split by a colour guided filter (:func:`project.guided_residual`, windows of 1.5 x ``detail_size``
    pixels of a 1024 view): the part that is a smooth function of the original colours - greying, tinting, re-lighting,
    re-colouring what is there, flattening edges - is taken out; the rest stays at full strength, so new features keep
    their brightness and colour (no halos).  Contrast the refiner added to the original's own structure is kept too.
    Identical inputs give the original back exactly."""
    from ..core.render import smooth
    from .project import guided_residual
    N, H, W, _ = refined.shape
    sig = max(4.0, 1.5 * float(detail_size) * max(H, W) / 1024)
    w = torch.tensor(_LUMA)
    out = torch.empty_like(original[..., :3], dtype=torch.float32)
    for n in range(N):
        org = original[n, ..., :3].float().permute(2, 0, 1)
        d = refined[n, ..., :3].float().permute(2, 0, 1) - org
        if not bool(d.abs().max() > 0):
            out[n] = org.permute(1, 2, 0)
            continue
        new = guided_residual(org, d, sig)                                    # not a colour mapping of the original: new
        yo = (org * w[:, None, None]).sum(0, keepdim=True)
        dy = (d * w[:, None, None]).sum(0, keepdim=True)
        mo, md = smooth(yo, sig), smooth(dy, sig)
        var = (smooth(yo * yo, sig) - mo * mo).clamp(min=0)
        gain = ((smooth(yo * dy, sig) - mo * md) / (var + 1e-4)).clamp(0.0, 1.0)      # contrast added to its own structure
        out[n] = (org + new + gain * (yo - mo)).clamp(0, 1).permute(1, 2, 0)
    return out


def merge_tiles(tiles: List[torch.Tensor], layout: TileLayout, keep_colors: bool = False, detail_size: float = 16.0) -> torch.Tensor:
    N, H, W, C = layout.shape
    if len(tiles) != len(layout.boxes):
        raise ValueError(f"Got {len(tiles)} tiles for a layout of {len(layout.boxes)}: connect the tiles of the matching 'STM — Split Tiles'.")
    th, tw, ys, xs, ov_y, ov_x = _grid(H, W, layout.tile, layout.overlap)
    acc = torch.zeros((N, H, W, C))
    wsum = torch.zeros((N, H, W, 1))
    emitted = set()
    for t, box in zip(tiles, layout.boxes):
        n, y0, x0, h, w = box[:5]
        rotated = len(box) > 5 and box[5]
        t = t.detach().cpu().float()
        t = t[0] if t.ndim == 4 else t
        if t.shape[0] != h or t.shape[1] != w:                    # the model changed the size: resample back
            t = F.interpolate(t.permute(2, 0, 1)[None], size=(h, w), mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
        if rotated:
            t = torch.rot90(t, 2, dims=(0, 1))
        wgt = _window(h, w, y0, x0, H, W, ov_y, ov_x)[:, :, None]
        acc[n, y0:y0 + h, x0:x0 + w] += t[:, :, :C] * wgt
        wsum[n, y0:y0 + h, x0:x0 + w] += wgt
        emitted.add((n, y0, x0))
    for n in range(N):                                           # skipped (background) tiles keep the original content
        for y0 in ys:
            for x0 in xs:
                if (n, y0, x0) not in emitted:
                    wgt = _window(th, tw, y0, x0, H, W, ov_y, ov_x)[:, :, None]
                    acc[n, y0:y0 + th, x0:x0 + tw] += layout.original[n, y0:y0 + th, x0:x0 + tw] * wgt
                    wsum[n, y0:y0 + th, x0:x0 + tw] += wgt
    merged = (acc / wsum.clamp(min=1e-8)).clamp(0, 1)
    if keep_colors and C >= 3:
        merged = torch.cat([restore_colors(merged[..., :3], layout.original[..., :3], detail_size), merged[..., 3:]], dim=-1)
    return merged

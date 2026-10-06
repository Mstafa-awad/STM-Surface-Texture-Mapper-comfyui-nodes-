"""Texture-space hole filling: multiscale pull-push.

* pull: pyramid of valid-weighted averages (2x2 box, ``ceil`` sizes so odd sizes work);
* push: walk back down, replacing only the *invalid* texels by the bilinearly upsampled value of the filled coarser level.

Valid texels are never modified; all channels are filled in one pass; O(N) work on any device.  Memory: the finest level is
never materialised, channels are processed in groups and the result can be written in place.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F

__all__ = ["pull_push_fill", "estimate_fill_bytes"]


def estimate_fill_bytes(channels: int, height: int, width: int, channel_group: int = 3) -> int:
    """Extra working set of :func:`pull_push_fill` (beyond the input tensor) in bytes: three channel-group planes
    (product / upsample / result) + weight pyramid + masks."""
    plane = height * width * 4
    return int(3 * min(channels, channel_group) * plane + plane * (1.0 + 0.4) + height * width)


def _pool(x: torch.Tensor) -> torch.Tensor:
    return F.avg_pool2d(x.unsqueeze(0), 2, stride=2, ceil_mode=True, count_include_pad=False).squeeze(0)


@torch.no_grad()
def pull_push_fill(values: torch.Tensor, valid: torch.Tensor, max_levels: int = 32, channel_group: int = 3,
                   inplace: bool = False) -> torch.Tensor:
    """Fill invalid texels of ``values`` ([C,H,W] float) from valid ones ([H,W] bool).

    Texels where ``valid`` is True are bit-identical to the input.  If no texel is valid the input is returned
    unchanged (nothing to propagate).  With ``inplace=True`` the result is written into ``values``.
    """
    if values.dim() != 3 or valid.shape != values.shape[1:]:
        raise ValueError(f"values must be [C,H,W] and valid [H,W]; got {tuple(values.shape)} / {tuple(valid.shape)}")
    if not bool(valid.any()) or bool(valid.all()):
        return values
    work = values if (inplace and values.dtype == torch.float32) else values.to(torch.float32).clone()
    c_total = work.shape[0]
    w0 = valid.to(torch.float32).unsqueeze(0)

    # weight pyramid (shared by all channels): level l >= 1
    w_levels: List[torch.Tensor] = []
    cur_w = w0
    while len(w_levels) + 1 < max_levels and not (cur_w.shape[1] == 1 and cur_w.shape[2] == 1):
        pooled = _pool(cur_w)
        w_levels.append(pooled)                       # un-normalised average coverage of the 2x2 children
        cur_w = (pooled > 0).to(torch.float32)
    if not w_levels or not bool((w_levels[-1] > 0).any()):
        return values

    for g0 in range(0, c_total, max(1, channel_group)):
        g1 = min(c_total, g0 + max(1, channel_group))
        v = work[g0:g1]                               # view of the finest level (level 0)
        # pull: colours of levels >= 1 (valid-weighted averages)
        colors: List[torch.Tensor] = []
        cur_v = _pool(v * w0)
        for lvl, wl in enumerate(w_levels):
            has = wl > 0
            colors.append(torch.where(has, cur_v / wl.clamp(min=1e-12), torch.zeros_like(cur_v)))
            if lvl + 1 < len(w_levels):
                cur_v = _pool(colors[-1] * has.to(torch.float32))
        # push: coarsest level first
        top = colors[-1]
        top_has = (w_levels[-1] > 0)
        mean_c = top.sum(dim=(1, 2), keepdim=True) / top_has.sum().clamp(min=1).to(torch.float32)
        filled = torch.where(top_has, top, mean_c.expand_as(top))
        for lvl in range(len(w_levels) - 2, -1, -1):
            up = F.interpolate(filled.unsqueeze(0), size=colors[lvl].shape[1:], mode="bilinear", align_corners=False).squeeze(0)
            filled = torch.where(w_levels[lvl] > 0, colors[lvl], up)
            colors[lvl + 1] = None                    # release the coarser level as soon as it is consumed
        up = F.interpolate(filled.unsqueeze(0), size=v.shape[1:], mode="bilinear", align_corners=False).squeeze(0)
        del filled, colors
        work[g0:g1] = torch.where(valid.unsqueeze(0), v, up)
        del up
    return work if work.dtype == values.dtype else work.to(values.dtype)

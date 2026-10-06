"""Sparse voxel sampling in pure PyTorch (bounded memory, no compiled extension).

Semantics of FlexGEMM's trilinear grid sample (MIT):

* query points are in voxel units: voxel ``i`` covers ``[i, i+1)`` and its centre is ``i + 0.5``;
* the 8 neighbours of ``q`` are ``floor(q - 0.5) + {0,1}^3``; a neighbour contributes ``prod(1 - |q - (n + 0.5)|)`` if it is
  inside the grid and active;
* the result is ``sum(w * feat) / max(sum(w), 1e-12)``.

``sample_tent`` additionally returns the weight sum, so "genuinely black" can be told from "no data here".
"""
from __future__ import annotations

import itertools
from typing import Tuple

import torch

__all__ = ["SparseVoxelIndex", "sample_tent"]


class SparseVoxelIndex:
    """Sorted-key lookup table for a set of active voxels (no hashing, no custom kernels)."""

    def __init__(self, coords_xyz: torch.Tensor, grid_shape: Tuple[int, int, int]):
        if coords_xyz.dim() != 2 or coords_xyz.shape[1] != 3:
            raise ValueError(f"coords_xyz must be [N,3], got {tuple(coords_xyz.shape)}")
        w, h, d = (int(v) for v in grid_shape)
        if w * h * d >= 2 ** 62:
            raise ValueError("grid too large for int64 keys")
        c = coords_xyz.to(torch.int64)
        keys = (c[:, 0] * h + c[:, 1]) * d + c[:, 2]
        self.sorted_keys, self.order = torch.sort(keys)
        self.grid_shape = (w, h, d)
        self.n = int(c.shape[0])
        self.device = coords_xyz.device

    @property
    def nbytes(self) -> int:
        return int(self.sorted_keys.numel() * 8 + self.order.numel() * 8)

    def lookup(self, xyz: torch.Tensor) -> torch.Tensor:
        """xyz: [M,3] int64 voxel coordinates -> [M] int64 row index into the feature table (-1 if absent)."""
        w, h, d = self.grid_shape
        inb = (xyz[:, 0] >= 0) & (xyz[:, 0] < w) & (xyz[:, 1] >= 0) & (xyz[:, 1] < h) & (xyz[:, 2] >= 0) & (xyz[:, 2] < d)
        x = xyz[:, 0].clamp(0, w - 1)
        y = xyz[:, 1].clamp(0, h - 1)
        z = xyz[:, 2].clamp(0, d - 1)
        keys = (x * h + y) * d + z
        pos = torch.searchsorted(self.sorted_keys, keys).clamp_(max=max(self.n - 1, 0))
        hit = inb & (self.sorted_keys[pos] == keys)
        return torch.where(hit, self.order[pos], torch.full_like(pos, -1))


@torch.no_grad()
def sample_tent(
    index: SparseVoxelIndex,
    feats: torch.Tensor,
    q: torch.Tensor,
    half_width: int = 1,
    chunk: int = 1 << 20,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample sparse voxel features at continuous points with a separable tent kernel.

    ``half_width=1`` is exactly the trilinear sampling of ``flex_gemm`` (8 neighbours).
    ``half_width=2`` uses a 4x4x4 neighbourhood with a wider tent and is used only as a
    second pass for points whose 8-neighbourhood contains no active voxel.

    Args:
        index: lookup table built from the active voxel coordinates.
        feats: [N, C] features (any float dtype; accumulation is float32).
        q:     [M, 3] float query points in voxel units.
        half_width: kernel half width in voxels (1 = trilinear).
        chunk: number of queries processed at once (bounds peak memory).

    Returns:
        (out [M, C] float32, wsum [M] float32). ``wsum == 0`` means "no active voxel nearby".
    """
    if feats.dim() != 2:
        raise ValueError(f"feats must be [N,C], got {tuple(feats.shape)}")
    if feats.shape[0] != index.n:
        raise ValueError("feats rows must match number of indexed voxels")
    m = q.shape[0]
    c = feats.shape[1]
    out = torch.zeros((m, c), dtype=torch.float32, device=q.device)
    wsum = torch.zeros((m,), dtype=torch.float32, device=q.device)
    if m == 0 or index.n == 0:
        return out, wsum

    hw = int(half_width)
    offs = list(itertools.product(range(1 - hw, hw + 1), repeat=3))
    feats_f = feats if feats.dtype == torch.float32 else feats.float()
    inv_hw = 1.0 / hw

    for s in range(0, m, chunk):
        qc = q[s:s + chunk].float()
        base = torch.floor(qc - 0.5).to(torch.int64)
        acc = torch.zeros((qc.shape[0], c), dtype=torch.float32, device=q.device)
        ws = torch.zeros((qc.shape[0],), dtype=torch.float32, device=q.device)
        for dx, dy, dz in offs:
            nb = base + torch.tensor((dx, dy, dz), dtype=torch.int64, device=q.device)
            idx = index.lookup(nb)
            hit = idx >= 0
            dist = (qc - (nb.to(torch.float32) + 0.5)).abs()
            w = ((1.0 - dist * inv_hw).clamp_(min=0.0)).prod(dim=1)
            w = torch.where(hit, w, torch.zeros_like(w))
            acc.addcmul_(w.unsqueeze(1), feats_f[idx.clamp(min=0)])
            ws += w
        out[s:s + chunk] = acc / ws.clamp(min=1e-12).unsqueeze(1)
        wsum[s:s + chunk] = ws
    return out, wsum

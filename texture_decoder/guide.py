"""Voxel structure for the texture decoder.

The texture decoder does not predict which child voxels exist (``pred_subdiv=False``): every up-sampling step needs to be told.
In a single-node pipeline that information travels inside the encoder's output tensor (its spatial cache).  Between separate nodes
that tensor is gone, so the same information is rebuilt here from the occupied voxels: for every up-sampling level, a ``[rows, 8]``
flag tensor that says which of the 8 children of each row exist, with rows ordered exactly as the decoder will produce them.
"""
from typing import List, Tuple

import torch

from ..modules import sparse as sp

_OFFSETS = [(s % 2, (s // 2) % 2, (s // 4) % 2) for s in range(8)]       # child s sits at (s % 2, s // 2 % 2, s // 4 % 2)


def _keys(xyz: torch.Tensor, base: int) -> torch.Tensor:
    return (xyz[:, 0] * base + xyz[:, 1]) * base + xyz[:, 2]


@torch.no_grad()
def build_guide_subs(coords: torch.Tensor, voxels: torch.Tensor, levels: int) -> Tuple[List[sp.SparseTensor], torch.Tensor]:
    """``coords``: latent coordinates [M,4] (batch, x, y, z) at 1/2**levels resolution; ``voxels``: occupied full-resolution
    voxels [N,3].  Returns the per-level guide tensors (coarse -> fine) and the coordinates the decoder will end up with."""
    dev = coords.device
    vox = voxels.to(dev).long()
    cur = coords.long()
    offsets = torch.tensor(_OFFSETS, device=dev, dtype=torch.long)
    subs = []
    for level in range(levels):
        occupied = vox // (2 ** (levels - 1 - level))                      # occupancy of the finer level
        base = max(int(occupied.max()), int(cur[:, 1:].max()) * 2 + 1) + 2
        occupied_keys = torch.unique(_keys(occupied, base))
        flags = torch.empty((cur.shape[0], 8), dtype=torch.bool, device=dev)
        parent = cur[:, 1:] * 2
        for s in range(8):
            k = _keys(parent + offsets[s], base)
            pos = torch.searchsorted(occupied_keys, k).clamp_(max=occupied_keys.numel() - 1)
            flags[:, s] = occupied_keys[pos] == k
        subs.append(sp.SparseTensor(feats=flags, coords=cur.to(torch.int32)))
        parent_row, child = flags.nonzero(as_tuple=True)                  # row-major: parent ascending, child ascending
        cur = torch.cat([cur[parent_row, :1], cur[parent_row, 1:] * 2 + offsets[child]], dim=1)
    return subs, cur

"""Data passed between STM nodes.  Everything lives on the CPU: ComfyUI caches node outputs, so nothing may pin VRAM."""
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch


@dataclass
class NormTransform:
    """``v_trellis = swap_axes((v - center) * scale)``: unit-cube normalisation + (x, y, z) -> (x, -z, y)."""
    center: np.ndarray            # float64 [3]
    scale: float


@dataclass
class Geometry:
    """The mesh in TRELLIS space (normalised, Z-up). Shared by reference between stages; the voxel resolution can be
    changed later (resolution fallback) without going back to the mesh node."""
    vertices: np.ndarray          # float32 [V, 3]
    faces: np.ndarray             # int64 [F, 3]
    transform: NormTransform


@dataclass
class DualGrid:
    coords: torch.Tensor          # int32 [N, 3]
    dual: torch.Tensor            # float32 [N, 3]  (relative to the grid origin)
    intersected: torch.Tensor     # bool [N, 3]
    resolution: int
    geometry: Geometry


@dataclass
class SparseLatent:
    coords: torch.Tensor          # int32 [M, 4]  (batch, x, y, z)
    feats: torch.Tensor           # float32 [M, C]
    resolution: int
    geometry: Geometry
    kind: str                     # "shape" | "texture"
    voxels: torch.Tensor          # int32 [N, 3]: occupied voxels at full resolution = the structure the decoder reproduces


@dataclass
class PbrVoxels:
    coords: torch.Tensor          # int32 [N, 3]
    feats: torch.Tensor           # float32 [N, 6]: base colour RGB, metallic, roughness, alpha
    resolution: int
    transform: Optional[NormTransform]   # None for voxels from ComfyUI's native VAE decode: the mesh bbox is used
    frame: str = "z_up"                  # "z_up": TRELLIS frame (x, -z, y); "y_up": ComfyUI native frame = normalised glTF frame


@dataclass
class DinoFeatures:
    tokens: torch.Tensor          # float32 [1, T, D]
    image_size: int
    images: List[np.ndarray] = field(default_factory=list)   # uint8 [H, W, 3]; kept so the resolution fallback can re-extract

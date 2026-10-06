"""Texture projection stage: rasterise UVs, sample the 3-D PBR field at every texel (islands and gutter), fill what is left."""
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import torch

from ..core.mesh import MeshData, normalize_uvs_to_unit
from ..core.normalize import compute_transform, to_trellis_space
from ..core.types import PbrVoxels
from ..runtime.device import make_room, pick_device
from ..runtime.stage import log, stage_scope
from ..runtime.vram import current_plan
from .bake import auto_gutter, bake_pbr_textures, sample_at_points


def pbr_from_native_voxel(voxel) -> PbrVoxels:
    """ComfyUI's native ``VOXEL`` (from 'Trellis2 VAE Decode Texture': coords [N,4] in the Y-up frame, colours [N,6]) -> :class:`PbrVoxels`."""
    data, colors, res = voxel.data, voxel.voxel_colors, voxel.resolution
    if colors is None:
        raise ValueError("The VOXEL carries no colours: connect the output of 'Trellis2 VAE Decode Texture'.")
    data, colors = data.detach().cpu(), colors.detach().cpu().float()
    if data.ndim == 2 and data.shape[1] == 4:
        first = data[:, 0] == 0                                  # batch item 0
        if not bool(first.any()):
            raise ValueError("The VOXEL holds no voxels for batch item 0.")
        data, colors = data[first][:, 1:], colors[first]
    coords = data.to(torch.int32)
    if colors.shape[1] == 3:                                     # colour-only voxels: matte, non-metallic, opaque
        colors = torch.cat([colors, torch.zeros(len(colors), 1), torch.ones(len(colors), 2)], dim=1)
    if colors.shape[1] < 6 or coords.shape[0] != colors.shape[0] or coords.shape[0] == 0:
        raise ValueError(f"Unsupported VOXEL: {coords.shape[0]} coordinates, colour width {colors.shape[1]} (expected 6: colour, metallic, roughness, alpha).")
    res = int(res) if res else int(coords.max()) + 1
    if int(coords.max()) >= res or int(coords.min()) < 0:
        raise ValueError(f"VOXEL coordinates exceed the stated resolution {res}.")
    return PbrVoxels(coords, colors[:, :6].contiguous(), res, None, "y_up")


@dataclass
class ProjectionResult:
    base_color_rgba: np.ndarray               # uint8 [H, W, 4]
    metallic_roughness: np.ndarray            # uint8 [H, W, 3]  (R = 0, G = roughness, B = metallic)
    coverage: np.ndarray                      # bool  [H, W]
    stats: Dict[str, float] = field(default_factory=dict)

    @property
    def base_color(self) -> np.ndarray:
        return self.base_color_rgba[..., :3]

    @property
    def alpha(self) -> np.ndarray:
        return self.base_color_rgba[..., 3]

    @property
    def metallic(self) -> np.ndarray:
        return self.metallic_roughness[..., 2]

    @property
    def roughness(self) -> np.ndarray:
        return self.metallic_roughness[..., 1]


def project_texture(mesh: MeshData, pbr: PbrVoxels, texture_size: int = 2048, gutter_px: int = 0, fill: str = "pull_push",
                    device_pref: Optional[str] = None, verbose: bool = False) -> ProjectionResult:
    """Bake ``pbr`` onto ``mesh``'s UV layout. The requested ``texture_size`` is never reduced: on out-of-memory the chunk sizes
    shrink, then the bake continues on the CPU, then a clear memory error is raised."""
    if mesh.uvs is None:
        raise ValueError("The mesh has no UVs: connect the 'STM — UV Unwrap' node (or a mesh that already has UVs).")
    if not 64 <= int(texture_size) <= 16384:
        raise ValueError(f"texture_size must be between 64 and 16384, got {texture_size}")
    uvs = normalize_uvs_to_unit(mesh.uvs)
    if not np.array_equal(uvs, mesh.uvs):
        log("UVs spilled outside [0,1]: fitted uniformly into the unit square (the Apply node applies the same fit).")
    device = pick_device(device_pref)
    plan = current_plan(device, texture_size, pbr.resolution, n_faces=len(mesh.faces))
    make_room(int(texture_size) ** 2 * 100, device)
    transform = pbr.transform if pbr.transform is not None else compute_transform(mesh.vertices)
    vertices = torch.from_numpy(to_trellis_space(mesh.vertices, transform, pbr.frame))
    with stage_scope("texture projection", device):
        res = bake_pbr_textures(vertices, torch.from_numpy(np.ascontiguousarray(mesh.faces, dtype=np.int64)), torch.from_numpy(uvs),
                                pbr.coords, pbr.feats, pbr.resolution, int(texture_size), plan, device,
                                gutter=int(gutter_px) if gutter_px > 0 else auto_gutter(int(texture_size)),
                                fill="none" if fill == "none" else "pull_push", progress=log if verbose else None)
    if res.stats.get("uv_overlap_ratio", 0) > 0.05:
        log(f"WARNING: about {res.stats['uv_overlap_ratio']:.0%} of the UV area overlaps other islands (mirrored/stacked UVs); "
            "only one of the overlapping surfaces receives the texture.")
    return ProjectionResult(res.base_color_rgba, res.metallic_roughness, res.coverage, res.stats)


def sample_vertex_colors(mesh: MeshData, pbr: PbrVoxels, opaque: bool, device_pref: Optional[str] = None) -> np.ndarray:
    """RGBA uint8 per vertex sampled from the voxel field (vertex-colour mode)."""
    device = pick_device(device_pref)
    plan = current_plan(device, 512, pbr.resolution)
    transform = pbr.transform if pbr.transform is not None else compute_transform(mesh.vertices)
    pts = torch.from_numpy(to_trellis_space(mesh.vertices, transform, pbr.frame)).to(device)
    out = sample_at_points(pbr.coords.to(device), pbr.feats.to(device), pbr.resolution, pts, plan.sample_chunk).cpu()
    rgb = (out[:, 0:3].clamp(0, 1) * 255 + 0.5).floor().to(torch.uint8)
    alpha = torch.full((rgb.shape[0], 1), 255, dtype=torch.uint8) if opaque else (out[:, 5:6].clamp(0, 1) * 255 + 0.5).floor().to(torch.uint8)
    return torch.cat([rgb, alpha], dim=1).numpy()

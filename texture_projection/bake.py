"""Texture projection: bake a sparse PBR voxel field onto a UV layout.

    UV triangles --rasterize_uv_coverage--> triangle id per texel                      [H,W] int32
          |   nearest_valid_texel: nearest covered texel within the gutter radius
          v
    per chunk of texels: barycentric 3-D position (islands *and* gutter; the gutter extrapolates the nearest triangle's plane)
          v
    sample_tent: trilinear sampling of the voxel field (2-voxel tent for texels without an active neighbour)
          v
    pull_push_fill for everything that still has no data
          v
    uint8 base colour (RGBA) + metallic / roughness

The gutter therefore holds real samples of the surface continuation instead of smeared island interiors, texels without voxel
data are refilled instead of baked black, and every intermediate is O(chunk): memory does not grow with the atlas size.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np
import torch

from ..runtime.device import StmMemoryError, is_oom, soft_empty_cache, sync
from ..runtime.vram import Plan
from .fill import pull_push_fill
from ..core.raster import barycentric, nearest_valid_texel, nearest_valid_texel_cpu, rasterize_uv_coverage
from .sparse_sample import SparseVoxelIndex, sample_tent

__all__ = ["BakeResult", "bake_pbr_textures", "sample_at_points", "auto_gutter"]

# channel layout of the decoded PBR voxels (identical to the original ``pbr_attr_layout``)
LAYOUT = {"base_color": slice(0, 3), "metallic": slice(3, 4), "roughness": slice(4, 5), "alpha": slice(5, 6)}


@dataclass
class BakeResult:
    base_color_rgba: np.ndarray            # uint8 [H, W, 4]
    metallic_roughness: np.ndarray         # uint8 [H, W, 3]  (R=0, G=roughness, B=metallic: glTF convention)
    stats: Dict[str, float] = field(default_factory=dict)
    coverage: Optional[np.ndarray] = None  # bool [H, W]: texels whose centre lies inside a UV triangle


def auto_gutter(texture_size: int) -> int:
    """Gutter width in texels: ~texture_size/256 (8 px at 2K, 16 px at 4K), clamped to [2, 16]."""
    return int(max(2, min(16, round(texture_size / 256))))


def _quantize(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(0.0, 1.0) * 255.0 + 0.5).floor().to(torch.uint8)


@torch.no_grad()
def sample_at_points(voxel_coords_xyz: torch.Tensor, voxel_feats: torch.Tensor, resolution: int,
                     points: torch.Tensor, chunk: int = 1 << 20) -> torch.Tensor:
    """Sample the voxel field at world points in [-0.5, 0.5]^3 (used for *bake on vertices*). Returns [M, C] float32."""
    index = SparseVoxelIndex(voxel_coords_xyz, (resolution,) * 3)
    q = (points.to(torch.float32) + 0.5) * resolution
    out, ws = sample_tent(index, voxel_feats, q, 1, chunk)
    miss = ws <= 0
    if bool(miss.any()):
        out2, ws2 = sample_tent(index, voxel_feats, q[miss], 2, chunk)
        out[miss] = out2
    return out


@torch.no_grad()
def _bake_once(vertices, faces, uvs, voxel_xyz, voxel_feats, resolution, texture_size, gutter, fill, plan: Plan,
               device: torch.device, progress: Optional[Callable[[str], None]]) -> BakeResult:
    log = progress or (lambda s: None)
    H = W = int(texture_size)
    t_all = time.time()
    stats: Dict[str, float] = {}

    verts = vertices.to(device=device, dtype=torch.float32)
    fcs = faces.to(device=device, dtype=torch.int64)
    uvp = uvs.to(device=device, dtype=torch.float32) * torch.tensor([W, H], dtype=torch.float32, device=device)

    # 1. which triangle covers each texel centre --------------------------------------------------
    t0 = time.time()
    tri_map = rasterize_uv_coverage(uvp, fcs, H, W, max_candidates=plan.raster_candidates)
    covered = tri_map >= 0
    stats["covered_texels"] = float(covered.sum())
    coverage = covered.cpu().numpy()
    tri_px = uvp[fcs]                                                  # [F,3,2] texel units
    e1, e2 = tri_px[:, 1] - tri_px[:, 0], tri_px[:, 2] - tri_px[:, 0]
    stats["uv_area_texels"] = float((0.5 * (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]).abs()).sum())
    stats["uv_overlap_ratio"] = max(0.0, 1.0 - stats["covered_texels"] / max(stats["uv_area_texels"], 1.0))
    del tri_px, e1, e2
    if stats["covered_texels"] == 0:
        raise ValueError("No texel is covered by any UV triangle: the UVs are empty, degenerate or outside [0,1]")
    if gutter <= 0:
        src = torch.where(covered, torch.arange(H * W, device=device, dtype=torch.int32).view(H, W), torch.full((H, W), -1, device=device, dtype=torch.int32))
    elif device.type == "cpu":
        src = nearest_valid_texel_cpu(covered, gutter)           # exact, O(N) in C
    else:
        src = nearest_valid_texel(covered, gutter)               # jump flooding: a handful of elementwise kernels on a GPU
    sync(device)
    stats["t_raster"] = time.time() - t0
    log(f"bake: rasterised {int(stats['covered_texels'])} texels in {stats['t_raster']:.2f}s")

    # 2. sample the voxel field, strip by strip ------------------------------------------------------
    t0 = time.time()
    index = SparseVoxelIndex(voxel_xyz.to(device), (resolution,) * 3)
    feats = voxel_feats.to(device)
    c_out = feats.shape[1]
    attrs = torch.zeros((c_out, H, W), dtype=torch.float32, device=device)
    have = torch.zeros((H, W), dtype=torch.bool, device=device)
    src_flat = src.view(-1)
    tri_flat = tri_map.view(-1)
    max_extrap = 2.5 / float(resolution)                  # gutter points further than 2.5 voxels from their island texel are dropped
    n_gutter = n_wide = n_hole = 0
    for r0 in range(0, H, plan.strip_rows):
        r1 = min(H, r0 + plan.strip_rows)
        ys_all, xs_all = torch.nonzero(src[r0:r1] >= 0, as_tuple=True)
        if ys_all.numel() == 0:
            continue
        ys_all = ys_all + r0
        # every intermediate below is O(chunk): memory does not depend on the strip height or on the atlas size
        for c0 in range(0, ys_all.numel(), plan.sample_chunk):
            ys, xs = ys_all[c0:c0 + plan.sample_chunk], xs_all[c0:c0 + plan.sample_chunk]
            flat = ys * W + xs
            sflat = src_flat[flat].to(torch.int64)
            tid = tri_flat[sflat].to(torch.int64)
            centre = torch.stack([xs.to(torch.float32) + 0.5, ys.to(torch.float32) + 0.5], dim=1)
            tri_uv = uvp[fcs[tid]]
            tri_pos = verts[fcs[tid]]
            pos = (barycentric(tri_uv, centre).unsqueeze(2) * tri_pos).sum(dim=1)
            is_gutter = sflat != flat
            usable = torch.ones_like(is_gutter)
            if bool(is_gutter.any()):
                sy, sx = sflat // W, sflat % W
                c_src = torch.stack([sx.to(torch.float32) + 0.5, sy.to(torch.float32) + 0.5], dim=1)
                pos_src = (barycentric(tri_uv, c_src).unsqueeze(2) * tri_pos).sum(dim=1)
                usable = ~(is_gutter & ((pos - pos_src).norm(dim=1) > max_extrap))
                n_gutter += int(is_gutter.sum())
            q = (pos + 0.5) * resolution
            out, ws = sample_tent(index, feats, q, 1, plan.sample_chunk)
            miss = (ws <= 0) & usable
            if bool(miss.any()):
                out2, ws2 = sample_tent(index, feats, q[miss], 2, plan.sample_chunk)
                out[miss] = out2
                ws[miss] = ws2
                n_wide += int((ws2 > 0).sum())
            good = (ws > 0) & usable
            n_hole += int(((~good) & ~is_gutter).sum())
            attrs[:, ys[good], xs[good]] = out[good].t()
            have[ys[good], xs[good]] = True
            del ys, xs, flat, sflat, tid, centre, tri_uv, tri_pos, pos, q, out, ws, good
        del ys_all, xs_all
    sync(device)
    stats.update(t_sample=time.time() - t0, gutter_texels=float(n_gutter), widened_texels=float(n_wide), island_hole_texels=float(n_hole))
    log(f"bake: sampled field in {stats['t_sample']:.2f}s (gutter {n_gutter}, holes {n_hole})")
    del tri_map, src, src_flat, tri_flat, covered, index

    # 3. fill what is still empty --------------------------------------------------------------------
    t0 = time.time()
    if fill != "none":
        attrs = pull_push_fill(attrs, have, inplace=True)
    sync(device)
    stats["t_fill"] = time.time() - t0

    # 4. quantise and download -----------------------------------------------------------------------
    t0 = time.time()
    base = torch.cat([_quantize(attrs[LAYOUT["base_color"]]), _quantize(attrs[LAYOUT["alpha"]])], dim=0)     # [4,H,W]
    rough = _quantize(attrs[LAYOUT["roughness"]])
    metal = _quantize(attrs[LAYOUT["metallic"]])
    mr = torch.cat([torch.zeros_like(rough), rough, metal], dim=0)                                        # [3,H,W]
    del attrs
    base_np = base.permute(1, 2, 0).contiguous().cpu().numpy()
    mr_np = mr.permute(1, 2, 0).contiguous().cpu().numpy()
    stats["t_download"] = time.time() - t0
    stats["t_total"] = time.time() - t_all
    return BakeResult(base_np, mr_np, stats, coverage)


def bake_pbr_textures(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    uvs: torch.Tensor,
    voxel_coords_xyz: torch.Tensor,
    voxel_feats: torch.Tensor,
    resolution: int,
    texture_size: int,
    plan: Plan,
    device: torch.device,
    gutter: Optional[int] = None,
    fill: str = "pull_push",
    progress: Optional[Callable[[str], None]] = None,
    max_retries: int = 4,
) -> BakeResult:
    """Bake PBR textures.  ``uvs`` are in raster convention (row = v * H); ``vertices`` in [-0.5, 0.5]^3.

    On out-of-memory the working set (chunk sizes) is halved and the bake is retried; after ``max_retries``
    attempts the bake falls back to the CPU.  Only if that fails too a :class:`StmMemoryError` with advice is raised.
    """
    gutter = auto_gutter(texture_size) if gutter is None else int(gutter)
    if texture_size < 64:
        raise ValueError(f"texture_size must be >= 64, got {texture_size}")
    if vertices.shape[0] != uvs.shape[0]:
        raise ValueError(f"vertices ({vertices.shape[0]}) and uvs ({uvs.shape[0]}) must have the same length")
    if not (torch.isfinite(uvs).all() and torch.isfinite(vertices).all()):
        raise ValueError("vertices / uvs contain NaN or inf")
    if voxel_feats.shape[0] != voxel_coords_xyz.shape[0] or voxel_feats.shape[0] == 0:
        raise ValueError("decoded voxel field is empty or inconsistent")

    attempts = []
    dev = device if plan.atlas_on_device or device.type == "cpu" else torch.device("cpu")
    last: Optional[BaseException] = None
    for attempt in range(max_retries + 1):
        try:
            return _bake_once(vertices, faces, uvs, voxel_coords_xyz, voxel_feats, resolution, texture_size, gutter, fill, plan, dev, progress)
        except BaseException as e:                                         # noqa: BLE001
            if not is_oom(e):
                raise
            last = e
            attempts.append(f"{dev.type}: sample_chunk={plan.sample_chunk} strip_rows={plan.strip_rows}")
            soft_empty_cache()
            if attempt < max_retries - 1 or dev.type == "cpu":
                plan.shrink()
            elif dev.type != "cpu":
                if progress:
                    progress("bake: out of GPU memory, continuing on the CPU")
                dev = torch.device("cpu")
                plan.shrink()
    raise StmMemoryError(
        f"Texture baking at {texture_size}x{texture_size} ran out of memory after {len(attempts)} attempts ({'; '.join(attempts)}). "
        f"Lower texture_size (e.g. 2048) or close other applications.") from last

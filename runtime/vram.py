"""VRAM-aware planning.

``make_plan`` converts *measured* free/total device memory (not a hard-coded GPU model) into chunk
sizes and policies.  The constants are conservative estimates of bytes per work item derived from
the tensor shapes in ``baking/`` (documented next to each formula); they are *estimates*, validated
by the OOM-retry logic in ``baking/bake.py`` rather than assumed to be exact.

Tiers (by total memory): ``<=6.5 GiB`` -> "low", ``<=10.5`` -> "8GB", ``<=14.5`` -> "12GB",
``<=20`` -> "16GB", else "24GB+".  Free memory is what actually limits the chunk sizes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .device import GiB, MemInfo, device_memory, system_ram

_TIERS = (("low", 6.5), ("8GB", 10.5), ("12GB", 14.5), ("16GB", 20.0), ("24GB+", 1e9))

# bytes of working memory per item (see baking/bake.py):
BYTES_PER_QUERY = 340          # per texel of one chunk in bake._bake_once: ids, bary, 3 corners x (uv, xyz), pos, q, sampler temporaries (measured ~330 B)
BYTES_PER_STRIP_TEXEL = 48     # per covered texel of a strip: the two int64 index arrays + nonzero() temporary
BYTES_PER_CANDIDATE = 64       # rasterizer candidate texel temporaries
BYTES_PER_ATLAS_TEXEL = 4 + 4 + 1 + 7 * 4 * 2.6 + 4   # tri_map + src_map + valid + (6 attrs + weight, pyramid x2.6) + jfa temp

# Tier caps: even with plenty of free memory, small tiers keep conservative chunks (fragmentation and other
# ComfyUI allocations make "free" an optimistic number), larger tiers are allowed bigger, faster chunks.
_TIER_SCALE = {"low": 1, "8GB": 2, "12GB": 4, "16GB": 8, "24GB+": 16, "cpu": 2}
_SAMPLE_CHUNK_BASE = 1 << 18          # queries
_RASTER_CAND_BASE = 1 << 18           # candidate texels
_DUAL_ROWS_BASE = 1 << 18
_DUAL_COLS_BASE = 1 << 19
_DUAL_CAND_BASE = 1 << 19


@dataclass
class Plan:
    tier: str
    device_kind: str
    total_gb: float
    free_gb: float
    sample_chunk: int
    strip_rows: int
    raster_candidates: int
    atlas_on_device: bool
    dual_on_device: bool
    dual_max_rows: int
    dual_max_cols: int
    dual_max_candidates: int
    conv_chunk_mb: int
    sampler_chunk_size: int
    keep_models_loaded: bool
    dino_dtype: str
    notes: List[str] = field(default_factory=list)

    def shrink(self) -> "Plan":
        """A plan with every working-set knob halved (used by the OOM retry loop)."""
        self.sample_chunk = max(1 << 15, self.sample_chunk // 2)
        self.strip_rows = max(16, self.strip_rows // 2)
        self.raster_candidates = max(1 << 16, self.raster_candidates // 2)
        self.dual_max_rows = max(1 << 16, self.dual_max_rows // 2)
        self.dual_max_cols = max(1 << 16, self.dual_max_cols // 2)
        self.dual_max_candidates = max(1 << 16, self.dual_max_candidates // 2)
        self.conv_chunk_mb = max(16, self.conv_chunk_mb // 2)
        self.sampler_chunk_size = max(4096, self.sampler_chunk_size // 2)
        return self


def tier_of(total_gb: float) -> str:
    for name, limit in _TIERS:
        if total_gb <= limit:
            return name
    return "24GB+"


def clamp(x, lo, hi):
    return int(max(lo, min(hi, x)))


def make_plan(mem: MemInfo, texture_size: int, resolution: int, ram_available: int, ram_total: int,
              keep_models_loaded=None, n_faces: int = 0) -> Plan:
    on_gpu = mem.kind in ("cuda", "xpu")
    tier = tier_of(mem.total_gb) if on_gpu else "cpu"
    notes: List[str] = []
    # Working budget of the baking stage: the diffusion / VAE models are already offloaded at that point.
    budget = mem.free * (0.55 if on_gpu else 0.35)
    if not on_gpu:
        budget = min(budget, ram_available * 0.35)

    scale = _TIER_SCALE[tier]
    sample_chunk = clamp(budget * 0.25 / BYTES_PER_QUERY, 1 << 16, _SAMPLE_CHUNK_BASE * scale)
    strip_rows = clamp(budget * 0.20 / (texture_size * BYTES_PER_STRIP_TEXEL), 16, texture_size)
    raster_cand = clamp(budget * 0.15 / BYTES_PER_CANDIDATE, 1 << 16, _RASTER_CAND_BASE * scale)

    atlas_bytes = texture_size * texture_size * BYTES_PER_ATLAS_TEXEL
    atlas_on_device = on_gpu and atlas_bytes <= 0.7 * budget
    if on_gpu and not atlas_on_device:
        notes.append(f"atlas working set ~{atlas_bytes / GiB:.1f} GiB exceeds the {budget / GiB:.1f} GiB budget -> baking runs on the CPU")
    host_need = atlas_bytes * (1.0 if not atlas_on_device else 0.3)
    if host_need > 0.8 * ram_available:
        notes.append(f"WARNING: {texture_size}px bake needs ~{host_need / GiB:.1f} GiB RAM but only {ram_available / GiB:.1f} GiB are free")

    # dual grid (mesh -> sparse voxels): ~5 * res^2 voxels, ~100 B each plus chunk buffers
    n_vox = 5 * resolution * resolution
    dual_bytes = n_vox * 100 + 1.5 * GiB
    dual_on_device = on_gpu and dual_bytes <= 0.8 * mem.free
    if not dual_on_device and n_vox * 100 > 0.6 * ram_available:
        notes.append(f"WARNING: resolution {resolution} needs ~{n_vox * 100 / GiB:.1f} GiB RAM for the voxelisation; lower the resolution")
    dual_cap = mem.free if dual_on_device else ram_available
    dual_rows = clamp(dual_cap * 0.05 / 64, 1 << 16, _DUAL_ROWS_BASE * scale)
    dual_cols = clamp(dual_cap * 0.08 / 96, 1 << 16, _DUAL_COLS_BASE * scale)
    dual_cand = clamp(dual_cap * 0.08 / 160, 1 << 16, _DUAL_CAND_BASE * scale)

    conv_mb = clamp((mem.free if on_gpu else ram_available) * 0.06 / (1 << 20), 64, 1024)
    sampler_chunk = clamp((mem.free if on_gpu else ram_available) * 0.5 / 20000, 16384, 262144)

    if keep_models_loaded is None:
        keep = ram_total >= 24 * GiB
    else:
        keep = bool(keep_models_loaded)
    if keep and ram_available < 10 * GiB:
        keep = False
        notes.append("keep_models_loaded disabled: less than 10 GiB RAM available")

    return Plan(tier=tier, device_kind=mem.kind, total_gb=mem.total_gb, free_gb=mem.free_gb,
                sample_chunk=sample_chunk, strip_rows=strip_rows, raster_candidates=raster_cand,
                atlas_on_device=atlas_on_device if on_gpu else False, dual_on_device=dual_on_device,
                dual_max_rows=dual_rows, dual_max_cols=dual_cols, dual_max_candidates=dual_cand,
                conv_chunk_mb=conv_mb, sampler_chunk_size=sampler_chunk, keep_models_loaded=keep,
                dino_dtype="bfloat16" if tier == "low" else "float32", notes=notes)


# Peak VRAM of each stage measured on an 8 GB RTX 5050 laptop at resolution 1024 (5.8M-face mesh); activations grow with the
# voxel count, i.e. roughly with resolution squared.
_MEASURED_PEAK_GIB_1024 = {"dual grid": 3.0, "shape encoder": 4.7, "texture generation": 3.8, "texture decoder": 5.1}


def stage_reserve_bytes(stage: str, resolution: int, total_bytes: int) -> int:
    """VRAM to ask ComfyUI to free before ``stage`` runs (never more than 92 % of the device)."""
    est = _MEASURED_PEAK_GIB_1024.get(stage, 2.0) * GiB * (resolution / 1024.0) ** 2
    return int(min(est, 0.92 * total_bytes))


def current_plan(device, texture_size: int = 2048, resolution: int = 1024, keep_models_loaded=None, n_faces: int = 0) -> Plan:
    """Plan from the memory that is free *right now* on ``device``."""
    ram_available, ram_total = system_ram()
    return make_plan(device_memory(device), int(texture_size), int(resolution), ram_available, ram_total, keep_models_loaded, n_faces)

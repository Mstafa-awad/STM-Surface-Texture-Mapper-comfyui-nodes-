"""Shape encoding for ComfyUI's native Trellis2 pipeline: mesh -> native ``LATENT`` + ``SHAPE_SUBDIVIDES``.

Replaces a hand-written encoder node that needed ``o_voxel``.  Differences that matter for memory:
* the subdivision guides are rebuilt from the mesh voxels instead of decoding the whole shape (a full mesh extraction at 1536);
* the encoder is released after use and every output lives on the intermediate (CPU) device, so nothing pins VRAM between runs.
"""
import dataclasses
import os
from typing import List, Optional

import torch

from ..core.mesh import comfy_mesh_from_data, mesh_data_from_comfy
from ..core.normalize import compute_transform, to_trellis_space
from ..core.types import SparseLatent
from ..geometry.stage import build_geometry, voxelize
from ..load_models.handle import StmModels
from ..runtime.stage import log
from ..texture_decoder.guide import build_guide_subs
from .stage import encode_shape

LATENT_LEVELS = 4                                           # f16 shape latent: 2**4 = 16x down-sampling


def resolve_encoder_path(value: str, models_dir: str) -> str:
    """Path (without extension) of the shape-encoder checkpoint; accepts a full path or a name below ComfyUI/models."""
    value = (value or "").strip().strip('"')
    names = [value] if value else []
    names.append("shape_enc_next_dc_f16c32_fp16.safetensors")
    roots = [""] if os.path.isabs(value) else []
    roots += [os.path.join(models_dir, "Trellis2", "encoders"), os.path.join(models_dir, "Trellis2"), models_dir, os.path.join(models_dir, "trellis2")]
    tried: List[str] = []
    for name in names:
        for root in roots:
            p = os.path.join(root, name) if root else name
            base = p[:-len(".safetensors")] if p.lower().endswith(".safetensors") else p
            tried.append(base + ".safetensors")
            if os.path.isfile(base + ".safetensors") and os.path.isfile(base + ".json"):
                return base
    raise FileNotFoundError("Shape encoder not found (needs the .safetensors and the .json next to it). Looked for:\n  " + "\n  ".join(tried[:8]))


def make_native_latent(latent: SparseLatent) -> dict:
    from comfy.latent_formats import Trellis2ShapeSLAT
    feats = Trellis2ShapeSLAT().process_in(latent.feats)
    if feats.ndim != 2 or feats.shape[-1] != 32:
        raise ValueError(f"Unexpected shape latent {tuple(feats.shape)}; expected [N, 32].")
    coords = latent.coords.to(torch.int32).cpu()
    return {"samples": feats.unsqueeze(0).permute(0, 2, 1).unsqueeze(-1).contiguous().cpu(), "coords": coords,
            "coord_counts": torch.tensor([coords.shape[0]], dtype=torch.int64), "coord_resolution": int(latent.resolution) // 16,
            "type": "trellis2", "model_frame": "z_up"}


def make_native_subdivides(latent: SparseLatent, device: torch.device) -> list:
    from comfy.ldm.trellis2.vae import SparseTensor as NativeSparse
    subs, _ = build_guide_subs(latent.coords.to(device), latent.voxels, LATENT_LEVELS)
    return [NativeSparse(feats=s.feats.cpu(), coords=s.coords.cpu()) for s in subs]


def encode_for_native(mesh, encoder_file: str, resolution: int, normalize_output: bool, models_dir: str):
    from comfy import model_management as mm
    data = mesh_data_from_comfy(mesh)
    geometry = build_geometry(data)
    grid = voxelize(geometry, int(resolution))
    models = StmModels.standalone({"shape_slat_encoder": resolve_encoder_path(encoder_file, models_dir)})
    latent = encode_shape(models, grid)
    del grid
    out = make_native_latent(latent), make_native_subdivides(latent, mm.get_torch_device())
    log(f"native shape latent: {latent.coords.shape[0]:,} tokens at {resolution}^3")
    if normalize_output:                                     # same mesh the old encoder node returned: unit cube, Y-up
        t = compute_transform(data.vertices)
        data = dataclasses.replace(data, vertices=to_trellis_space(data.vertices, t, "y_up"))
    return out[0], out[1], comfy_mesh_from_data(data)

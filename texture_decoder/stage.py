"""Texture decoder stage: texture latent -> sparse PBR voxel field (base colour, metallic, roughness, alpha)."""
import torch

from ..core.types import PbrVoxels, SparseLatent
from ..models.chunking import configure_vae, run_chunked
from ..modules import sparse as sp
from ..runtime.stage import stage_scope
from ..runtime.device import device_memory
from ..runtime.vram import current_plan, stage_reserve_bytes
from .guide import build_guide_subs


@torch.no_grad()
def decode_texture(models, latent: SparseLatent) -> PbrVoxels:
    device = models.device
    plan = current_plan(device, resolution=latent.resolution)
    with stage_scope("texture decoder", device, reserve=stage_reserve_bytes("texture decoder", latent.resolution, device_memory(device).total)):
        slat = sp.SparseTensor(feats=latent.feats.to(device), coords=latent.coords.to(device))
        decoder = models.acquire_network("tex_slat_decoder", plan)
        try:
            configure_vae(decoder, plan)
            guide = None
            if not decoder.pred_subdiv:                              # the real texture decoder: structure is given, not predicted
                guide, _ = build_guide_subs(slat.coords, latent.voxels, len(decoder.blocks) - 1)
            pbr = run_chunked(lambda: decoder(slat, guide_subs=guide) * 0.5 + 0.5, decoder, plan, "texture decoder")
        finally:
            models.release("tex_slat_decoder", decoder, plan)
        return PbrVoxels(pbr.coords[:, 1:].to(torch.int32).cpu(), pbr.feats.float().cpu(), latent.resolution, latent.geometry.transform)

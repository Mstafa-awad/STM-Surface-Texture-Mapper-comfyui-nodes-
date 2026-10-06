"""Shape encoder stage: dual grid -> shape latent (SC-VAE encoder)."""
import torch
import torch.nn.functional as F

from ..core.types import DualGrid, SparseLatent
from ..models.chunking import configure_vae, run_chunked
from ..models.sc_vaes.fdg_vae import FlexiDualGridVaeEncoder
from ..models.sc_vaes.sparse_unet_vae import chunked_apply
from ..modules import sparse as sp
from ..runtime.stage import stage_scope
from ..runtime.device import device_memory
from ..runtime.vram import current_plan, stage_reserve_bytes


@torch.no_grad()
def lean_encode(encoder, grid: DualGrid, device: torch.device, chunk: int) -> sp.SparseTensor:
    """Exactly ``FlexiDualGridVaeEncoder.forward(vertices, intersected)`` (posterior mean), but the 6-channel float32 input and the
    float32 output of the input layer are produced chunk by chunk straight into the half-precision hidden tensor, from CPU data:
    the full-resolution input never exists on the GPU and only one full-size hidden tensor is allocated at level 0."""
    n = grid.coords.shape[0]
    lin = encoder.input_layer
    hidden = torch.empty((n, lin.out_features), dtype=encoder.dtype, device=device)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        dual = grid.dual[s:e] * grid.resolution - grid.coords[s:e]
        x = torch.cat([dual - 0.5, grid.intersected[s:e].float() - 0.5], dim=1).to(device)
        hidden[s:e] = F.linear(x, lin.weight, lin.bias).to(encoder.dtype)
    coords = torch.cat([torch.zeros_like(grid.coords[:, :1]), grid.coords], dim=-1).to(device)
    h = sp.SparseTensor(feats=hidden, coords=coords)
    del hidden, coords
    for res in encoder.blocks:
        for block in res:
            h = block(h)

    def finalize(t):
        t = F.layer_norm(t.to(torch.float32), (t.shape[-1],))
        return F.linear(t, encoder.to_latent.weight, encoder.to_latent.bias)

    out = chunked_apply(finalize, h.feats, chunk)
    return h.replace(out[:, : out.shape[1] // 2].contiguous())           # posterior mean


def _plain_encode(encoder, grid: DualGrid, device: torch.device) -> sp.SparseTensor:
    feats = grid.dual * grid.resolution - grid.coords
    coords = torch.cat([torch.zeros_like(grid.coords[:, :1]), grid.coords], dim=-1)
    verts = sp.SparseTensor(feats=feats.to(device), coords=coords.to(device))
    return encoder(verts, verts.replace(grid.intersected.to(device)))


@torch.no_grad()
def encode_shape(models, grid: DualGrid) -> SparseLatent:
    device = models.device
    plan = current_plan(device, resolution=grid.resolution)
    with stage_scope("shape encoder", device, reserve=stage_reserve_bytes("shape encoder", grid.resolution, device_memory(device).total)):
        encoder = models.acquire_network("shape_slat_encoder", plan)
        try:
            configure_vae(encoder, plan)
            if isinstance(encoder, FlexiDualGridVaeEncoder):     # TRELLIS.2 shape encoder: memory-lean entry, identical result
                run = lambda: lean_encode(encoder, grid, device, max(4096, plan.sampler_chunk_size))
            else:
                run = lambda: _plain_encode(encoder, grid, device)
            latent = run_chunked(run, encoder, plan, "shape encoder")
        finally:
            models.release("shape_slat_encoder", encoder, plan)
        return SparseLatent(latent.coords.cpu(), latent.feats.float().cpu(), grid.resolution, grid.geometry, "shape", grid.coords)

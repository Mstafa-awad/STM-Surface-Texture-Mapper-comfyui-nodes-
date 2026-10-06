"""Texture decoding with ComfyUI's native Trellis2 texture VAE (the file loaded by 'Load VAE'), so the TRELLIS.2 decoder checkpoint
is not needed.  The structure guidance (``guide_subs``) is rebuilt from the mesh voxels exactly as in :mod:`.stage`."""
import torch

from ..core.types import PbrVoxels, SparseLatent
from ..runtime.device import StmMemoryError, is_oom
from ..runtime.stage import stage_scope
from .guide import build_guide_subs


def _decode_memory(point_count: int, dtype) -> int:
    """Same estimate ComfyUI's own decode node uses (last 128-channel stage + 27 neighbour indices + workspace)."""
    try:
        from comfy_extras.nodes_trellis2 import _sparse_vae_decode_memory
        return int(_sparse_vae_decode_memory(point_count, dtype))
    except Exception:
        from comfy import model_management as mm
        return 2 * 1024 ** 3 + int(point_count) * (896 * mm.dtype_size(dtype) + 27 * 4)


@torch.no_grad()
def decode_texture_native(vae, latent: SparseLatent) -> PbrVoxels:
    from comfy import model_management as mm
    from comfy.ldm.trellis2.vae import SparseTensor as NativeSparse
    device = mm.get_torch_device()
    texture_vae = vae.first_stage_model
    levels = len(texture_vae.txt_dec.blocks) - 1
    with stage_scope("texture decoder (VAE)", device):
        subs, final = build_guide_subs(latent.coords.to(device), latent.voxels, levels)
        subs = [NativeSparse(feats=s.feats, coords=s.coords) for s in subs]
        shape = torch.Size([1, latent.feats.shape[1], latent.feats.shape[0], 1])
        memory = _decode_memory(final.shape[0], vae.vae_dtype)
        try:
            vae.prepare_decode(shape, memory_required=memory)
        except TypeError:                                    # older ComfyUI: no memory_required argument
            vae.prepare_decode(shape)
        slat = NativeSparse(feats=latent.feats.to(device), coords=latent.coords.to(device)).to(vae.vae_dtype)
        try:
            voxel = texture_vae.decode_tex_slat(slat, subs)
        except BaseException as e:                           # noqa: BLE001
            if is_oom(e):
                raise StmMemoryError("The texture VAE ran out of memory. Lower the resolution or close other GPU applications.") from e
            raise
        return PbrVoxels(voxel.coords[:, 1:].to(torch.int32).cpu(), voxel.feats.float().cpu(), latent.resolution, latent.geometry.transform)

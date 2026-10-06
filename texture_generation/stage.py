"""Texture generation stage: shape latent + image tokens --flow-matching DiT--> texture latent."""
import torch
from PIL import Image

from ..core.types import DinoFeatures, SparseLatent
from ..modules import sparse as sp
from ..runtime.device import StmMemoryError, is_oom
from ..runtime.stage import log, stage_scope
from ..runtime.device import device_memory
from ..runtime.vram import current_plan, stage_reserve_bytes
from . import samplers

_PREFIX = {"euler": "Euler", "heun": "Heun", "rk4": "RK4", "rk5": "RK5"}


def _flow_model_name(resolution: int) -> str:
    return "tex_slat_flow_model_512" if resolution == 512 else "tex_slat_flow_model_1024"


@torch.no_grad()
def _sample(models, shape: SparseLatent, dino: DinoFeatures, seed, params, sampler_kind, dino_lock, verbose) -> SparseLatent:
    device = models.device
    plan = current_plan(device, resolution=shape.resolution)
    name = _flow_model_name(shape.resolution)
    spec = models.sampler_spec()
    sampler = getattr(samplers, f"Flow{_PREFIX.get(sampler_kind, 'Euler')}GuidanceIntervalSampler")(**spec["args"])
    sampler_params = {**spec["params"], **params}
    with stage_scope("texture generation", device, reserve=stage_reserve_bytes("texture generation", shape.resolution, device_memory(device).total)):
        flow = models.acquire_network(name, plan)
        try:
            s_mean, s_std = models.normalization("shape_slat_normalization", device)
            t_mean, t_std = models.normalization("tex_slat_normalization", device)
            shape_norm = (sp.SparseTensor(feats=shape.feats.to(device), coords=shape.coords.to(device)) - s_mean) / s_std
            # Private generator: the noise depends only on `seed` (the global RNG is also consumed by model construction).
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(seed))
            noise = shape_norm.replace(feats=torch.randn(shape_norm.coords.shape[0], flow.in_channels - shape_norm.feats.shape[1], generator=gen).to(device))
            cond = {"cond": dino.tokens.to(device), "neg_cond": torch.zeros_like(dino.tokens).to(device)}
            try:
                out = sampler.sample(flow, noise, concat_cond=shape_norm, **cond, **sampler_params, verbose=verbose,
                                     dino_lock=dino_lock[0], dino_substeps=dino_lock[1], dino_foundation_cap=dino_lock[2],
                                     tqdm_desc="Sampling texture latent").samples
            except BaseException as e:                       # noqa: BLE001
                if is_oom(e):
                    raise StmMemoryError(f"The texture flow model ran out of memory at resolution {shape.resolution} "
                                         f"({shape.coords.shape[0]} tokens). Use a lower resolution, the fp8 model files, or enable oom_fallback.") from e
                raise
            latent = out * t_std + t_mean
        finally:
            models.release(name, flow, plan)
        return SparseLatent(latent.coords.cpu(), latent.feats.float().cpu(), shape.resolution, shape.geometry, "texture", shape.voxels)


def generate_texture_latent(models, shape: SparseLatent, dino: DinoFeatures, *, seed=0, steps=12, guidance_strength=3.0,
                            guidance_rescale=0.2, rescale_t=3.0, guidance_interval=(0.0, 0.9), sampler="euler", dino_lock=0.0,
                            dino_substeps=4, dino_foundation_cap=1.0, verbose=False, oom_fallback=True) -> SparseLatent:
    """Run the flow model. With ``oom_fallback`` an out-of-memory error steps the *model* resolution down (1536 -> 1024 -> 512)
    and re-runs the geometry / encoder / conditioning stages internally for the lower resolution."""
    params = {"steps": int(steps), "guidance_strength": float(guidance_strength), "guidance_rescale": float(guidance_rescale),
              "guidance_interval": list(guidance_interval), "rescale_t": float(rescale_t)}
    try:
        return _sample(models, shape, dino, seed, params, sampler, (dino_lock, dino_substeps, dino_foundation_cap), verbose)
    except StmMemoryError as e:
        if not oom_fallback or shape.resolution <= 512:
            raise
        lower = 1024 if shape.resolution == 1536 else 512
        log(f"WARNING: {e}\n  -> retrying automatically at resolution {lower}")
        from ..dino.stage import extract_dino_features
        from ..geometry.stage import voxelize
        from ..shape_encoder.stage import encode_shape
        shape_lo = encode_shape(models, voxelize(shape.geometry, lower, models.device_pref, verbose))
        size = min(lower, 1024)
        dino_lo = dino if dino.image_size == size else extract_dino_features(models, [Image.fromarray(a) for a in dino.images], size)
        return generate_texture_latent(models, shape_lo, dino_lo, seed=seed, steps=steps, guidance_strength=guidance_strength,
                                       guidance_rescale=guidance_rescale, rescale_t=rescale_t, guidance_interval=guidance_interval,
                                       sampler=sampler, dino_lock=dino_lock, dino_substeps=dino_substeps,
                                       dino_foundation_cap=dino_foundation_cap, verbose=verbose, oom_fallback=oom_fallback)

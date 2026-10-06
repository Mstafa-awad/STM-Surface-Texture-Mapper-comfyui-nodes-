from ..load_models.handle import coerce_models
from .stage import generate_texture_latent


class STMTextureGeneration:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "models": ("TRELLIS2PIPELINE",),
                "shape_latent": ("STM_SHAPE_LATENT",),
                "dino_features": ("STM_DINO_FEATURES",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7fffffff, "control_after_generate": True}),
                "steps": ("INT", {"default": 12, "min": 1, "max": 100}),
                "guidance_strength": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 99.99, "step": 0.01, "tooltip": "Official TRELLIS.2 default 1.0 = no guidance, one model pass per step (about half the time). Higher values add a second pass per step."}),
                "guidance_rescale": ("FLOAT", {"default": 0.00, "min": 0.00, "max": 1.00, "step": 0.01}),
                "rescale_t": ("FLOAT", {"default": 3.00, "min": 0.00, "max": 9.99, "step": 0.01}),
                "guidance_interval_start": ("FLOAT", {"default": 0.60, "min": 0.00, "max": 1.00, "step": 0.01}),
                "guidance_interval_end": ("FLOAT", {"default": 0.90, "min": 0.00, "max": 1.00, "step": 0.01}),
                "sampler": (["euler", "heun", "rk4", "rk5"], {"default": "euler"}),
                "dino_lock": ("FLOAT", {"default": 0.00, "min": 0.00, "max": 1.00, "step": 0.01}),
                "dino_substeps": ("INT", {"default": 4, "min": 1, "max": 99, "step": 1}),
                "dino_foundation_cap": ("FLOAT", {"default": 1.00, "min": 0.01, "max": 1.00, "step": 0.01}),
                "oom_fallback": ("BOOLEAN", {"default": True, "tooltip": "On out-of-memory step the model resolution down (1536 -> 1024 -> 512). Texture size is never reduced."}),
            },
            "optional": {"verbose": ("BOOLEAN", {"default": False})},
        }

    RETURN_TYPES = ("STM_TEXTURE_LATENT",)
    RETURN_NAMES = ("texture_latent",)
    FUNCTION = "run"
    CATEGORY = "STM"

    def run(self, models, shape_latent, dino_features, seed, steps, guidance_strength, guidance_rescale, rescale_t,
            guidance_interval_start, guidance_interval_end, sampler, dino_lock, dino_substeps, dino_foundation_cap, oom_fallback, verbose=False):
        return (generate_texture_latent(
            coerce_models(models), shape_latent, dino_features, seed=seed, steps=steps, guidance_strength=guidance_strength,
            guidance_rescale=guidance_rescale, rescale_t=rescale_t, guidance_interval=(guidance_interval_start, guidance_interval_end),
            sampler=sampler, dino_lock=dino_lock, dino_substeps=dino_substeps, dino_foundation_cap=dino_foundation_cap,
            verbose=bool(verbose), oom_fallback=bool(oom_fallback)),)


NODE_CLASS_MAPPINGS = {"STM_TextureGeneration": STMTextureGeneration}
NODE_DISPLAY_NAME_MAPPINGS = {"STM_TextureGeneration": "STM — Texture Generation"}

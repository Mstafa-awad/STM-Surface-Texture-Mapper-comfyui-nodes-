from ..load_models.handle import coerce_models
from .native import decode_texture_native
from .stage import decode_texture


class STMTextureDecoder:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"models": ("TRELLIS2PIPELINE",), "texture_latent": ("STM_TEXTURE_LATENT",)}}

    RETURN_TYPES = ("STM_PBR_VOXELS", "INT")
    RETURN_NAMES = ("pbr_voxels", "voxel_count")
    FUNCTION = "run"
    CATEGORY = "STM"

    def run(self, models, texture_latent):
        pbr = decode_texture(coerce_models(models), texture_latent)
        return (pbr, int(pbr.coords.shape[0]))


class STMTextureDecoderVAE:
    """Same as 'STM — Texture Decoder' but decodes with the VAE from ComfyUI's 'Load VAE' (native Trellis2 texture VAE)."""

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"vae": ("VAE",), "texture_latent": ("STM_TEXTURE_LATENT",)}}

    RETURN_TYPES = ("STM_PBR_VOXELS", "INT")
    RETURN_NAMES = ("pbr_voxels", "voxel_count")
    FUNCTION = "run"
    CATEGORY = "STM"

    def run(self, vae, texture_latent):
        pbr = decode_texture_native(vae, texture_latent)
        return (pbr, int(pbr.coords.shape[0]))


NODE_CLASS_MAPPINGS = {"STM_TextureDecoder": STMTextureDecoder, "STM_TextureDecoderVAE": STMTextureDecoderVAE}
NODE_DISPLAY_NAME_MAPPINGS = {"STM_TextureDecoder": "STM — Texture Decoder", "STM_TextureDecoderVAE": "STM — Texture Decoder (VAE)"}

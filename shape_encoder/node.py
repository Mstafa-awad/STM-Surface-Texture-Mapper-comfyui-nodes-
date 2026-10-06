from ..load_models.handle import coerce_models

from ..load_models.node import models_dir
from .native import encode_for_native
from .stage import encode_shape


class STMShapeEncoder:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"models": ("TRELLIS2PIPELINE",), "dual_grid": ("STM_DUAL_GRID",)}}

    RETURN_TYPES = ("STM_SHAPE_LATENT",)
    RETURN_NAMES = ("shape_latent",)
    FUNCTION = "run"
    CATEGORY = "STM"

    def run(self, models, dual_grid):
        return (encode_shape(coerce_models(models), dual_grid),)


class STMMeshEncoderNative:
    """MESH -> native Trellis2 LATENT + SHAPE_SUBDIVIDES (for ComfyUI's own sampler / VAE nodes). Needs no VAE, no TRELLIS.2 folder and no o_voxel."""

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "mesh": ("MESH",),
            "encoder_file": ("STRING", {"default": "shape_enc_next_dc_f16c32_fp16.safetensors",
                                        "tooltip": "Shape encoder checkpoint (with its .json): full path, or a name below ComfyUI/models/Trellis2/encoders."}),
            "resolution": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 128}),
            "normalize_output": ("BOOLEAN", {"default": True, "tooltip": "Return the mesh centred and scaled into the unit cube (Y-up), as the native bake expects."}),
        }}

    RETURN_TYPES = ("LATENT", "SHAPE_SUBDIVIDES", "MESH")
    RETURN_NAMES = ("shape_latent", "shape_subdivides", "mesh")
    FUNCTION = "run"
    CATEGORY = "STM"

    def run(self, mesh, encoder_file, resolution, normalize_output):
        return encode_for_native(mesh, encoder_file, int(resolution), bool(normalize_output), models_dir())


NODE_CLASS_MAPPINGS = {"STM_ShapeEncoder": STMShapeEncoder, "STM_MeshEncoderNative": STMMeshEncoderNative}
NODE_DISPLAY_NAME_MAPPINGS = {"STM_ShapeEncoder": "STM — Shape Encoder", "STM_MeshEncoderNative": "STM — Mesh Encoder (Native Latent)"}

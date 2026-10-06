import torch

from ..core.mesh import mesh_data_from_comfy
from .stage import pbr_from_native_voxel, project_texture


def _image(a):
    return torch.from_numpy(a).to(torch.float32).div_(255.0).unsqueeze(0)


def _gray_image(a):
    return _image(a).unsqueeze(-1).repeat(1, 1, 1, 3)           # a real copy: downstream nodes may modify it in place


class STMTextureProjection:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESH", {"tooltip": "Mesh WITH UVs (use 'STM — UV Unwrap' first)."}),
                "pbr_voxels": ("STM_PBR_VOXELS,VOXEL", {"tooltip": "STM Texture Decoder output, or the native VOXEL of 'Trellis2 VAE Decode Texture'."}),
                "texture_size": ("INT", {"default": 2048, "min": 512, "max": 16384, "step": 256,
                                         "tooltip": "Output texture size. Never reduced automatically; baked in tiles, so memory is bounded."}),
                "gutter_px": ("INT", {"default": 0, "min": 0, "max": 64, "tooltip": "Texels of true-surface samples around each UV island (0 = auto: 8 at 2K, 16 at 4K)."}),
                "fill": (["pull_push", "none"], {"default": "pull_push", "tooltip": "pull_push fills the texels no triangle reaches; none leaves them black."}),
            },
            "optional": {"verbose": ("BOOLEAN", {"default": False})},
        }

    RETURN_TYPES = ("MESH", "IMAGE", "IMAGE", "IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("mesh", "base_color", "metallic", "roughness", "alpha", "uv_coverage")
    FUNCTION = "run"
    CATEGORY = "STM"

    def run(self, mesh, pbr_voxels, texture_size, gutter_px, fill, verbose=False):
        if hasattr(pbr_voxels, "voxel_colors"):                    # native ComfyUI VOXEL
            pbr_voxels = pbr_from_native_voxel(pbr_voxels)
        r = project_texture(mesh_data_from_comfy(mesh), pbr_voxels, int(texture_size), int(gutter_px), fill, verbose=bool(verbose))
        return (mesh, _image(r.base_color), _gray_image(r.metallic), _gray_image(r.roughness),
                torch.from_numpy(r.alpha).to(torch.float32).div_(255.0).unsqueeze(0),
                torch.from_numpy(r.coverage.astype("float32")).unsqueeze(0))


NODE_CLASS_MAPPINGS = {"STM_TextureProjection": STMTextureProjection}
NODE_DISPLAY_NAME_MAPPINGS = {"STM_TextureProjection": "STM — Texture Projection"}

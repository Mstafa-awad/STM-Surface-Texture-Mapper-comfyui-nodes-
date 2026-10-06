from ..core.mesh import comfy_mesh_from_data, mesh_data_from_comfy
from .stage import unwrap_mesh


class STMUVUnwrap:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESH",),
                "texture_size": ("INT", {"default": 2048, "min": 256, "max": 16384, "step": 256, "tooltip": "Target atlas size (sets the gutter between charts)."}),
                "cone_half_angle": ("FLOAT", {"default": 60.0, "min": 0.0, "max": 90.0, "step": 1.0,
                                              "tooltip": "Degrees. Largest angle between a face normal and its chart's projection axis; smaller = less stretch, more charts."}),
                "force_unwrap": ("BOOLEAN", {"default": False, "tooltip": "Re-unwrap even if the mesh already has UVs (default: keep existing UVs)."}),
            }
        }

    RETURN_TYPES = ("MESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "run"
    CATEGORY = "STM"

    def run(self, mesh, texture_size, cone_half_angle, force_unwrap):
        data = mesh_data_from_comfy(mesh)
        if data.uvs is not None and not force_unwrap:
            return (mesh,)                                   # untouched: no copy, no conversion
        return (comfy_mesh_from_data(unwrap_mesh(data, texture_size, cone_half_angle, force=True)),)


NODE_CLASS_MAPPINGS = {"STM_UVUnwrap": STMUVUnwrap}
NODE_DISPLAY_NAME_MAPPINGS = {"STM_UVUnwrap": "STM — UV Unwrap"}

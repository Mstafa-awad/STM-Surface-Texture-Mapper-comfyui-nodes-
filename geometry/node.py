from ..core.mesh import mesh_data_from_comfy
from .stage import build_geometry, voxelize


class STMDualGrid:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"mesh": ("MESH",), "resolution": ([512, 1024, 1536], {"default": 1024, "tooltip": "Voxel grid resolution (1536 is the largest TRELLIS.2 was trained for)."})},
                "optional": {"verbose": ("BOOLEAN", {"default": False})}}

    RETURN_TYPES = ("STM_DUAL_GRID", "INT")
    RETURN_NAMES = ("dual_grid", "voxel_count")
    FUNCTION = "run"
    CATEGORY = "STM"

    def run(self, mesh, resolution, verbose=False):
        grid = voxelize(build_geometry(mesh_data_from_comfy(mesh)), int(resolution), verbose=bool(verbose))
        return (grid, int(grid.coords.shape[0]))


NODE_CLASS_MAPPINGS = {"STM_DualGrid": STMDualGrid}
NODE_DISPLAY_NAME_MAPPINGS = {"STM_DualGrid": "STM — Geometry / Dual Grid"}

import os

import numpy as np
import torch

from ..core.mesh import comfy_mesh_from_data
from ..runtime.stage import log
from .stage import load_asset, resolve_mesh_path

try:
    import folder_paths
except Exception:                                      # pragma: no cover - outside ComfyUI
    folder_paths = None


def _image(a):
    return None if a is None else torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))[None]


class STMLoadMesh:
    """Reads a mesh file straight into a native MESH - with its textures (base colour, metallic-roughness, normal map) when the
    file has them, so it can go straight into the enhancement nodes. No VAE, no encoder, nothing is kept in VRAM."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh_path": ("STRING", {"default": "", "tooltip": "File name inside ComfyUI/input, or an absolute path (.glb, .gltf, .obj, .ply, .stl). Can be linked from a text node."}),
            },
            "optional": {
                "keep_uvs": ("BOOLEAN", {"default": True, "tooltip": "Keep the file's UVs (and textures). Off = geometry only, e.g. to re-texture from scratch."}),
                "normalize": ("BOOLEAN", {"default": False, "tooltip": "Centre and scale the mesh into the unit cube (what the old Mesh Encoder node output)."}),
                "load_textures": ("BOOLEAN", {"default": True, "tooltip": "Read the file's textures into the MESH. Parts with different textures are packed into one atlas."}),
            },
        }

    RETURN_TYPES = ("MESH", "INT", "INT", "IMAGE")
    RETURN_NAMES = ("mesh", "vertex_count", "face_count", "base_color")
    FUNCTION = "run"
    CATEGORY = "STM"

    def run(self, mesh_path, keep_uvs=True, normalize=False, load_textures=True):
        input_dir = folder_paths.get_input_directory() if folder_paths is not None else None
        path = resolve_mesh_path(mesh_path, input_dir)
        a = load_asset(path, bool(keep_uvs), bool(normalize), bool(load_textures))
        mesh = comfy_mesh_from_data(a.mesh, _image(a.texture), _image(a.metallic_roughness))
        mesh.normal_map = _image(a.normal_map)
        mesh.material = a.material
        what = " (with UVs)" if a.mesh.uvs is not None else ""
        if a.texture is not None:
            what = f" with a {a.texture.shape[1]}x{a.texture.shape[0]} base-colour texture"
        log(f"loaded {os.path.basename(path)}: {len(a.mesh.vertices):,} vertices / {len(a.mesh.faces):,} faces{what}"
            + "".join(f"; {n}" for n in a.notes))
        preview = _image(a.texture) if a.texture is not None else torch.full((1, 64, 64, 3), 0.5)
        return (mesh, int(len(a.mesh.vertices)), int(len(a.mesh.faces)), preview)


NODE_CLASS_MAPPINGS = {"STM_LoadMesh": STMLoadMesh}
NODE_DISPLAY_NAME_MAPPINGS = {"STM_LoadMesh": "STM — Load Mesh"}

"""Compatibility layer: the original all-in-one node and the TRIMESH converters, built from the same stage functions.

Node id, display name, category, required inputs, outputs and defaults are unchanged (``Trellis2MeshTexturingStandalone``), so
existing workflows keep loading.  The output mesh is, as before, the input rescaled to the unit cube (centre at the origin).
"""
import numpy as np
import torch

from ..core.mesh import _comfy_mesh_type, MeshData
from ..core.normalize import to_glb_axes, to_trellis_space
from ..dino.stage import extract_dino_features, images_from_comfy
from ..geometry.stage import build_geometry, voxelize
from ..load_models.handle import coerce_models
from ..shape_encoder.stage import encode_shape
from ..texture_decoder.stage import decode_texture
from ..texture_generation.stage import generate_texture_latent
from ..texture_projection.stage import project_texture, sample_vertex_colors
from ..uv_unwrap.stage import unwrap_mesh
from .meshio import _trimesh, comfy_mesh_to_trimesh, trimesh_to_comfy_mesh_kwargs


def _mesh_data_from_trimesh(tm, with_normals: bool) -> MeshData:
    verts = np.asarray(tm.vertices, dtype=np.float32)
    uvs = None
    vis = tm.visual
    if getattr(vis, "uv", None) is not None and len(vis.uv) == len(verts):
        uvs = np.asarray(vis.uv, dtype=np.float32).copy()
        uvs[:, 1] = 1.0 - uvs[:, 1]                               # trimesh (bottom-origin) -> glTF / texture-row convention
    normals = np.asarray(tm.vertex_normals, dtype=np.float32) if with_normals else None
    return MeshData(verts, np.asarray(tm.faces, dtype=np.int64), uvs, normals)


def _u8_image(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a)).to(torch.float32).div_(255.0).unsqueeze(0)


class Trellis2MeshTexturing:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("TRELLIS2PIPELINE",),
                "image": ("IMAGE",),
                "trimesh": ("TRIMESH",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7fffffff}),
                "texture_steps": ("INT", {"default": 12, "min": 1, "max": 100}),
                "texture_guidance_strength": ("FLOAT", {"default": 3.00, "min": 0.00, "max": 99.99, "step": 0.01}),
                "texture_guidance_rescale": ("FLOAT", {"default": 0.20, "min": 0.00, "max": 1.00, "step": 0.01}),
                "texture_rescale_t": ("FLOAT", {"default": 3.00, "min": 0.00, "max": 9.99, "step": 0.01}),
                "resolution": ([512, 1024, 1536], {"default": 1024, "tooltip": "Voxel grid resolution. 1536 is the largest the model was trained for."}),
                "texture_size": ("INT", {"default": 4096, "min": 512, "max": 16384,
                                         "tooltip": "Output texture size. Baked in tiles: memory is bounded by RAM, not VRAM (2048 and 4096 are fine on 8 GB)."}),
                "texture_alpha_mode": (["OPAQUE", "MASK", "BLEND"], {"default": "OPAQUE"}),
                "double_side_material": ("BOOLEAN", {"default": False, "tooltip": "Kept for compatibility. As in the original node the material is always written double sided."}),
                "texture_guidance_interval_start": ("FLOAT", {"default": 0.00, "min": 0.00, "max": 1.00, "step": 0.01}),
                "texture_guidance_interval_end": ("FLOAT", {"default": 0.90, "min": 0.00, "max": 1.00, "step": 0.01}),
                "max_views": ("INT", {"default": 4, "min": 1, "max": 16}),
                "bake_on_vertices": ("BOOLEAN", {"default": False}),
                "use_custom_normals": ("BOOLEAN", {"default": False}),
                "mesh_cluster_threshold_cone_half_angle_rad": ("FLOAT", {"default": 60.0, "min": 0.0, "max": 359.9,
                                                                          "tooltip": "Degrees. Used by the built-in UV unwrapper (meshes without UVs): max angle between a face normal and its chart axis."}),
                "sampler": (["euler", "heun", "rk4", "rk5"], {"default": "euler"}),
                "inpainting": (["pull_push", "telea", "ns"], {"default": "pull_push",
                                "tooltip": "Texture fill. 'pull_push' = built-in GPU multiscale fill with 3-D-consistent gutters. 'telea'/'ns' are legacy values kept so old workflows load; they run the same built-in fill (OpenCV was removed)."}),
                "verbose": ("BOOLEAN", {"default": False}),
                "dino_lock": ("FLOAT", {"default": 0.00, "min": 0.00, "max": 1.00, "step": 0.01}),
                "dino_substeps": ("INT", {"default": 4, "min": 1, "max": 99, "step": 1}),
                "dino_foundation_cap": ("FLOAT", {"default": 1.00, "min": 0.01, "max": 1.00, "step": 0.01}),
            },
            "optional": {
                "oom_fallback": ("BOOLEAN", {"default": True, "tooltip": "If the model runs out of memory at 1536/1024, retry automatically one resolution lower instead of failing."}),
                "mask": ("MASK", {"tooltip": "Optional foreground mask (white = object), one per image or a single one for all. When connected the image is cropped around the object and the background is set to black, exactly like the official TRELLIS.2 preprocessing."}),
                "preprocess_image": ("BOOLEAN", {"default": False, "tooltip": "Apply the official crop + black-background step using the alpha of the image (or the mask). Needs an RGBA image or a mask: no background-removal model is bundled (use any permissively licensed rembg node). Automatically on when a mask is connected."}),
            },
        }

    RETURN_TYPES = ("TRIMESH", "IMAGE", "IMAGE")
    RETURN_NAMES = ("trimesh", "base_color_texture", "metallic_roughness_texture")
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, pipeline, image, trimesh, seed, texture_steps, texture_guidance_strength, texture_guidance_rescale, texture_rescale_t,
                resolution, texture_size, texture_alpha_mode, double_side_material, texture_guidance_interval_start,
                texture_guidance_interval_end, max_views, bake_on_vertices, use_custom_normals,
                mesh_cluster_threshold_cone_half_angle_rad, sampler, inpainting, verbose, dino_lock, dino_substeps,
                dino_foundation_cap, oom_fallback=True, mask=None, preprocess_image=False, moge_camera_config=None):
        tm = _trimesh()
        models = coerce_models(pipeline)
        mesh = _mesh_data_from_trimesh(trimesh, bool(use_custom_normals))
        views = images_from_comfy(image, mask, bool(preprocess_image), int(max_views))

        dino = extract_dino_features(models, views, min(int(resolution), 1024))
        grid = voxelize(build_geometry(mesh), int(resolution), verbose=bool(verbose))
        shape = encode_shape(models, grid)
        del grid
        latent = generate_texture_latent(
            models, shape, dino, seed=seed, steps=texture_steps, guidance_strength=texture_guidance_strength,
            guidance_rescale=texture_guidance_rescale, rescale_t=texture_rescale_t,
            guidance_interval=(texture_guidance_interval_start, texture_guidance_interval_end), sampler=sampler, dino_lock=dino_lock,
            dino_substeps=dino_substeps, dino_foundation_cap=dino_foundation_cap, verbose=bool(verbose), oom_fallback=bool(oom_fallback))
        del shape
        pbr = decode_texture(models, latent)
        del latent

        if bake_on_vertices:
            rgba = sample_vertex_colors(mesh, pbr, opaque=texture_alpha_mode == "OPAQUE")
            kw = dict(vertices=to_glb_axes(to_trellis_space(mesh.vertices, pbr.transform)), faces=mesh.faces, vertex_colors=rgba, process=False)
            if use_custom_normals:
                kw["vertex_normals"] = mesh.normals
            placeholder = torch.zeros((1, 1, 1, 4), dtype=torch.float32)
            return (tm.Trimesh(**kw), placeholder, placeholder.clone())

        if mesh.uvs is None:
            mesh = unwrap_mesh(mesh, int(texture_size), float(mesh_cluster_threshold_cone_half_angle_rad), force=True)
        result = project_texture(mesh, pbr, int(texture_size), 0, "none" if inpainting == "none" else "pull_push", verbose=bool(verbose))
        from PIL import Image
        material = tm.visual.material.PBRMaterial(
            baseColorTexture=Image.fromarray(result.base_color_rgba), baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
            metallicRoughnessTexture=Image.fromarray(result.metallic_roughness), metallicFactor=1.0, roughnessFactor=1.0,
            alphaMode=texture_alpha_mode, doubleSided=True)                  # the original always wrote doubleSided=True
        uv = mesh.uvs.copy()
        uv[:, 1] = 1.0 - uv[:, 1]
        kw = dict(vertices=to_glb_axes(to_trellis_space(mesh.vertices, pbr.transform)), faces=mesh.faces, process=False,
                  visual=tm.visual.TextureVisuals(uv=uv, material=material))
        if use_custom_normals:
            kw["vertex_normals"] = mesh.normals
        return (tm.Trimesh(**kw), _u8_image(result.base_color_rgba), _u8_image(result.metallic_roughness))


class Trellis2MeshToTrimesh:
    """ComfyUI-native MESH -> TRIMESH (for nodes that still use the TRIMESH type)."""

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"mesh": ("MESH",)},
                "optional": {"batch_index": ("INT", {"default": 0, "min": 0, "max": 4095, "tooltip": "Which mesh of the batch to convert."})}}

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "convert"
    CATEGORY = "Trellis2Wrapper"

    def convert(self, mesh, batch_index=0):
        return (comfy_mesh_to_trimesh(mesh, batch_index),)


class Trellis2TrimeshToMesh:
    """TRIMESH -> ComfyUI-native MESH, in memory (the base-colour alpha is dropped: MESH.texture is RGB)."""

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"trimesh": ("TRIMESH",)}}

    RETURN_TYPES = ("MESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "convert"
    CATEGORY = "Trellis2Wrapper"

    def convert(self, trimesh):
        return (_comfy_mesh_type()(**trimesh_to_comfy_mesh_kwargs(trimesh)),)


NODE_CLASS_MAPPINGS = {
    "Trellis2MeshTexturingStandalone": Trellis2MeshTexturing,
    "Trellis2MeshToTrimesh": Trellis2MeshToTrimesh,
    "Trellis2TrimeshToMesh": Trellis2TrimeshToMesh,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Trellis2MeshTexturingStandalone": "Trellis2 - Mesh Texturing (Standalone)",
    "Trellis2MeshToTrimesh": "Trellis2 - MESH to TRIMESH",
    "Trellis2TrimeshToMesh": "Trellis2 - TRIMESH to MESH",
}

"""Reading the base colour of a native MESH (texture x glTF base-colour factor) and writing an enhanced one back."""
import copy

import numpy as np
import torch

from ..core.mesh import MeshData


def _srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(min=0)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x.clamp(min=1e-12) ** (1 / 2.4) - 0.055)


def _factor(mesh) -> torch.Tensor:
    mat = getattr(mesh, "material", None)
    f = mat.get("base_color_factor") if isinstance(mat, dict) else None
    if f is None:
        return torch.ones(3)
    return torch.tensor([float(c) for c in list(f)[:3]], dtype=torch.float32).clamp(min=1e-3)


def mesh_appearance(mesh, override=None):
    """-> (base colour as it looks [H,W,3] float32 CPU, glTF base-colour factor [3]) of batch item 0."""
    tex = override if override is not None else getattr(mesh, "texture", None)
    if tex is None:
        raise ValueError("The mesh has no base-colour texture. Texture it first (STM texturing nodes, 'Apply Texture to Mesh'), "
                         "load it with 'STM — Load Mesh' / 'Get 3D Components' from a file that has a texture, or connect "
                         "an IMAGE to 'base_color'.")
    t = tex[0] if tex.ndim == 4 else tex
    t = t[..., :3].detach().float().cpu().clamp(0, 1)
    factor = _factor(mesh)
    if bool((factor != 1).any()):
        t = _linear_to_srgb(_srgb_to_linear(t) * factor).clamp(0, 1)
    return t, factor


def store_texture(appearance: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`mesh_appearance`: the texture that, times the factor, looks like ``appearance``."""
    if bool((factor != 1).any()):
        return _linear_to_srgb(_srgb_to_linear(appearance) / factor).clamp(0, 1)
    return appearance.clamp(0, 1)


def unit_uvs(data: MeshData, tolerance: float = 0.01) -> MeshData:
    """UVs inside the unit square (tiny overshoot clamped).  Repeating (tiling) UVs cannot be enhanced by projection: several
    surface points would share one texel."""
    if data.uvs is None:
        raise ValueError("The mesh has no UVs. Unwrap it ('Unwrap Mesh UVs' / 'STM — UV Unwrap') and texture it first.")
    uv = np.asarray(data.uvs, dtype=np.float32)
    lo, hi = float(uv.min()), float(uv.max())
    if lo < -tolerance or hi > 1 + tolerance:
        raise ValueError(f"The UVs reach {lo:.2f}..{hi:.2f}: the texture repeats (tiling UVs), so one texel sits on several places "
                         "and cannot take detail from the views. Re-unwrap the mesh and bake its look into a single atlas first.")
    if lo < 0 or hi > 1:
        data = copy.copy(data)
        data.uvs = np.clip(uv, 0.0, 1.0)
    return data


def with_texture(mesh, data: MeshData, texture: torch.Tensor):
    """New single-item MESH: geometry/UVs of ``data`` (batch item 0 of ``mesh``), every other map and material setting of item 0
    carried over, and ``texture`` [H,W,3] as its base colour."""
    from ..core.mesh import comfy_mesh_from_data

    def item0(name):
        t = getattr(mesh, name, None)
        return None if t is None else t[0:1].detach().cpu()

    out = comfy_mesh_from_data(data, texture[None].float().cpu(), item0("metallic_roughness"))
    counts = getattr(mesh, "vertex_counts", None)
    tangents = getattr(mesh, "tangents", None)
    if tangents is not None:
        nv = int(counts[0]) if counts is not None else tangents.shape[1]
        tangents = tangents[0:1, :nv].detach().cpu()
        if tangents.shape[1] != data.vertices.shape[0]:
            tangents = None
    for name, value in (("tangents", tangents), ("normal_map", item0("normal_map")), ("emissive", item0("emissive")),
                        ("material", getattr(mesh, "material", None)), ("unlit", bool(getattr(mesh, "unlit", False))),
                        ("occlusion_in_mr", bool(getattr(mesh, "occlusion_in_mr", False)))):
        setattr(out, name, value)
    return out

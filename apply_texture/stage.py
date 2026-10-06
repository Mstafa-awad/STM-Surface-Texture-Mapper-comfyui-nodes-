"""Apply stage: attach (optionally image-processed) texture maps to a mesh that has UVs."""
import copy
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..core.mesh import normalize_uvs_to_unit


def _normalized_uvs(mesh):
    """UVs fitted into the unit square exactly like the projection did; the original tensor is returned when nothing changes."""
    uvs = mesh.uvs
    if isinstance(uvs, (list, tuple)):
        return [_fit(u) for u in uvs]
    if uvs.ndim == 2:
        return _fit(uvs)
    counts = getattr(mesh, "vertex_counts", None)
    out = uvs
    for i in range(uvs.shape[0]):
        n = int(counts[i]) if counts is not None else uvs.shape[1]
        fitted = normalize_uvs_to_unit(uvs[i, :n].detach().cpu().numpy())
        if not np.array_equal(fitted, uvs[i, :n].detach().cpu().numpy()):
            out = out.clone() if out is uvs else out
            out[i, :n] = torch.from_numpy(fitted).to(uvs)
    return out


def _fit(uv: torch.Tensor) -> torch.Tensor:
    src = uv.detach().cpu().numpy()
    fitted = normalize_uvs_to_unit(src)
    return uv if np.array_equal(fitted, src) else torch.from_numpy(fitted).to(uv)


def apply_texture(mesh, base_color: torch.Tensor, metallic: Optional[torch.Tensor] = None, roughness: Optional[torch.Tensor] = None):
    """Return a copy of the native ``MESH`` carrying ``texture`` and a packed glTF ``metallic_roughness`` map.

    Packing follows ComfyUI's own Apply Texture node: R = occlusion (1 = none), G = roughness (default 1), B = metallic (default 0);
    channel 0 of the given IMAGEs is used and maps of different sizes are resized to a common size.
    """
    if mesh.uvs is None:
        raise ValueError("The mesh has no UVs: connect the mesh that was used for the projection.")
    out = copy.copy(mesh)
    out.uvs = _normalized_uvs(mesh)                         # the same fit the projection used, so texture and UVs agree
    out.texture = base_color[..., :3].float().clamp(0.0, 1.0).cpu()
    if metallic is not None or roughness is not None:
        given = [m for m in (metallic, roughness) if m is not None]
        b = int(given[0].shape[0])
        h, w = max(int(m.shape[1]) for m in given), max(int(m.shape[2]) for m in given)

        def channel(img, default):
            if img is None:
                return torch.full((b, h, w, 1), float(default))
            t = img[..., 0:1].float().clamp(0.0, 1.0).cpu()
            if t.shape[1] != h or t.shape[2] != w:
                t = F.interpolate(t.permute(0, 3, 1, 2), size=(h, w), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
            return t

        out.metallic_roughness = torch.cat([torch.ones((b, h, w, 1)), channel(roughness, 1.0), channel(metallic, 0.0)], dim=-1)
    return out

"""Conversions between ComfyUI's native ``MESH`` type and ``trimesh.Trimesh`` (the ``TRIMESH`` type of this node).

Facts this module relies on (read from ComfyUI's source, ``comfy_api/latest/_util/geometry_types.py`` and
``comfy_extras/nodes_save_3d.py``, 2026-10-03):

* ``MESH`` holds *batched* tensors: ``vertices (B,N,3)``, ``faces (B,M,3)``, optional ``uvs (B,N,2)``, ``vertex_colors (B,N,3|4)``,
  ``texture (B,H,W,3)`` (base colour), ``metallic_roughness (B,H,W,3)`` (R unused, G=roughness, B=metallic), ``normals (B,N,3)`` and,
  for variable-size batches, ``vertex_counts`` / ``face_counts`` giving the real lengths.
* ``save_glb`` writes ``MESH.uvs`` unchanged into glTF ``TEXCOORD_0``, i.e. MESH UVs use the **glTF convention (v = 0 at the top of
  the image)**, whereas ``trimesh`` stores UVs with v = 0 at the bottom and flips on export.  Hence the ``1 - v`` below.
* Coordinates are passed through unchanged (both sides are Y-up glTF-style, as the original node's TRIMESH input was).

No ComfyUI import happens here: ``trimesh_to_comfy_mesh_kwargs`` returns the constructor arguments for ``Types.MESH``.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image


def _trimesh():
    try:
        import trimesh
        return trimesh
    except ImportError as e:                                  # pragma: no cover
        raise ImportError("The legacy TRIMESH nodes need the 'trimesh' package (pip install trimesh). The STM — nodes do not.") from e


def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    a = t.detach().cpu().float().clamp(0, 1).numpy()
    return Image.fromarray((a * 255.0 + 0.5).astype(np.uint8))


def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(img.convert("RGB"))).to(torch.float32).div_(255.0).unsqueeze(0)


def _batch_size(t) -> int:
    """MESH tensors may be batched ``(B,N,3)``, a Python list of per-item tensors (variable-size batches) or unbatched ``(N,3)``."""
    if isinstance(t, (list, tuple)):
        return len(t)
    return int(t.shape[0]) if t.ndim == 3 else 1


def _pick(t, i, n=None, image=False):
    """Item ``i`` of a (possibly list / batched / unbatched) MESH tensor as numpy, cut to ``n`` rows. ``image``: (H,W,3) maps."""
    if t is None:
        return None
    if isinstance(t, (list, tuple)):
        x = t[i]
    else:
        batched = t.ndim == (4 if image else 3)
        x = t[i] if batched else t
    if n is not None:
        x = x[:n]
    return x


def comfy_mesh_to_trimesh(mesh: Any, batch_index: int = 0):
    """MESH (duck-typed object with the attributes listed above) -> ``trimesh.Trimesh``.

    Accepts all layouts ComfyUI's own nodes produce: batched tensors, lists of per-item tensors and unbatched tensors."""
    trimesh = _trimesh()
    b = _batch_size(mesh.vertices)
    if not 0 <= int(batch_index) < b:
        raise ValueError(f"batch_index {batch_index} is out of range for a MESH batch of {b}")
    i = int(batch_index)
    vc, fc = getattr(mesh, "vertex_counts", None), getattr(mesh, "face_counts", None)
    nv = int(vc[i]) if vc is not None else None
    nf = int(fc[i]) if fc is not None else None

    def item(t, n=None):
        x = _pick(t, i, n)
        return None if x is None else x.detach().cpu().numpy()

    verts = item(mesh.vertices, nv).astype(np.float32)
    faces = item(mesh.faces, nf).astype(np.int64)
    if verts.shape[0] == 0 or faces.shape[0] == 0:
        raise ValueError("The MESH is empty (no vertices or no faces)")
    if faces.min() < 0 or faces.max() >= verts.shape[0]:
        raise ValueError(f"MESH face indices out of range [0, {verts.shape[0]}): {int(faces.min())}..{int(faces.max())}")
    tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    uvs = item(getattr(mesh, "uvs", None), nv)
    if uvs is not None:
        if uvs.shape[0] != verts.shape[0]:
            raise ValueError(f"MESH uvs ({uvs.shape[0]}) must be 1:1 with vertices ({verts.shape[0]})")
        uv = uvs.astype(np.float32).copy()
        uv[:, 1] = 1.0 - uv[:, 1]                                   # glTF (top-origin) -> trimesh (bottom-origin)
        tex = _pick(getattr(mesh, "texture", None), i, image=True)
        mr = _pick(getattr(mesh, "metallic_roughness", None), i, image=True)
        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=_tensor_to_pil(tex) if tex is not None else None,
            metallicRoughnessTexture=_tensor_to_pil(mr) if mr is not None else None)
        tm.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
    else:
        colors = item(getattr(mesh, "vertex_colors", None), nv)
        if colors is not None:
            tm.visual = trimesh.visual.ColorVisuals(tm, vertex_colors=(np.clip(colors, 0, 1) * 255 + 0.5).astype(np.uint8))
    normals = item(getattr(mesh, "normals", None), nv)
    if normals is not None and normals.shape[0] == verts.shape[0]:
        tm.vertex_normals = normals.astype(np.float64)
    return tm


def trimesh_to_comfy_mesh_kwargs(tm) -> Dict[str, Optional[torch.Tensor]]:
    """``trimesh.Trimesh`` -> keyword arguments for ``comfy_api.latest.Types.MESH`` (batch size 1, CPU tensors)."""
    trimesh = _trimesh()
    if not isinstance(tm, trimesh.Trimesh) or len(tm.faces) == 0:
        raise ValueError("Expected a trimesh.Trimesh with at least one face")
    kw: Dict[str, Optional[torch.Tensor]] = dict(
        vertices=torch.from_numpy(np.asarray(tm.vertices, dtype=np.float32).copy()).unsqueeze(0),
        faces=torch.from_numpy(np.asarray(tm.faces, dtype=np.int64).copy()).unsqueeze(0),
        uvs=None, vertex_colors=None, texture=None, metallic_roughness=None, normals=None)
    vis = tm.visual
    if isinstance(vis, trimesh.visual.TextureVisuals) and vis.uv is not None:
        uv = np.asarray(vis.uv, dtype=np.float32).copy()
        uv[:, 1] = 1.0 - uv[:, 1]                                   # trimesh (bottom-origin) -> glTF (top-origin)
        kw["uvs"] = torch.from_numpy(uv).unsqueeze(0)
        mat = vis.material
        base = getattr(mat, "baseColorTexture", None)
        if base is None:
            base = getattr(mat, "image", None)                      # SimpleMaterial
        if base is not None:
            kw["texture"] = _pil_to_tensor(base)                    # alpha (if any) is dropped: MESH.texture is RGB
        mr = getattr(mat, "metallicRoughnessTexture", None)
        if mr is not None:
            kw["metallic_roughness"] = _pil_to_tensor(mr)
    elif getattr(vis, "kind", None) == "vertex":
        vc = np.asarray(vis.vertex_colors, dtype=np.float32) / 255.0
        kw["vertex_colors"] = torch.from_numpy(vc).unsqueeze(0)
    try:                                                            # keep normals only if they were set explicitly (cached)
        cached = tm._cache.cache.get("vertex_normals")
    except Exception:
        cached = None
    if cached is not None and len(cached) == len(tm.vertices):
        kw["normals"] = torch.from_numpy(np.asarray(cached, dtype=np.float32).copy()).unsqueeze(0)
    return kw

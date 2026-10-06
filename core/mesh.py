"""In-memory mesh container and conversion to/from ComfyUI's native ``MESH`` (no disk serialisation anywhere).

``MESH`` (``comfy_api.latest.Types.MESH``) holds batched tensors: ``vertices (B,N,3)``, ``faces (B,M,3)``, optional
``uvs (B,N,2)``, ``vertex_colors (B,N,3|4)``, ``normals (B,N,3)``, ``texture (B,H,W,3)`` and ``metallic_roughness (B,H,W,3)``
(G = roughness, B = metallic); variable-size batches are padded and carry ``vertex_counts`` / ``face_counts``.  Its UVs use the
glTF convention (v = 0 at the *top* of the image), which is exactly the "texture row = v * H" convention the baker uses, so UVs
pass through unchanged.
"""
from dataclasses import dataclass, replace
from typing import Any, Optional

import numpy as np
import torch


@dataclass
class MeshData:
    vertices: np.ndarray                      # float32 [V, 3]
    faces: np.ndarray                         # int64   [F, 3]
    uvs: Optional[np.ndarray] = None          # float32 [V, 2], glTF convention
    normals: Optional[np.ndarray] = None      # float32 [V, 3]
    vertex_colors: Optional[np.ndarray] = None  # float32 [V, 3|4] in 0..1

    def remapped(self, vertices, faces, uvs, vmap) -> "MeshData":
        """New mesh after a vertex-duplicating operation (UV unwrap); per-vertex attributes follow ``vmap``."""
        return replace(self, vertices=vertices, faces=faces, uvs=uvs,
                       normals=None if self.normals is None else self.normals[vmap],
                       vertex_colors=None if self.vertex_colors is None else self.vertex_colors[vmap])


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted smooth vertex normals."""
    v = vertices.astype(np.float64)
    fn = np.cross(v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]])
    out = np.zeros_like(v)
    for k in range(3):
        np.add.at(out, faces[:, k], fn)
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return (out / np.maximum(n, 1e-20)).astype(np.float32)


def normalize_uvs_to_unit(uvs: np.ndarray) -> np.ndarray:
    """Uniformly fit the UV bounding box into [0,1] when it spills outside (aspect preserved); unchanged if already inside.

    ComfyUI's native bake and apply nodes do the same, so UVs from any unwrap node line up with the baked textures."""
    uvs = np.asarray(uvs, dtype=np.float32)
    lo, hi = uvs.min(axis=0), uvs.max(axis=0)
    if not (lo.min() < -1e-4 or hi.max() > 1.0001):
        return uvs
    extent = float((hi - lo).max())
    return ((uvs - lo) / extent).astype(np.float32) if extent > 0 else uvs


# ------------------------------------------------------------------------------------------ MESH -> MeshData
def _batch_size(t) -> int:
    if isinstance(t, (list, tuple)):
        return len(t)
    return int(t.shape[0]) if t.ndim == 3 else 1


def _pick(t, i, n=None):
    if t is None:
        return None
    x = t[i] if isinstance(t, (list, tuple)) or t.ndim == 3 else t
    x = x if n is None else x[:n]
    return x.detach().cpu().numpy()


def mesh_data_from_comfy(mesh: Any, batch_index: int = 0) -> MeshData:
    """Native ``MESH`` (batched / list / unbatched layouts) -> :class:`MeshData` (item ``batch_index``)."""
    b = _batch_size(mesh.vertices)
    if not 0 <= int(batch_index) < b:
        raise ValueError(f"batch_index {batch_index} is out of range for a MESH batch of {b}")
    i = int(batch_index)
    vc, fc = getattr(mesh, "vertex_counts", None), getattr(mesh, "face_counts", None)
    nv = int(vc[i]) if vc is not None else None
    nf = int(fc[i]) if fc is not None else None
    verts = _pick(mesh.vertices, i, nv).astype(np.float32)
    faces = _pick(mesh.faces, i, nf).astype(np.int64)
    if verts.shape[0] == 0 or faces.shape[0] == 0:
        raise ValueError("The MESH is empty (no vertices or no faces)")
    if faces.min() < 0 or faces.max() >= verts.shape[0]:
        raise ValueError(f"MESH face indices out of range [0, {verts.shape[0]}): {int(faces.min())}..{int(faces.max())}")
    if not np.isfinite(verts).all():
        raise ValueError("The MESH contains NaN/inf vertices")

    def per_vertex(name, width=None):
        a = _pick(getattr(mesh, name, None), i, nv)
        if a is None:
            return None
        if a.shape[0] != verts.shape[0]:
            raise ValueError(f"MESH {name} ({a.shape[0]}) must be 1:1 with vertices ({verts.shape[0]})")
        return a.astype(np.float32)

    return MeshData(verts, faces, per_vertex("uvs"), per_vertex("normals"), per_vertex("vertex_colors"))


# ------------------------------------------------------------------------------------------ MeshData -> MESH
def _comfy_mesh_type():
    try:
        from comfy_api.latest import Types
        return Types.MESH
    except Exception as e:                                  # pragma: no cover - older ComfyUI
        raise RuntimeError("This ComfyUI has no native MESH type (comfy_api.latest.Types.MESH). Update ComfyUI.") from e


def comfy_mesh_from_data(m: MeshData, texture: Optional[torch.Tensor] = None, metallic_roughness: Optional[torch.Tensor] = None):
    """:class:`MeshData` (+ optional ``(1,H,W,3)`` maps) -> native ``MESH`` with batch size 1."""
    def t(a, dtype=torch.float32):
        return None if a is None else torch.from_numpy(np.ascontiguousarray(a)).to(dtype).unsqueeze(0)

    return _comfy_mesh_type()(
        vertices=t(m.vertices), faces=t(m.faces, torch.int64), uvs=t(m.uvs), vertex_colors=t(m.vertex_colors), normals=t(m.normals),
        texture=texture, metallic_roughness=metallic_roughness)

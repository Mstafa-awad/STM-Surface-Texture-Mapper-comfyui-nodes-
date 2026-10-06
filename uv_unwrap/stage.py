"""UV unwrap stage (native chart unwrapper, no external dependency)."""
import numpy as np

from ..core.mesh import MeshData
from .uv import unwrap_native


def unwrap_mesh(mesh: MeshData, texture_size: int = 2048, cone_half_angle_deg: float = 60.0, force: bool = False) -> MeshData:
    """Return ``mesh`` with UVs. Vertices on chart seams are duplicated (same position, own UV). Existing UVs are kept unless ``force``."""
    if mesh.uvs is not None and not force:
        return mesh
    if not np.isfinite(mesh.vertices).all():
        raise ValueError("The mesh contains NaN/inf vertices")
    vertices, faces, uvs, vmap = unwrap_native(mesh.vertices, mesh.faces, texture_size=int(texture_size),
                                               cone_half_angle_deg=float(cone_half_angle_deg) if cone_half_angle_deg > 0 else 60.0)
    return mesh.remapped(vertices, faces, uvs, vmap)

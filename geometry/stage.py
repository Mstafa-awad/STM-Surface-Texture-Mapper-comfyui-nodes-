"""Geometry stage: normalise the mesh into TRELLIS space and build the sparse flexible dual grid."""
import numpy as np
import torch

from ..core.mesh import MeshData
from ..core.normalize import compute_transform, to_trellis_space
from ..core.types import DualGrid, Geometry
from ..runtime.device import StmMemoryError, is_oom, pick_device, soft_empty_cache
from ..runtime.stage import log, stage_scope
from ..runtime.device import device_memory
from ..runtime.vram import current_plan, stage_reserve_bytes
from .dual_grid import mesh_to_flexible_dual_grid

_AABB = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]


def build_geometry(mesh: MeshData) -> Geometry:
    if mesh.faces.size == 0:
        raise ValueError("The mesh has no faces")
    transform = compute_transform(mesh.vertices)
    return Geometry(to_trellis_space(mesh.vertices, transform), np.ascontiguousarray(mesh.faces, dtype=np.int64), transform)


@torch.no_grad()
def voxelize(geometry: Geometry, resolution: int, device_pref=None, verbose: bool = False) -> DualGrid:
    """Dual grid at ``resolution``; chunk sizes follow the measured free memory, OOM -> smaller chunks -> CPU."""
    device = pick_device(device_pref)
    plan = current_plan(device, resolution=resolution, n_faces=len(geometry.faces))
    vertices, faces = torch.from_numpy(geometry.vertices), torch.from_numpy(geometry.faces)
    work_device = device if plan.dual_on_device else torch.device("cpu")
    last = None
    with stage_scope("dual grid", device, reserve=stage_reserve_bytes("dual grid", resolution, device_memory(device).total) if plan.dual_on_device else 0):
        for attempt in range(4):
            try:
                coords, dual, inter = mesh_to_flexible_dual_grid(
                    vertices, faces, grid_size=resolution, aabb=_AABB, face_weight=1.0, boundary_weight=0.2, regularization_weight=1e-2,
                    device=work_device, max_rows=plan.dual_max_rows, max_cols=plan.dual_max_cols, max_candidates=plan.dual_max_candidates,
                    progress=log if verbose else None)
                return DualGrid(coords, dual, inter, int(resolution), geometry)
            except BaseException as e:                       # noqa: BLE001
                if not is_oom(e):
                    raise
                last = e
                soft_empty_cache()
                plan.shrink()
                if attempt >= 1:
                    work_device = torch.device("cpu")
    raise StmMemoryError(f"Voxelising the mesh at resolution {resolution} ran out of memory. Lower the resolution.") from last

"""Mesh normalisation shared by every stage that has to agree on TRELLIS space."""
import numpy as np

from .types import NormTransform


def compute_transform(vertices: np.ndarray) -> NormTransform:
    v = np.asarray(vertices, dtype=np.float64)
    if not np.isfinite(v).all():
        raise ValueError("The mesh contains NaN/inf vertices")
    lo, hi = v.min(axis=0), v.max(axis=0)
    extent = float((hi - lo).max())
    if extent <= 0:
        raise ValueError("The mesh is degenerate (all vertices coincide)")
    return NormTransform(center=(lo + hi) / 2, scale=0.99999 / extent)


def to_trellis_space(vertices: np.ndarray, t: NormTransform, frame: str = "z_up") -> np.ndarray:
    """Centre and scale to the unit cube; for the TRELLIS frame also apply the axis convention (x, y, z) -> (x, -z, y).

    ``frame="y_up"`` keeps the glTF axes (the frame of ComfyUI's native Trellis2 voxels)."""
    v = (np.asarray(vertices, dtype=np.float64) - t.center) * t.scale
    if frame == "z_up":
        v[:, [1, 2]] = np.stack([-v[:, 2], v[:, 1].copy()], axis=1)
    if not (np.all(v >= -0.5) and np.all(v <= 0.5)):
        raise ValueError("vertices out of range after normalisation")
    return v.astype(np.float32)


def to_glb_axes(v: np.ndarray) -> np.ndarray:
    """Inverse of the axis swap only (the scale/centre are not undone)."""
    a = np.array(v, copy=True)
    a[:, 1], a[:, 2] = a[:, 2].copy(), -a[:, 1].copy()
    return a

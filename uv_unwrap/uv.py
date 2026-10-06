"""Native UV unwrapper (NumPy + SciPy + PyTorch only).

1. Charts: every face goes to the closest of K projection directions (K is the smallest direction set whose covering radius
   is within ``cone_half_angle``, so planar projection can never flip a triangle); faces of one direction that are connected
   through shared edges form a chart.
2. Parametrisation: orthographic projection along the chart axis, rotated to the chart's principal axes, one global scale.
3. Packing: shelf packing with a gutter-sized margin; binary search for the largest scale that fits the unit square.
4. Validation: the atlas is rasterised; charts whose triangles overlap are split with a finer direction set (last resort:
   single-face charts), so the result is always overlap free.

UVs use the glTF convention: texture row = v * H.
"""
from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from ..core.raster import rasterize_uv_coverage
from ..runtime.stage import log

__all__ = ["unwrap_native"]

_DIR_COUNTS = (6, 14, 26, 50, 98, 194, 386)
_dir_cache: dict = {}


def _fibonacci(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)


def _directions(n: int) -> Tuple[np.ndarray, float]:
    """Direction set with ``n`` axes and its covering radius in degrees."""
    if n in _dir_cache:
        return _dir_cache[n]
    if n == 6:
        d = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=np.float64)
    elif n == 14:
        c = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=np.float64) / np.sqrt(3)
        d = np.concatenate([_directions(6)[0], c])
    else:
        d = _fibonacci(n)
    probe = _fibonacci(6000)
    radius = float(np.degrees(np.arccos(np.clip((probe @ d.T).max(axis=1), -1, 1)).max()))
    _dir_cache[n] = (d, radius)
    return _dir_cache[n]


def _pick_direction_set(cone_deg: float, min_index: int = 0) -> int:
    for i, n in enumerate(_DIR_COUNTS):
        if i >= min_index and _directions(n)[1] <= cone_deg:
            return i
    return len(_DIR_COUNTS) - 1


def _face_geometry(v: np.ndarray, f: np.ndarray):
    p0, p1, p2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    cr = np.cross(p1 - p0, p2 - p0)
    a2 = np.linalg.norm(cr, axis=1)
    n = cr / np.maximum(a2, 1e-30)[:, None]
    n[a2 < 1e-30] = (0.0, 0.0, 1.0)
    return n, 0.5 * a2


def _edge_adjacency(f: np.ndarray):
    """Pairs of faces sharing an edge (consecutive faces on each edge; handles non-manifold edges)."""
    nf = f.shape[0]
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    fid = np.tile(np.arange(nf), 3)
    e.sort(axis=1)
    key = e[:, 0].astype(np.int64) * (int(f.max()) + 1) + e[:, 1]
    order = np.argsort(key, kind="stable")
    ks, fs = key[order], fid[order]
    same = ks[1:] == ks[:-1]
    return fs[:-1][same], fs[1:][same]


def _components(nf: int, a: np.ndarray, b: np.ndarray, label: np.ndarray) -> np.ndarray:
    """Connected components over edges whose two faces share ``label``."""
    keep = label[a] == label[b]
    g = coo_matrix((np.ones(int(keep.sum()), dtype=np.int8), (a[keep], b[keep])), shape=(nf, nf))
    return connected_components(g, directed=False)[1]


def _project(v: np.ndarray, f: np.ndarray, axis: np.ndarray):
    """Orthographic projection of the faces' vertices; returns (per-corner 2-D coords [n,3,2], bbox w, h)."""
    p = v[f]                                                  # [n,3,3]
    helper = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(helper, axis)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    xy = np.stack([p @ e1, p @ e2], axis=-1)                  # [n,3,2]
    pts = xy.reshape(-1, 2)
    c = pts - pts.mean(axis=0)
    # rotate to the principal axes (minimum-area bounding box for elongated charts)
    if len(c) >= 3:
        w, vecs = np.linalg.eigh(c.T @ c)
        rot = vecs[:, ::-1]
        if np.linalg.det(rot) < 0:
            rot[:, 1] *= -1
        xy = xy @ rot
    lo = xy.reshape(-1, 2).min(axis=0)
    hi = xy.reshape(-1, 2).max(axis=0)
    xy = xy - lo
    return xy, float(hi[0] - lo[0]), float(hi[1] - lo[1])


def _shelf_pack(widths: np.ndarray, heights: np.ndarray, pad: float):
    """Pack rectangles (already scaled, in unit-square units) into width 1.  Returns (x, y, total_height, ok)."""
    n = len(widths)
    order = np.argsort(-heights, kind="stable")
    x = np.zeros(n)
    y = np.zeros(n)
    cx = cy = row_h = 0.0
    for i in order:
        w, h = widths[i] + pad, heights[i] + pad
        if w > 1.0:
            return x, y, np.inf, False
        if cx + w > 1.0:
            cy += row_h
            cx = 0.0
            row_h = 0.0
        x[i], y[i] = cx + pad * 0.5, cy + pad * 0.5
        cx += w
        row_h = max(row_h, h)
    return x, y, cy + row_h, True


def _layout(chart_wh, pad: float):
    wh = np.asarray(chart_wh)
    lo, hi = 1e-6, 1.0 / max(float(wh.max()), 1e-9)
    best = None
    for _ in range(40):                                       # binary search for the largest scale that fits
        s = 0.5 * (lo + hi)
        x, y, total, ok = _shelf_pack(wh[:, 0] * s, wh[:, 1] * s, pad)
        if ok and total <= 1.0:
            best, lo = (s, x, y), s
        else:
            hi = s
    if best is None:
        s = lo
        x, y, _, _ = _shelf_pack(wh[:, 0] * s, wh[:, 1] * s, pad)
        best = (s, x, y)
    return best


def _build_layout(v, f, chart, axis_of_face, pad):
    """Project every chart, pack all charts into the unit square; returns (uv per face corner [F,3,2], scale)."""
    nf = len(f)
    n_charts = int(chart.max()) + 1
    order = np.argsort(chart, kind="stable")
    bounds = np.searchsorted(chart[order], np.arange(n_charts + 1))
    xy_faces = np.zeros((nf, 3, 2))
    wh = np.zeros((n_charts, 2))
    for c in range(n_charts):
        idx = order[bounds[c]:bounds[c + 1]]
        xy, w, h = _project(v, f[idx], axis_of_face[idx[0]])
        xy_faces[idx] = xy
        wh[c] = (w, h)
    s, cx, cy = _layout(wh, pad)
    uv_faces = xy_faces * s + np.stack([cx[chart], cy[chart]], axis=1)[:, None, :]
    return uv_faces, s


def _overlapping_charts(uv_faces, chart, res):
    """Boolean per chart: True if its triangles overlap each other in UV space (checked by rasterisation)."""
    nf = len(chart)
    n_charts = int(chart.max()) + 1
    uvp = torch.from_numpy((uv_faces.reshape(-1, 2) * res).astype(np.float32))
    tri = torch.arange(nf * 3, dtype=torch.int64).reshape(nf, 3)
    owner = rasterize_uv_coverage(uvp, tri, res, res).numpy().reshape(-1)
    owned = np.bincount(owner[owner >= 0], minlength=nf).astype(np.float64)
    d1, d2 = uv_faces[:, 1] - uv_faces[:, 0], uv_faces[:, 2] - uv_faces[:, 0]
    expected = 0.5 * np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) * res * res
    exp_c = np.bincount(chart, weights=expected, minlength=n_charts)
    own_c = np.bincount(chart, weights=owned, minlength=n_charts)
    return (exp_c >= 64) & (own_c < 0.97 * exp_c)


def _smooth_labels(label, normals, dirs, a, b, iterations: int = 4, max_angle_deg: float = 70.0):
    """Majority filter over edge-adjacent faces: noisy dense meshes otherwise shatter into thousands of tiny charts.
    A face only takes a neighbour label whose direction stays within ``max_angle_deg`` of its normal (no flipped projections)."""
    nf = len(label)
    if len(a) == 0:
        return label
    # up to 3 edge neighbours per face (manifold edges); missing neighbours point at the face itself
    nb = np.repeat(np.arange(nf)[:, None], 3, axis=1)
    x = np.concatenate([a, b])
    y = np.concatenate([b, a])
    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y[order]
    slot = np.arange(len(xs)) - np.searchsorted(xs, xs, side="left")
    ok = slot < 3
    nb[xs[ok], slot[ok]] = ys[ok]
    cos_lim = np.cos(np.radians(max_angle_deg))
    for _ in range(iterations):
        l0, l1, l2 = label[nb[:, 0]], label[nb[:, 1]], label[nb[:, 2]]
        cand = np.where((l0 == l1) | (l0 == l2), l0, np.where(l1 == l2, l1, label))
        allowed = np.einsum("ij,ij->i", normals, dirs[cand]) > cos_lim
        new = np.where(allowed, cand, label)
        if np.array_equal(new, label):
            break
        label = new
    return label


def unwrap_native(
    vertices: np.ndarray,
    faces: np.ndarray,
    texture_size: int = 2048,
    cone_half_angle_deg: float = 60.0,
    gutter_px: Optional[int] = None,
    check_resolution: int = 1024,
    max_rounds: int = 3,
    verbose: bool = False,
    max_atomized_fraction: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Unwrap a triangle mesh.

    Returns ``(new_vertices [M,3], new_faces [F,3], uvs [M,2], vmap [M])`` where ``vmap`` maps every new
    vertex to the original vertex it duplicates (charts do not share vertices).
    """
    t0 = time.time()
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if f.ndim != 2 or f.shape[1] != 3 or len(f) == 0:
        raise ValueError("faces must be a non-empty [F,3] array")
    nf = len(f)
    if nf > 1_500_000:
        log(f"WARNING: unwrapping {nf:,} faces with the built-in unwrapper is slow; ComfyUI's 'Unwrap Mesh UVs' node suits dense meshes better.")
    normals, _areas = _face_geometry(v, f)
    a, b = _edge_adjacency(f)
    gutter = gutter_px if gutter_px is not None else max(2, min(16, round(texture_size / 256)))
    pad = gutter / float(texture_size)

    # --- initial charts: nearest projection direction + edge connectivity ----------------------
    set_idx = _pick_direction_set(cone_half_angle_deg)
    dirs = _directions(_DIR_COUNTS[set_idx])[0]
    label = _smooth_labels(np.argmax(normals @ dirs.T, axis=1), normals, dirs, a, b)
    chart = _components(nf, a, b, label)
    axis_of_face = dirs[label]

    for rnd in range(max_rounds + 1):
        uv_faces, scale = _build_layout(v, f, chart, axis_of_face, pad)
        bad = _overlapping_charts(uv_faces, chart, check_resolution)
        if verbose:
            print(f"[uv] round {rnd}: {int(chart.max()) + 1} charts, scale={scale:.4f}, overlapping charts={int(bad.sum())}")
        if not bad.any():
            break
        bad_faces = bad[chart]
        if rnd < max_rounds:
            # split the offending charts with a finer direction set (connected components of equal direction)
            set_idx = min(set_idx + 1, len(_DIR_COUNTS) - 1)
            dirs = _directions(_DIR_COUNTS[set_idx])[0]
            new_label = np.argmax(normals @ dirs.T, axis=1)
            axis_of_face = np.where(bad_faces[:, None], dirs[new_label], axis_of_face)
            comp_label = np.where(bad_faces, new_label + 1, 0)
            keep = (chart[a] == chart[b]) & (comp_label[a] == comp_label[b])
            g = coo_matrix((np.ones(int(keep.sum()), dtype=np.int8), (a[keep], b[keep])), shape=(nf, nf))
            chart = connected_components(g, directed=False)[1]
        else:
            # last resort: one chart per face for the still-overlapping charts (cannot overlap by construction)
            atomized = int(bad_faces.sum())
            if atomized > max_atomized_fraction * nf:
                raise ValueError(
                    f"The built-in unwrapper cannot flatten this mesh without shredding it: {atomized:,} of {nf:,} faces ({atomized / nf:.0%}) "
                    "lie in charts that still overlap after splitting. Use ComfyUI's 'Unwrap Mesh UVs' node (or another UV node) for dense or "
                    "organic meshes and connect that mesh to the STM nodes - they use existing UVs as they are.")
            chart = chart.copy()
            chart[bad_faces] = int(chart.max()) + 1 + np.arange(atomized)
            chart = np.unique(chart, return_inverse=True)[1].reshape(-1)       # compact labels: the replaced charts no longer exist
            axis_of_face = axis_of_face.copy()
            axis_of_face[bad_faces] = normals[bad_faces]
            uv_faces, scale = _build_layout(v, f, chart, axis_of_face, pad)
            if verbose:
                print(f"[uv] fallback: {int(bad_faces.sum())} faces became single-face charts")

    # a layout whose charts shrink to a few texels is useless: say so instead of returning it
    d1, d2 = uv_faces[:, 1] - uv_faces[:, 0], uv_faces[:, 2] - uv_faces[:, 0]
    covered = 0.5 * float(np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]).sum())
    n_charts = int(chart.max()) + 1
    if covered < 0.05:
        raise ValueError(
            f"The built-in unwrapper produced {n_charts:,} charts that cover only {covered:.1%} of the atlas - too fragmented to "
            f"texture. Use ComfyUI's 'Unwrap Mesh UVs' node for dense or organic meshes and connect that mesh to the STM nodes.")
    log(f"UV unwrap: {nf:,} faces -> {n_charts:,} charts, {covered:.0%} of the atlas used")

    # --- build the output mesh (vertices duplicated per chart) ----------------------------------
    corner_vert = f.reshape(-1)
    corner_chart = np.repeat(chart, 3)
    key = corner_chart.astype(np.int64) * (int(f.max()) + 1) + corner_vert
    uniq, first, inv = np.unique(key, return_index=True, return_inverse=True)
    new_faces = inv.reshape(-1, 3).astype(np.int64)
    vmap = corner_vert[first]
    new_vertices = v[vmap]
    uvs = np.zeros((len(uniq), 2))
    uvs[inv] = uv_faces.reshape(-1, 2)
    if verbose:
        print(f"[uv] unwrap: {nf} faces -> {int(chart.max()) + 1} charts, {len(uniq)} vertices in {time.time() - t0:.2f}s")
    return (new_vertices.astype(np.float32), new_faces, np.clip(uvs, 0.0, 1.0).astype(np.float32), vmap.astype(np.int64))

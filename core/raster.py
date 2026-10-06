"""Pure-PyTorch UV-space rasterization: which triangle covers each texel centre (no depth test).

* triangles are binned by the size of their texel bounding box (2 ... 64 texels); each bin is evaluated as a dense candidate
  grid whose size is bounded by ``max_candidates``; larger triangles are split into 64x64 blocks;
* overlaps are resolved deterministically (lowest triangle id wins) with ``scatter_reduce``;
* texel centres: row ``r`` / column ``c`` has its centre at ``((c + 0.5) / W, (r + 0.5) / H)``.

``nearest_valid_texel`` (jump flooding, for GPUs) and ``nearest_valid_texel_cpu`` (exact distance transform) give every empty
texel within ``max_dist`` pixels of an island the index of its nearest covered texel.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["rasterize_uv_coverage", "nearest_valid_texel", "nearest_valid_texel_cpu", "barycentric"]

_INT_MAX = 2 ** 31 - 1
_SIZE_CLASSES = (2, 4, 8, 16, 32, 64)


def barycentric(tri_uv: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    """Barycentric coordinates of ``pts`` w.r.t. triangles, valid *outside* the triangle too.

    tri_uv: [N,3,2] triangle corners (texel units), pts: [N,2] -> [N,3] (b0,b1,b2) summing to 1.
    Degenerate triangles return (1,0,0).
    """
    p0 = tri_uv[:, 0]
    a = tri_uv[:, 1] - p0
    b = tri_uv[:, 2] - p0
    d = pts - p0
    area2 = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    safe = torch.where(area2.abs() > 1e-20, area2, torch.ones_like(area2))
    s = (d[:, 0] * b[:, 1] - d[:, 1] * b[:, 0]) / safe
    t = (a[:, 0] * d[:, 1] - a[:, 1] * d[:, 0]) / safe
    bad = area2.abs() <= 1e-20
    s = torch.where(bad, torch.zeros_like(s), s)
    t = torch.where(bad, torch.zeros_like(t), t)
    return torch.stack([1.0 - s - t, s, t], dim=1)


def _raster_blocks(buf, W, tri, c0, r0, c1, r1, p0, a, b, area2, S, eps):
    """Evaluate S x S candidate blocks and min-scatter covered texels into ``buf``."""
    dev = buf.device
    g = torch.arange(S, device=dev, dtype=torch.float32)
    cc = c0[:, None, None] + g[None, None, :]
    rr = r0[:, None, None] + g[None, :, None]
    in_box = (cc <= c1[:, None, None]) & (rr <= r1[:, None, None])
    dx = cc + 0.5 - p0[:, 0, None, None]
    dy = rr + 0.5 - p0[:, 1, None, None]
    inv = 1.0 / area2[:, None, None]
    s = (dx * b[:, 1, None, None] - dy * b[:, 0, None, None]) * inv
    t = (a[:, 0, None, None] * dy - a[:, 1, None, None] * dx) * inv
    inside = in_box & (s >= -eps) & (t >= -eps) & (s + t <= 1.0 + eps)
    if not bool(inside.any()):
        return
    # int64 on purpose: float32 cannot represent texel indices above 2**24 (an 8192x8192 atlas)
    flat = (rr.to(torch.int64) * W + cc.to(torch.int64))[inside]
    val = tri[:, None, None].expand_as(inside)[inside].to(torch.int32)
    buf.scatter_reduce_(0, flat, val, reduce="amin", include_self=True)


@torch.no_grad()
def rasterize_uv_coverage(
    uv_px: torch.Tensor,
    faces: torch.Tensor,
    height: int,
    width: int,
    max_candidates: int = 2_000_000,
    eps: float = 1e-5,
    face_chunk: int = 1 << 21,
) -> torch.Tensor:
    """Return an int32 ``[H, W]`` map of the triangle covering each texel centre (-1 = empty).

    Args:
        uv_px: [V, 2] float UVs in *texel units* (x = u * W, y = v * H).
        faces: [F, 3] integer vertex indices.
        max_candidates: upper bound on candidate texels evaluated at once (memory knob).
        face_chunk: triangles processed per pass (bounds the per-triangle setup tensors for multi-million-face meshes).
    """
    dev = uv_px.device
    uv_px = uv_px.to(torch.float32)
    faces = faces.to(torch.int64)
    n_faces = faces.shape[0]
    buf = torch.full((height * width,), _INT_MAX, dtype=torch.int32, device=dev)
    if n_faces == 0:
        return torch.full((height, width), -1, dtype=torch.int32, device=dev)
    for f0 in range(0, n_faces, face_chunk):
        _rasterize_chunk(buf, uv_px, faces[f0:f0 + face_chunk], f0, height, width, max_candidates, eps)
    buf = torch.where(buf == _INT_MAX, torch.full_like(buf, -1), buf)
    return buf.view(height, width)


def _rasterize_chunk(buf, uv_px, faces, id_offset, height, width, max_candidates, eps):
    dev = uv_px.device
    p = uv_px[faces]                                  # [F,3,2]
    p0 = p[:, 0]
    a = p[:, 1] - p0
    b = p[:, 2] - p0
    area2 = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    lo = p.amin(dim=1)
    hi = p.amax(dim=1)
    c0 = torch.ceil(lo[:, 0] - 0.5 - eps).clamp_(min=0)
    c1 = torch.floor(hi[:, 0] - 0.5 + eps).clamp_(max=width - 1)
    r0 = torch.ceil(lo[:, 1] - 0.5 - eps).clamp_(min=0)
    r1 = torch.floor(hi[:, 1] - 0.5 + eps).clamp_(max=height - 1)
    nw = c1 - c0 + 1
    nh = r1 - r0 + 1
    ok = (area2.abs() > 1e-12) & (nw > 0) & (nh > 0) & torch.isfinite(area2)
    ids = torch.nonzero(ok, as_tuple=False).squeeze(1)
    if ids.numel() == 0:
        return
    gid = ids + id_offset                                # global triangle ids written into the buffer
    size = torch.maximum(nw, nh)[ids]
    prev = 0
    for S in _SIZE_CLASSES:
        sel = (size > prev) & (size <= S)
        prev = S
        if not bool(sel.any()):
            continue
        t_loc = ids[sel]
        t_glob = gid[sel]
        step = max(1, max_candidates // (S * S))
        for k in range(0, t_loc.numel(), step):
            t = t_loc[k:k + step]
            _raster_blocks(buf, width, t_glob[k:k + step], c0[t], r0[t], c1[t], r1[t], p0[t], a[t], b[t], area2[t], S, eps)

    sel = size > _SIZE_CLASSES[-1]
    if bool(sel.any()):
        big = ids[sel]
        big_glob = gid[sel]
        S = _SIZE_CLASSES[-1]
        nbx = torch.ceil(nw[big] / S).to(torch.int64)
        nby = torch.ceil(nh[big] / S).to(torch.int64)
        per = nbx * nby
        owner = torch.repeat_interleave(torch.arange(big.numel(), device=dev), per)
        local = torch.arange(int(per.sum()), device=dev) - torch.repeat_interleave(torch.cumsum(per, 0) - per, per)
        nbx_o = torch.repeat_interleave(nbx, per)
        bx = (local % nbx_o).to(torch.float32)
        by = (local // nbx_o).to(torch.float32)
        t_loc = big[owner]
        bc0 = c0[t_loc] + bx * S
        br0 = r0[t_loc] + by * S
        step = max(1, max_candidates // (S * S))
        for k in range(0, owner.numel(), step):
            sl = slice(k, k + step)
            t = t_loc[sl]
            _raster_blocks(buf, width, big_glob[owner[sl]], bc0[sl], br0[sl], c1[t], r1[t], p0[t], a[t], b[t], area2[t], S, eps)


@torch.no_grad()
def nearest_valid_texel(valid: torch.Tensor, max_dist: int) -> torch.Tensor:
    """Jump-flooding propagation of the nearest valid texel.

    Returns an int32 ``[H, W]`` tensor holding, for every texel, the flat index
    (``row * W + col``) of the nearest valid texel within ``max_dist`` pixels, or -1.
    Valid texels map to themselves.  The result is the usual JFA(+1) approximation of the
    exact Euclidean nearest neighbour (errors only affect which of two nearly equidistant
    islands claims a gutter texel).
    """
    h, w = valid.shape
    dev = valid.device
    idx = torch.arange(h * w, device=dev, dtype=torch.int32).view(h, w)
    neg = torch.full_like(idx, -1)
    src = torch.where(valid, idx, neg)
    if max_dist <= 0 or not bool(valid.any()):
        return src
    rows = idx // w
    cols = idx - rows * w
    big = torch.full((h, w), _INT_MAX, dtype=torch.int32, device=dev)
    best = torch.where(valid, torch.zeros_like(big), big)

    step = 1
    while step * 2 <= max_dist:
        step *= 2
    steps = []
    s = step
    while s >= 1:
        steps.append(s)
        s //= 2
    steps.append(1)

    for s in steps:
        pad = F.pad(src, (s, s, s, s), value=-1)
        for dy in (-s, 0, s):
            for dx in (-s, 0, s):
                if dx == 0 and dy == 0:
                    continue
                cand = pad[s + dy: s + dy + h, s + dx: s + dx + w]
                has = cand >= 0
                cr = torch.div(cand, w, rounding_mode="floor")
                cc = cand - cr * w
                d2 = (cr - rows) ** 2 + (cc - cols) ** 2
                better = has & (d2 < best)
                src = torch.where(better, cand, src)
                best = torch.where(better, d2, best)
    return torch.where(best <= max_dist * max_dist, src, neg)


def nearest_valid_texel_cpu(valid: torch.Tensor, max_dist: int) -> torch.Tensor:
    """Exact Euclidean version of :func:`nearest_valid_texel` for CPU tensors (SciPy's linear-time EDT).

    On a CPU the jump-flooding passes cost ~48 full-image tensor sweeps (24 s at 4096^2 on one core); the
    SciPy transform is O(N) in C (about 2 s).  GPUs keep the jump-flooding version, which is a few elementwise
    kernels there.  Same contract: flat index of the nearest valid texel within ``max_dist`` px, else -1.
    """
    from scipy import ndimage
    v = valid.cpu().numpy().astype(bool)
    h, w = v.shape
    idx = np.arange(h * w, dtype=np.int32).reshape(h, w)
    if max_dist <= 0 or not v.any():
        return torch.from_numpy(np.where(v, idx, -1).astype(np.int32))
    dist, (iy, ix) = ndimage.distance_transform_edt(~v, return_indices=True)
    flat = (iy.astype(np.int64) * w + ix).astype(np.int32)
    flat[dist > max_dist] = -1
    del dist, iy, ix
    return torch.from_numpy(flat)

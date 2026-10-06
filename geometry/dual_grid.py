"""Flexible dual grid in pure PyTorch: mesh -> sparse voxels with one dual vertex each (the shape encoder's input).

Algorithm of TRELLIS.2's ``o-voxel`` (Microsoft, MIT License), expressed with tensor ops so it runs on CUDA, ROCm and CPU:

1. Hit enumeration: every triangle is scan-converted along the three axes; each time a lattice line (an edge of the voxel
   grid) pierces the triangle, the four voxels sharing that edge are activated and receive the triangle's plane quadric, the hit
   point and, for the voxel owning the edge, an ``intersected`` flag.
2. Face quadrics: every active voxel that overlaps a triangle (separating-axis test) also receives its quadric.
3. Boundary quadrics: open boundary edges add a weighted line quadric to the voxels they traverse (3-D DDA).
4. Dual vertex: the regularised quadric error is minimised per voxel; if the minimiser leaves the voxel, the best point on the
   voxel boundary (faces, edges, corners) is used.

Work is chunked (rows, columns, candidates) so memory is bounded; the voxel table is a sorted key array queried with
``torch.searchsorted``.  Scan arithmetic is float64, the SAT test float32.  ``regularization_weight`` must be > 0.
"""
from __future__ import annotations

from typing import Callable, Iterator, List, Optional, Sequence, Tuple

import torch

__all__ = ["mesh_to_flexible_dual_grid"]

_I = (0, 0, 0, 0, 1, 1, 1, 2, 2, 3)      # upper-triangular packing of a symmetric 4x4
_J = (0, 1, 2, 3, 1, 2, 3, 2, 3, 3)


# --------------------------------------------------------------------------------------- helpers
def _plane_quadric(tri: torch.Tensor):
    """tri: [T,3,3] float32 -> (normal [T,3], Q10 [T,10]) with the reference's float32 arithmetic."""
    v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
    e0 = v1 - v0
    e1 = v2 - v1
    c = torch.cross(e0, e1, dim=1)
    z = c[:, 0] * c[:, 0] + c[:, 1] * c[:, 1] + c[:, 2] * c[:, 2]
    zs = z.unsqueeze(1)
    n = torch.where(zs > 0, c / torch.sqrt(zs.clamp(min=1e-38)), c)
    d = -(n[:, 0] * v0[:, 0] + n[:, 1] * v0[:, 1] + n[:, 2] * v0[:, 2])
    p = torch.cat([n, d.unsqueeze(1)], dim=1)
    q10 = p[:, list(_I)] * p[:, list(_J)]
    return n, q10


def _chunk_bounds(weights: torch.Tensor, limit: int) -> List[Tuple[int, int]]:
    """Split ``range(len(weights))`` into consecutive chunks whose weight sum is <= limit (min 1 item)."""
    n = int(weights.numel())
    out: List[Tuple[int, int]] = []
    if n == 0:
        return out
    csum = torch.cumsum(weights.to(torch.int64), 0).cpu()
    start = 0
    base = 0
    while start < n:
        end = int(torch.searchsorted(csum, torch.tensor(base + limit), right=True))
        end = max(end, start + 1)
        end = min(end, n)
        out.append((start, end))
        base = int(csum[end - 1])
        start = end
    return out


def _expand(counts: torch.Tensor):
    """For per-item counts return (item_id [sum], offset_in_item [sum])."""
    total = int(counts.sum())
    dev = counts.device
    ids = torch.repeat_interleave(torch.arange(counts.numel(), device=dev), counts)
    starts = torch.cumsum(counts, 0) - counts
    offs = torch.arange(total, device=dev) - torch.repeat_interleave(starts, counts)
    return ids, offs


def _lerp_xz(a: torch.Tensor, b: torch.Tensor, y: torch.Tensor):
    """Reference ``lerp(a.y, b.y, y, (a.x,a.z), (b.x,b.z))`` in float64 -> [R,2]."""
    ay, by = a[:, 1], b[:, 1]
    same = ay == by
    alpha = (y - ay) / torch.where(same, torch.ones_like(by), by - ay)
    va = torch.stack([a[:, 0], a[:, 2]], 1)
    vb = torch.stack([b[:, 0], b[:, 2]], 1)
    val = (1.0 - alpha).unsqueeze(1) * va + alpha.unsqueeze(1) * vb
    return torch.where(same.unsqueeze(1), va, val)


# ------------------------------------------------------------------------- 1. scan-line hit pass
def _iter_hits(tri32: torch.Tensor, ax2: int, vs32: torch.Tensor, gmin: Sequence[int], gmax: Sequence[int],
               max_rows: int, max_cols: int) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Yield (tri_id, coord [H,3] int64, point [H,3] float32) for every lattice-line / triangle hit.

    ``coord`` is the voxel that owns the pierced edge (dx = dy = 0); the other three voxels are
    obtained by adding 1 along the two axes orthogonal to ``ax2``.
    """
    dev = tri32.device
    ax0, ax1 = (ax2 + 1) % 3, (ax2 + 2) % 3
    P = tri32[:, :, [ax0, ax1, ax2]].to(torch.float64)
    order = torch.argsort(P[:, :, 1], dim=1)
    P = torch.gather(P, 1, order.unsqueeze(-1).expand(-1, -1, 3))
    v0d, v1d, v2d = (float(vs32[a]) for a in (ax0, ax1, ax2))
    vs0f, vs1f = vs32[ax0].to(dev), vs32[ax1].to(dev)

    def clamp_idx(t, lo, hi):
        return torch.clamp(torch.trunc(t).to(torch.int64), lo, hi)

    start = clamp_idx(P[:, 0, 1] / v1d, gmin[ax1], gmax[ax1] - 1)
    mid = clamp_idx(P[:, 1, 1] / v1d, gmin[ax1], gmax[ax1] - 1)
    end = clamp_idx(P[:, 2, 1] / v1d, gmin[ax1], gmax[ax1] - 1)
    n_rows = (end - start).clamp(min=0)
    for t0, t1 in _chunk_bounds(n_rows, max_rows):
        nr = n_rows[t0:t1]
        if int(nr.sum()) == 0:
            continue
        tid_local, offs = _expand(nr)
        tid = tid_local + t0
        y_idx = start[tid] + offs
        first = (y_idx < mid[tid]).unsqueeze(1)
        Pt = P[tid]
        A = torch.where(first, Pt[:, 0], Pt[:, 2])
        B = Pt[:, 1]
        C = torch.where(first, Pt[:, 2], Pt[:, 0])
        y = ((y_idx + 1).to(torch.float32) * vs1f).to(torch.float64)     # float32 product, as in the reference
        t3 = _lerp_xz(A, B, y)
        t4 = _lerp_xz(A, C, y)
        swap = (t3[:, 0] > t4[:, 0]).unsqueeze(1)
        lo = torch.where(swap, t4, t3)
        hi = torch.where(swap, t3, t4)
        ls = clamp_idx(lo[:, 0] / v0d, gmin[ax0], gmax[ax0] - 1)
        le = clamp_idx(hi[:, 0] / v0d, gmin[ax0], gmax[ax0] - 1)
        n_cols = (le - ls).clamp(min=0)
        for r0, r1 in _chunk_bounds(n_cols, max_cols):
            nc = n_cols[r0:r1]
            if int(nc.sum()) == 0:
                continue
            row_local, coff = _expand(nc)
            row = row_local + r0
            x_idx = ls[row] + coff
            x = ((x_idx + 1).to(torch.float32) * vs0f).to(torch.float64)
            lx, hx = lo[row, 0], hi[row, 0]
            same = lx == hx
            alpha = (x - lx) / torch.where(same, torch.ones_like(hx), hx - lx)
            z = torch.where(same, lo[row, 1], (1.0 - alpha) * lo[row, 1] + alpha * hi[row, 1])
            z_idx = torch.trunc(z / v2d).to(torch.int64)
            ok = (z_idx >= gmin[ax2]) & (z_idx < gmax[ax2])
            if not bool(ok.any()):
                continue
            x_idx, z_idx, x, z, row = x_idx[ok], z_idx[ok], x[ok], z[ok], row[ok]
            h = x_idx.numel()
            coord = torch.empty((h, 3), dtype=torch.int64, device=dev)
            coord[:, ax0] = x_idx
            coord[:, ax1] = y_idx[row]
            coord[:, ax2] = z_idx
            pt = torch.empty((h, 3), dtype=torch.float32, device=dev)
            pt[:, ax0] = x.to(torch.float32)
            pt[:, ax1] = y[row].to(torch.float32)
            pt[:, ax2] = z.to(torch.float32)
            yield tid[row], coord, pt


def _voxel_key(coord: torch.Tensor, g: Sequence[int]) -> torch.Tensor:
    return (coord[:, 0] * g[1] + coord[:, 1]) * g[2] + coord[:, 2]


def _lookup(sorted_keys: torch.Tensor, keys: torch.Tensor):
    """Return (row index, found mask) of ``keys`` in the sorted voxel table."""
    n = sorted_keys.numel()
    pos = torch.searchsorted(sorted_keys, keys).clamp_(max=max(n - 1, 0))
    return pos, sorted_keys[pos] == keys


# ----------------------------------------------------------------------------------- 2. face QEF
def _face_qef(tri32, normal, q10, vs32, gmin, gmax, sorted_keys, g, acc, max_cand):
    """Add each triangle's quadric to every *existing* voxel it overlaps (separating-axis test).

    Candidates are enumerated per (u, v) column of the triangle's bounding box along its dominant
    normal axis: only the few voxels whose slab the triangle plane crosses are generated, so cost is
    proportional to the triangle's projected area (not to its bounding-box volume).
    """
    dev = tri32.device
    T = tri32.shape[0]
    vs = vs32.to(dev)
    v0, v1, v2 = tri32[:, 0], tri32[:, 1], tri32[:, 2]
    e0, e1, e2 = v1 - v0, v2 - v1, v0 - v2
    n = normal
    nonzero = (n != 0).any(dim=1)

    def f2i(x):
        return torch.trunc(x).to(torch.int64)

    gmin_t = torch.tensor(gmin, dtype=torch.int64, device=dev)
    gmax_t = torch.tensor(gmax, dtype=torch.int64, device=dev)
    bb_min = torch.maximum(f2i(tri32.amin(dim=1) / vs), gmin_t)
    bb_max = torch.minimum(f2i(tri32.amax(dim=1) / vs + 1.0), gmax_t)
    ext = (bb_max - bb_min).clamp(min=0)

    c = torch.where(n > 0, vs.expand_as(n), torch.zeros_like(n))
    d1 = n[:, 0] * (c[:, 0] - v0[:, 0]) + n[:, 1] * (c[:, 1] - v0[:, 1]) + n[:, 2] * (c[:, 2] - v0[:, 2])
    rem = vs.expand_as(n) - c - v0
    d2 = n[:, 0] * rem[:, 0] + n[:, 1] * rem[:, 1] + n[:, 2] * rem[:, 2]

    one = torch.ones_like(n[:, 0])
    muls = (torch.where(n[:, 2] < 0, -one, one), torch.where(n[:, 0] < 0, -one, one), torch.where(n[:, 1] < 0, -one, one))
    proj = ((0, 1), (1, 2), (2, 0))                          # xy, yz, zx projections
    cols = []                                                # per triangle: 3 projections x 3 edges x (na, nb, dd)
    for (ai, bi), mul in zip(proj, muls):
        for e, v in ((e0, v0), (e1, v1), (e2, v2)):
            na = -mul * e[:, bi]
            nb = mul * e[:, ai]
            dd = -(na * v[:, ai] + nb * v[:, bi]) + (na.clamp(min=0) * vs[ai] + nb.clamp(min=0) * vs[bi])
            cols += [na, nb, dd]
    cst = torch.stack(cols, dim=1)                           # [T, 27]
    base = torch.stack([n[:, 0], n[:, 1], n[:, 2], d1, d2], dim=1)  # [T, 5]

    a_ax = torch.argmax(n.abs(), dim=1)
    ua, va = (a_ax + 1) % 3, (a_ax + 2) % 3
    ar = torch.arange(T, device=dev)
    ext_u, ext_v, ext_a = ext[ar, ua], ext[ar, va], ext[ar, a_ax]
    live = nonzero & (ext_u > 0) & (ext_v > 0) & (ext_a > 0)
    if not bool(live.any()):
        return
    vs_d = vs.double()
    n_d = n.double()
    n_a = n_d[ar, a_ax]
    den = n_a * vs_d[a_ax]
    dmin = torch.minimum(d1.double(), d2.double())
    dmax = torch.maximum(d1.double(), d2.double())
    cu_scale = n_d[ar, ua] * vs_d[ua]
    cv_scale = n_d[ar, va] * vs_d[va]
    tol = 1e-3                                               # voxel units; covers float32 rounding of the exact SAT test

    items = torch.where(live, ext_u, torch.zeros_like(ext_u))        # one item = one u-slice of the column grid
    for t0, t1 in _chunk_bounds(items * ext_v.clamp(min=1), max_cand // 4):
        it = items[t0:t1]
        if int(it.sum()) == 0:
            continue
        sub, uoff = _expand(it)
        tid = sub + t0
        for i0, i1 in _chunk_bounds(ext_v[tid], max_cand // 4):
            ti, uo = tid[i0:i1], uoff[i0:i1]
            row_i, vo = _expand(ext_v[ti])
            tt = ti[row_i]
            cu = bb_min[tt, ua[tt]] + uo[row_i]
            cv = bb_min[tt, va[tt]] + vo
            at = a_ax[tt]
            tcol = cu_scale[tt] * cu.double() + cv_scale[tt] * cv.double()
            w1 = (-dmax[tt] - tcol) / den[tt]
            w2 = (-dmin[tt] - tcol) / den[tt]
            wlo = torch.ceil(torch.minimum(w1, w2) - tol).to(torch.int64)
            whi = torch.floor(torch.maximum(w1, w2) + tol).to(torch.int64)
            lo_a, hi_a = bb_min[tt, at], bb_max[tt, at]
            wlo = torch.maximum(wlo, lo_a)
            whi = torch.minimum(whi, hi_a - 1)
            nw = (whi - wlo + 1).clamp(min=0)
            if int(nw.sum()) == 0:
                continue
            col, koff = _expand(nw)
            t_k = tt[col]
            xyz = torch.empty((t_k.numel(), 3), dtype=torch.int64, device=dev)
            xyz.scatter_(1, ua[t_k].unsqueeze(1), cu[col].unsqueeze(1))
            xyz.scatter_(1, va[t_k].unsqueeze(1), cv[col].unsqueeze(1))
            xyz.scatter_(1, at[col].unsqueeze(1), (wlo[col] + koff).unsqueeze(1))
            p = vs * xyz.to(torch.float32)
            bk = base[t_k]
            ndp = bk[:, 0] * p[:, 0] + bk[:, 1] * p[:, 1] + bk[:, 2] * p[:, 2]
            keep = ~(((ndp + bk[:, 3]) * (ndp + bk[:, 4])) > 0.0)
            ck = cst[t_k]
            for j, (ai, bi) in enumerate(proj):
                pa, pb = p[:, ai], p[:, bi]
                for k in range(3):
                    o = (j * 3 + k) * 3
                    keep &= ~((ck[:, o] * pa + ck[:, o + 1] * pb + ck[:, o + 2]) < 0)
            if not bool(keep.any()):
                continue
            xyz, t_k = xyz[keep], t_k[keep]
            pos, found = _lookup(sorted_keys, _voxel_key(xyz, g))
            if bool(found.any()):
                acc.index_add_(0, pos[found], q10[t_k[found]])


# -------------------------------------------------------------------------------- 3. boundary QEF
def _boundary_qef(vertices, faces, vs32, gmin, gmax, weight, sorted_keys, g, acc, max_seg=1 << 20):
    dev = vertices.device
    f = faces.to(torch.int64)
    e = torch.cat([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], dim=0)
    e, _ = torch.sort(e, dim=1)
    nv = int(vertices.shape[0])
    key = e[:, 0] * nv + e[:, 1]
    uniq, counts = torch.unique(key, return_counts=True)
    bk = uniq[counts == 1]
    if bk.numel() == 0:
        return
    ia, ib = bk // nv, bk % nv
    for s in range(0, bk.numel(), max_seg):
        p0 = vertices[ia[s:s + max_seg]].to(dev)
        p1 = vertices[ib[s:s + max_seg]].to(dev)
        dirf = p1 - p0                                                   # float32 subtraction, as in the reference
        dir_d = dirf.to(torch.float64)
        seg_len = torch.linalg.norm(dir_d, dim=1)
        ok = seg_len >= 1e-6
        if not bool(ok.any()):
            continue
        p0, p1, dir_d, seg_len = p0[ok], p1[ok], dir_d[ok], seg_len[ok]
        dir_d = dir_d / seg_len.unsqueeze(1)
        A = torch.eye(3, dtype=torch.float64, device=dev).unsqueeze(0) - dir_d.unsqueeze(2) * dir_d.unsqueeze(1)
        Af = A.to(torch.float32)
        b = -(Af @ p0.unsqueeze(2)).squeeze(2)
        c = (p0.unsqueeze(1) @ Af @ p0.unsqueeze(2)).reshape(-1)
        q10 = torch.stack([Af[:, 0, 0], Af[:, 0, 1], Af[:, 0, 2], b[:, 0], Af[:, 1, 1], Af[:, 1, 2], b[:, 1],
                           Af[:, 2, 2], b[:, 2], c], dim=1)
        vs = vs32.to(dev)
        cur = torch.floor(p0 / vs).to(torch.int64)
        step = torch.where(dir_d > 0, 1, -1).to(torch.int64)
        border = vs * (cur + (step > 0).to(torch.int64)).to(torch.float32)
        inf = torch.full_like(dir_d, float("inf"))
        zero_dir = dir_d == 0.0
        safe = torch.where(zero_dir, torch.ones_like(dir_d), dir_d)
        t_max = torch.where(zero_dir, inf, (border - p0).to(torch.float64) / safe)
        t_delta = torch.where(zero_dir, inf, vs.to(torch.float64) / safe.abs())
        seg_ids = torch.arange(cur.shape[0], device=dev)
        visited_coord = [cur.clone()]
        visited_seg = [seg_ids]
        act = seg_ids
        while act.numel() > 0:
            tm = t_max[act]
            axis = torch.where(tm[:, 0] < tm[:, 1],
                               torch.where(tm[:, 0] < tm[:, 2], 0, 2),
                               torch.where(tm[:, 1] < tm[:, 2], 1, 2))
            cond = tm.gather(1, axis.unsqueeze(1)).squeeze(1) <= seg_len[act]
            act, axis = act[cond], axis[cond]
            if act.numel() == 0:
                break
            cur[act, axis] += step[act, axis]
            t_max[act, axis] += t_delta[act, axis]
            visited_coord.append(cur[act].clone())
            visited_seg.append(act)
        coord = torch.cat(visited_coord, 0)
        seg = torch.cat(visited_seg, 0)
        inb = ((coord >= torch.tensor(gmin, device=dev)) & (coord < torch.tensor(gmax, device=dev))).all(1)
        coord, seg = coord[inb], seg[inb]
        pos, found = _lookup(sorted_keys, _voxel_key(coord, g))
        if bool(found.any()):
            acc.index_add_(0, pos[found], weight * q10[seg[found]])


# ------------------------------------------------------------------------------------- 4. solve
def _q_from_q10(q10: torch.Tensor) -> torch.Tensor:
    n = q10.shape[0]
    Q = torch.zeros((n, 4, 4), dtype=torch.float64, device=q10.device)
    for k, (i, j) in enumerate(zip(_I, _J)):
        Q[:, i, j] = q10[:, k].double()
        Q[:, j, i] = q10[:, k].double()
    return Q


def _solve_cells(Q: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """Minimise p^T Q p over each cell [lo,hi] (float64). Q:[n,4,4], lo/hi:[n,3] -> [n,3]."""
    n = Q.shape[0]
    dev = Q.device
    A = Q[:, :3, :3]
    b = -Q[:, :3, 3]
    sol = torch.linalg.solve_ex(A, b.unsqueeze(2))
    v = sol.result.squeeze(2)
    bad_sing = sol.info != 0
    if bool(bad_sing.any()):
        v[bad_sing] = torch.linalg.lstsq(A[bad_sing], b[bad_sing].unsqueeze(2)).solution.squeeze(2)
    inside = ((v >= lo) & (v <= hi)).all(dim=1) & torch.isfinite(v).all(dim=1)
    out = v.clone()
    idx = torch.nonzero(~inside, as_tuple=False).squeeze(1)
    if idx.numel() == 0:
        return out
    Qs, lo_s, hi_s = Q[idx], lo[idx], hi[idx]
    m = idx.numel()
    best = torch.full((m,), float("inf"), dtype=torch.float64, device=dev)
    best_p = torch.zeros((m, 3), dtype=torch.float64, device=dev)

    def consider(p3, valid):
        nonlocal best, best_p
        p4 = torch.cat([p3, torch.ones((m, 1), dtype=torch.float64, device=dev)], dim=1)
        err = torch.einsum("ni,nij,nj->n", p4, Qs, p4)
        better = valid & (err < best)
        best = torch.where(better, err, best)
        best_p = torch.where(better.unsqueeze(1), p3, best_p)

    def solve2(Am, rhs):
        r = torch.linalg.solve_ex(Am, rhs.unsqueeze(2))
        x = r.result.squeeze(2)
        sing = r.info != 0
        if bool(sing.any()):
            x[sing] = torch.linalg.lstsq(Am[sing], rhs[sing].unsqueeze(2)).solution.squeeze(2)
        return x

    for fixed in range(3):                                   # face-constrained (fix one axis)
        a1, a2 = (fixed + 1) % 3, (fixed + 2) % 3
        Am = torch.stack([torch.stack([Qs[:, a1, a1], Qs[:, a1, a2]], 1),
                          torch.stack([Qs[:, a2, a1], Qs[:, a2, a2]], 1)], 1)
        Bm = torch.stack([torch.stack([Qs[:, a1, fixed], Qs[:, a1, 3]], 1),
                          torch.stack([Qs[:, a2, fixed], Qs[:, a2, 3]], 1)], 1)
        for corner in (lo_s, hi_s):
            q = torch.stack([corner[:, fixed], torch.ones_like(corner[:, fixed])], 1)
            x = solve2(Am, -(Bm @ q.unsqueeze(2)).squeeze(2))
            valid = (x[:, 0] >= lo_s[:, a1]) & (x[:, 0] <= hi_s[:, a1]) & (x[:, 1] >= lo_s[:, a2]) & (x[:, 1] <= hi_s[:, a2])
            p3 = torch.zeros((m, 3), dtype=torch.float64, device=dev)
            p3[:, fixed] = corner[:, fixed]
            p3[:, a1] = x[:, 0]
            p3[:, a2] = x[:, 1]
            consider(p3, valid)
    for free in range(3):                                    # edge-constrained (one free axis)
        a1, a2 = (free + 1) % 3, (free + 2) % 3
        aa = Qs[:, free, free]
        bb = torch.stack([Qs[:, free, a1], Qs[:, free, a2], Qs[:, free, 3]], 1)
        for c1, c2 in ((lo_s[:, a1], lo_s[:, a2]), (lo_s[:, a1], hi_s[:, a2]), (hi_s[:, a1], lo_s[:, a2]), (hi_s[:, a1], hi_s[:, a2])):
            q = torch.stack([c1, c2, torch.ones_like(c1)], 1)
            x = -(bb * q).sum(1) / aa
            valid = torch.isfinite(x) & (x >= lo_s[:, free]) & (x <= hi_s[:, free])
            p3 = torch.zeros((m, 3), dtype=torch.float64, device=dev)
            p3[:, free] = x
            p3[:, a1] = c1
            p3[:, a2] = c2
            consider(p3, valid)
    for xc in range(2):                                      # corners
        for yc in range(2):
            for zc in range(2):
                p3 = torch.stack([lo_s[:, 0] if xc else hi_s[:, 0], lo_s[:, 1] if yc else hi_s[:, 1],
                                  lo_s[:, 2] if zc else hi_s[:, 2]], 1)
                consider(p3, torch.ones(m, dtype=torch.bool, device=dev))
    out[idx] = best_p
    return out


# -------------------------------------------------------------------------------------- driver
def _as_vec(x, dtype):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return torch.full((3,), x, dtype=dtype)
    return torch.as_tensor(x, dtype=dtype).reshape(3)


@torch.no_grad()
def _accumulate(verts, faces, vs32, g, face_weight, boundary_weight, max_rows, max_cols, max_candidates, progress):
    """Passes 1-4: voxel set, hit statistics and quadrics.  Returns (keys, means, cnt, qacc, intersected).

    ``verts`` are relative to the grid origin; ``qacc`` holds the 10 unique entries of each voxel's 4x4 quadric
    *before* regularisation.
    """
    dev = verts.device
    gmin = (0, 0, 0)
    tri = verts[faces]
    good = torch.isfinite(tri).all(dim=(1, 2))
    if not bool(good.all()):                                 # drop NaN/inf triangles
        tri, faces = tri[good], faces[good]
    n_tri = tri.shape[0]
    tri_chunk = max(1, max_rows // 8)
    log = progress or (lambda s: None)

    # pass 1 - the voxel set (four voxels around every pierced lattice edge)
    key_parts: List[torch.Tensor] = []
    pending = 0
    for s in range(0, n_tri, tri_chunk):
        tc = tri[s:s + tri_chunk]
        for ax in range(3):
            a0, a1 = (ax + 1) % 3, (ax + 2) % 3
            for _, coord, _pt in _iter_hits(tc, ax, vs32, gmin, g, max_rows, max_cols):
                ks = []
                for dx in (0, 1):
                    for dy in (0, 1):
                        c = coord.clone()
                        c[:, a0] += dx
                        c[:, a1] += dy
                        ks.append(_voxel_key(c, g))
                part = torch.unique(torch.cat(ks))
                key_parts.append(part)
                pending += int(part.numel())
                if len(key_parts) > 16 and pending > (1 << 23):
                    key_parts = [torch.unique(torch.cat(key_parts))]
                    pending = int(key_parts[0].numel())
        log(f"dual grid: scanned {min(s + tri_chunk, n_tri)}/{n_tri} triangles")
    if not key_parts:
        raise ValueError("mesh does not intersect the voxel grid (check normalisation / aabb)")
    keys = torch.unique(torch.cat(key_parts))
    del key_parts
    n_vox = int(keys.numel())

    # pass 2 - hit counts, mean hit point, plane quadrics, edge-intersection flags
    means = torch.zeros((n_vox, 3), dtype=torch.float32, device=dev)
    cnt = torch.zeros((n_vox,), dtype=torch.float32, device=dev)
    qacc = torch.zeros((n_vox, 10), dtype=torch.float32, device=dev)
    inter = torch.zeros((n_vox, 3), dtype=torch.bool, device=dev)
    for s in range(0, n_tri, tri_chunk):
        tc = tri[s:s + tri_chunk]
        _n, q10_c = _plane_quadric(tc)
        for ax in range(3):
            a0, a1 = (ax + 1) % 3, (ax + 2) % 3
            for tid, coord, pt in _iter_hits(tc, ax, vs32, gmin, g, max_rows, max_cols):
                qh = q10_c[tid]
                for dx in (0, 1):
                    for dy in (0, 1):
                        c = coord.clone()
                        c[:, a0] += dx
                        c[:, a1] += dy
                        pos, _found = _lookup(keys, _voxel_key(c, g))
                        means.index_add_(0, pos, pt)
                        cnt.index_add_(0, pos, torch.ones_like(pos, dtype=torch.float32))
                        qacc.index_add_(0, pos, qh)
                        if dx == 0 and dy == 0:
                            inter[pos, ax] = True

    # pass 3 - quadrics of every triangle overlapping an existing voxel
    if face_weight > 0.0:
        for s in range(0, n_tri, tri_chunk):
            tc = tri[s:s + tri_chunk]
            nrm, q10_c = _plane_quadric(tc)
            _face_qef(tc, nrm, q10_c, vs32, gmin, g, keys, g, qacc, max_candidates)

    # pass 4 - open boundary edges
    if boundary_weight > 0.0:
        _boundary_qef(verts, faces, vs32, gmin, g, float(boundary_weight), keys, g, qacc)
    return keys, means, cnt, qacc, inter


@torch.no_grad()
def _solve_all(keys, means, cnt, qacc, vs32, g, regularization_weight):
    """Pass 5: regularised QEF minimisation per voxel.  Returns (coords int32 [N,3], dual vertices float32 [N,3])."""
    dev = keys.device
    n_vox = keys.numel()
    out_v = torch.empty((n_vox, 3), dtype=torch.float32, device=dev)
    coords = torch.empty((n_vox, 3), dtype=torch.int32, device=dev)
    step = 1 << 20
    for s in range(0, n_vox, step):
        k = keys[s:s + step]
        cc = torch.stack([k // (g[2] * g[1]), (k // g[2]) % g[1], k % g[2]], 1)
        coords[s:s + step] = cc.to(torch.int32)
        q10 = qacc[s:s + step].clone()
        if regularization_weight > 0.0:
            p = means[s:s + step] / cnt[s:s + step].unsqueeze(1)
            w = (regularization_weight * cnt[s:s + step]).to(torch.float32)
            q10[:, 0] += w
            q10[:, 4] += w
            q10[:, 7] += w
            q10[:, 3] += w * (-p[:, 0])
            q10[:, 6] += w * (-p[:, 1])
            q10[:, 8] += w * (-p[:, 2])
            q10[:, 9] += w * (p[:, 0] * p[:, 0] + p[:, 1] * p[:, 1] + p[:, 2] * p[:, 2])
        lo = cc.to(torch.float32) * vs32
        hi = (cc + 1).to(torch.float32) * vs32
        out_v[s:s + step] = _solve_cells(_q_from_q10(q10), lo.double(), hi.double()).to(torch.float32)
    return coords, out_v


@torch.no_grad()
def mesh_to_flexible_dual_grid(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    voxel_size=None,
    grid_size=None,
    aabb=None,
    face_weight: float = 1.0,
    boundary_weight: float = 1.0,
    regularization_weight: float = 0.1,
    device: Optional[torch.device] = None,
    max_rows: int = 1 << 21,
    max_cols: int = 1 << 22,
    max_candidates: int = 1 << 22,
    progress: Optional[Callable[[str], None]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mesh -> sparse voxels with one dual vertex each.

    Returns ``(voxel_indices int32 [N,3], dual_vertices float32 [N,3], intersected bool [N,3])`` on the CPU, with
    ``dual_vertices`` expressed relative to ``aabb[0]``.  Voxels are sorted by linear index.
    """
    dev = torch.device(device) if device is not None else vertices.device
    if dev.type in ("mps", "xpu"):
        dev = torch.device("cpu")                           # float64 is missing (MPS) or missing on many Intel GPUs (XPU)

    vertices = vertices.detach().to(torch.float32)
    faces = faces.detach().to(torch.int64)
    if vertices.dim() != 2 or vertices.shape[1] != 3 or faces.dim() != 2 or faces.shape[1] != 3:
        raise ValueError("vertices must be [V,3] and faces [F,3]")
    if faces.numel() == 0:
        raise ValueError("mesh has no faces")
    if int(faces.min()) < 0 or int(faces.max()) >= vertices.shape[0]:
        raise ValueError("face indices out of range")

    voxel_size_t = _as_vec(voxel_size, torch.float32)
    grid_size_t = _as_vec(grid_size, torch.int32)
    if voxel_size_t is None and grid_size_t is None:
        raise ValueError("Either voxel_size or grid_size must be provided")
    if aabb is None:
        mn, mx = vertices.amin(0), vertices.amax(0)
        pad = (torch.ceil((mx - mn) / voxel_size_t) * voxel_size_t - (mx - mn)) if voxel_size_t is not None else (mx - mn) / (grid_size_t.float() - 1)
        aabb_t = torch.stack([mn - pad * 0.5, mx + pad * 0.5])
    else:
        aabb_t = torch.as_tensor(aabb, dtype=torch.float32).reshape(2, 3)
    if voxel_size_t is None:
        voxel_size_t = (aabb_t[1] - aabb_t[0]) / grid_size_t.float()
    if grid_size_t is None:
        grid_size_t = ((aabb_t[1] - aabb_t[0]) / voxel_size_t).round().to(torch.int32)
    g = tuple(int(v) for v in grid_size_t.tolist())
    if g[0] * g[1] * g[2] >= 2 ** 62:
        raise ValueError("grid too large")

    verts = (vertices - aabb_t[0].reshape(1, 3)).to(dev)
    vs32 = voxel_size_t.to(torch.float32).to(dev)
    keys, means, cnt, qacc, inter = _accumulate(verts, faces.to(dev), vs32, g, face_weight, boundary_weight,
                                                max_rows, max_cols, max_candidates, progress)
    coords, dual = _solve_all(keys, means, cnt, qacc, vs32, g, regularization_weight)
    return coords.cpu(), dual.cpu(), inter.cpu()

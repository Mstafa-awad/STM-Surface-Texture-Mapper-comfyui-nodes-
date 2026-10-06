"""Orthographic multi-view rendering of a textured mesh with a pure-PyTorch z-buffer rasteriser.

Views are **unlit**: every pixel shows the texture colour and nothing else - no light, no shadow, no shading - so an image model
sees the surface itself, and lighting can be added later by the engine.

Views are **anti-aliased**: one texture lookup per pixel of a 4K texture on a 1K view skips texels and folds fine patterns into
moiré (rows of dots) that a refiner then sharpens into the texture.  Each view is therefore rasterised ``supersample`` times
larger (chosen from the texel density so that one lookup reads about one texel) and filtered down with a non-negative
antialiasing kernel (no ringing).
The high-resolution depth buffer is kept for an exact visibility test when the views are projected back.

Triangles are binned by screen-space size and evaluated as dense candidate blocks; depth and triangle id are packed into one
int64 key (positive float32 depth bits in the high word), so a single ``scatter_reduce(amin)`` keeps the nearest surface per
pixel - deterministic, no atomics on floats, no compiled extension.
"""
import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .raster import _SIZE_CLASSES, barycentric

_EMPTY = torch.iinfo(torch.int64).max

# (azimuth, elevation) in degrees; azimuth 0 looks at the front (+Z) of a glTF (Y-up) mesh
VIEW_SETS = {
    4: [(0, 0), (90, 0), (180, 0), (270, 0)],
    6: [(0, 0), (90, 0), (180, 0), (270, 0), (0, 90), (0, -90)],
    8: [(0, 0), (90, 0), (180, 0), (270, 0), (45, 30), (135, 30), (225, 30), (315, 30)],
    10: [(0, 0), (90, 0), (180, 0), (270, 0), (45, 30), (135, 30), (225, 30), (315, 30), (0, 90), (0, -90)],
    12: [(0, 0), (90, 0), (180, 0), (270, 0), (45, 35), (135, 35), (225, 35), (315, 35), (45, -35), (135, -35), (225, -35), (315, -35)],
}
VIEW_SET_HELP = ("4: front/right/back/left; 6: + top/bottom; 8: 4 + 4 raised diagonals; 10: 8 + top/bottom; "
                 "12: 4 + 4 raised + 4 lowered diagonals (best coverage of tops and undersides).")


@dataclass
class Camera:
    rotation: np.ndarray        # [3,3] rows: right, up, toward-camera (world -> camera)
    center: np.ndarray          # [3]
    scale: float                # pixels per world unit
    size: int                   # square image size
    near: float                 # depth = near - z_cam > 0 for the whole mesh

    @property
    def direction(self) -> np.ndarray:
        """Unit vector from the object towards the camera."""
        return self.rotation[2]

    def project(self, p: torch.Tensor, ss: int = 1) -> torch.Tensor:
        """World points [N,3] -> [N,3] (x px, y px, depth) on the view enlarged ``ss`` times; pixel centres at i + 0.5,
        y grows downwards."""
        R = torch.as_tensor(self.rotation, dtype=torch.float32, device=p.device)
        c = torch.as_tensor(self.center, dtype=torch.float32, device=p.device)
        q = (p.to(torch.float32) - c) @ R.T
        k, half = self.scale * ss, self.size * ss / 2
        return torch.stack([q[:, 0] * k + half, half - q[:, 1] * k, self.near - q[:, 2]], dim=1)


def camera_at(vertices: np.ndarray, size: int, azimuth: float, elevation: float = 0.0, margin: float = 0.04) -> Camera:
    """Orthographic camera looking at the mesh from (azimuth, elevation) degrees; azimuth 0 = the front (+Z) of a glTF mesh.
    The whole bounding sphere fits the square view (same framing for every direction)."""
    v = np.asarray(vertices, dtype=np.float64)
    center = (v.min(0) + v.max(0)) / 2
    radius = float(np.sqrt(((v - center) ** 2).sum(1)).max()) or 1.0
    scale = size / (2 * radius * (1 + margin))
    a, e = math.radians(azimuth), math.radians(elevation)
    z = np.array([math.sin(a) * math.cos(e), math.sin(e), math.cos(a) * math.cos(e)])
    up_world = np.array([0.0, 1.0, 0.0]) if abs(z[1]) < 0.99 else np.array([0.0, 0.0, -1.0 if z[1] > 0 else 1.0])
    right = np.cross(up_world, z)
    right /= np.linalg.norm(right)
    return Camera(np.stack([right, np.cross(z, right), z]), center, scale, int(size), radius * (1 + margin) + 1e-3)


def orbit_cameras(vertices: np.ndarray, size: int, views: int = 8, margin: float = 0.04) -> List[Camera]:
    if int(views) not in VIEW_SETS:
        raise ValueError(f"views must be one of {sorted(VIEW_SETS)}")
    return [camera_at(vertices, size, az, el, margin) for az, el in VIEW_SETS[int(views)]]


# ------------------------------------------------------------------------------------------------ z-buffer rasteriser
def _depth_blocks(buf, size, tri, c0, r0, c1, r1, p0, a, b, area2, z0, dz1, dz2, S, eps):
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
    z = (z0[:, None, None] + s * dz1[:, None, None] + t * dz2[:, None, None])[inside].clamp(min=1e-6).contiguous()
    key = (z.view(torch.int32).to(torch.int64) << 32) | tri[:, None, None].expand_as(inside)[inside].to(torch.int64)
    flat = (rr.to(torch.int64) * size + cc.to(torch.int64))[inside]
    buf.scatter_reduce_(0, flat, key, reduce="amin", include_self=True)


def _depth_chunk(buf, xyz, faces, id_offset, size, max_candidates, eps):
    dev = xyz.device
    p = xyz[faces]
    p0 = p[:, 0, :2]
    a, b = p[:, 1, :2] - p0, p[:, 2, :2] - p0
    z0, dz1, dz2 = p[:, 0, 2], p[:, 1, 2] - p[:, 0, 2], p[:, 2, 2] - p[:, 0, 2]
    area2 = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    lo, hi = p[:, :, :2].amin(dim=1), p[:, :, :2].amax(dim=1)
    c0 = torch.ceil(lo[:, 0] - 0.5 - eps).clamp_(min=0)
    c1 = torch.floor(hi[:, 0] - 0.5 + eps).clamp_(max=size - 1)
    r0 = torch.ceil(lo[:, 1] - 0.5 - eps).clamp_(min=0)
    r1 = torch.floor(hi[:, 1] - 0.5 + eps).clamp_(max=size - 1)
    nw, nh = c1 - c0 + 1, r1 - r0 + 1
    ok = (area2.abs() > 1e-12) & (nw > 0) & (nh > 0) & torch.isfinite(area2) & (p[:, :, 2] > 0).all(dim=1)
    ids = torch.nonzero(ok, as_tuple=False).squeeze(1)
    if ids.numel() == 0:
        return
    extent = torch.maximum(nw, nh)[ids]
    args = lambda t: (c1[t], r1[t], p0[t], a[t], b[t], area2[t], z0[t], dz1[t], dz2[t])
    prev = 0
    for S in _SIZE_CLASSES:
        sel = ids[(extent > prev) & (extent <= S)]
        prev = S
        step = max(1, max_candidates // (S * S))
        for k in range(0, sel.numel(), step):
            t = sel[k:k + step]
            c1_, r1_, p0_, a_, b_, ar_, z0_, d1_, d2_ = args(t)
            _depth_blocks(buf, size, t + id_offset, c0[t], r0[t], c1_, r1_, p0_, a_, b_, ar_, z0_, d1_, d2_, S, eps)
    big = ids[extent > _SIZE_CLASSES[-1]]
    if big.numel():
        S = _SIZE_CLASSES[-1]
        nbx, nby = torch.ceil(nw[big] / S).to(torch.int64), torch.ceil(nh[big] / S).to(torch.int64)
        per = nbx * nby
        owner = torch.repeat_interleave(torch.arange(big.numel(), device=dev), per)
        local = torch.arange(int(per.sum()), device=dev) - torch.repeat_interleave(torch.cumsum(per, 0) - per, per)
        nbx_o = torch.repeat_interleave(nbx, per)
        t_all = big[owner]
        bc0 = c0[t_all] + (local % nbx_o).to(torch.float32) * S
        br0 = r0[t_all] + (local // nbx_o).to(torch.float32) * S
        step = max(1, max_candidates // (S * S))
        for k in range(0, owner.numel(), step):
            t = t_all[k:k + step]
            c1_, r1_, p0_, a_, b_, ar_, z0_, d1_, d2_ = args(t)
            _depth_blocks(buf, size, t + id_offset, bc0[k:k + step], br0[k:k + step], c1_, r1_, p0_, a_, b_, ar_, z0_, d1_, d2_, S, eps)


@torch.no_grad()
def rasterize_depth(xyz: torch.Tensor, faces: torch.Tensor, size: int, max_candidates: int = 2_000_000, face_chunk: int = 1 << 21,
                    eps: float = 1e-5):
    """xyz: [V,3] (x px, y px, depth > 0). Returns (triangle id [S,S] int32, -1 = empty) and (depth [S,S] float32, inf = empty)."""
    xyz = xyz.to(torch.float32)
    faces = faces.to(torch.int64)
    buf = torch.full((size * size,), _EMPTY, dtype=torch.int64, device=xyz.device)
    for f0 in range(0, faces.shape[0], face_chunk):
        _depth_chunk(buf, xyz, faces[f0:f0 + face_chunk], f0, size, max_candidates, eps)
    empty = buf == _EMPTY
    tri = torch.where(empty, torch.full_like(buf, -1), buf & 0xFFFFFFFF).to(torch.int32)
    depth = (buf >> 32).to(torch.int32).view(torch.float32)
    depth = torch.where(empty, torch.full_like(depth, float("inf")), depth)
    return tri.view(size, size), depth.view(size, size)


def sample_image(img_chw: torch.Tensor, xy: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    """Bilinear (or bicubic) lookup of [C,H,W] at pixel coordinates [N,2] (centres at i + 0.5) -> [N,C]; borders clamped."""
    C, H, W = img_chw.shape
    grid = torch.stack([xy[:, 0] / W * 2 - 1, xy[:, 1] / H * 2 - 1], dim=1).view(1, 1, -1, 2).to(img_chw.dtype)
    return F.grid_sample(img_chw[None], grid, mode=mode, padding_mode="border", align_corners=False)[0, :, 0].t()


# ------------------------------------------------------------------------------------------------ geometry helpers
def smooth_vertex_normals(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Area-weighted normals, welded by position: vertices split along UV seams get the same normal (no seam in the weights)."""
    v = vertices.to(torch.float32)
    f = faces.to(torch.int64)
    lo = v.min(0).values
    extent = float((v.max(0).values - lo).max()) or 1.0
    q = torch.round((v - lo) / extent * (1 << 20)).to(torch.int64)
    key = (q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]
    _, weld = torch.unique(key, return_inverse=True)
    fn = torch.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]], dim=1)
    acc = torch.zeros((int(weld.max()) + 1, 3), dtype=torch.float32, device=v.device)
    for k in range(3):
        acc.index_add_(0, weld[f[:, k]], fn)
    return F.normalize(acc, dim=1)[weld]


def texel_ratio(vertices: torch.Tensor, faces: torch.Tensor, uv_px: torch.Tensor, pixels_per_unit: float) -> float:
    """Area-weighted median of (texels per view pixel), linear - how much a view pixel minifies the texture."""
    v = vertices.to(torch.float32)
    f = faces.to(torch.int64)
    a3 = torch.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]], dim=1).norm(dim=1) * 0.5
    u = uv_px.to(torch.float32)
    e1, e2 = u[f[:, 1]] - u[f[:, 0]], u[f[:, 2]] - u[f[:, 0]]
    a2 = (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]).abs() * 0.5
    ok = (a3 > 0) & (a2 > 0)
    if not bool(ok.any()):
        return 1.0
    r = torch.sqrt(a2[ok] / a3[ok]) / pixels_per_unit
    w = a3[ok]
    order = torch.argsort(r)
    cw = torch.cumsum(w[order], 0)
    return float(r[order][torch.searchsorted(cw, cw[-1] * 0.5).clamp(max=r.numel() - 1)])


def _gauss1d(sigma: float, device) -> torch.Tensor:
    r = max(1, int(math.ceil(3 * sigma)))
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=device, dtype=torch.float32) / sigma) ** 2)
    return k / k.sum()


def blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur of [C,H,W] (edges replicated)."""
    if sigma <= 0:
        return x
    k = _gauss1d(sigma, x.device)
    r = (k.numel() - 1) // 2
    C = x.shape[0]
    y = F.conv2d(F.pad(x[None], (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1).expand(C, 1, 1, -1), groups=C)
    return F.conv2d(F.pad(y, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1).expand(C, 1, -1, 1), groups=C)[0]


def smooth(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Gaussian blur of [C,H,W] that stays cheap for large ``sigma``: blurred at a reduced resolution (area-averaged by
    about sigma/3) and interpolated back - accurate for the smooth fields it is used for."""
    if sigma <= 6.0:
        return blur(x, sigma)
    r = int(sigma // 3)
    H, W = x.shape[1:]
    low = F.avg_pool2d(x[None], r, ceil_mode=True)
    low = blur(low[0], math.sqrt(max(sigma * sigma - r * r / 4.0, 1.0)) / r)
    return F.interpolate(low[None], size=(H, W), mode="bilinear", align_corners=False)[0]


def pick_supersample(ratio: float, size: int, max_pixels: int = 4096) -> int:
    """Smallest power of two >= the texel ratio (at least 2x for clean edges), capped so the raster stays <= max_pixels."""
    cap = max(1, max_pixels // int(size))
    s = 2
    while s < ratio and s * 2 <= cap:
        s *= 2
    return int(min(s, cap)) if cap >= 2 else 1


# ------------------------------------------------------------------------------------------------ rendering
@dataclass
class RenderedViews:
    cameras: List[Camera]
    images: torch.Tensor            # [N,S,S,3] unlit, anti-aliased, composited over ``background`` (CPU)
    masks: torch.Tensor             # [N,S,S] pixel coverage 0..1 (CPU)
    depth: torch.Tensor             # [N,S*k,S*k] nearest-surface depth at ``meta['depth_scale']`` = k times the view size, inf = empty
    normals: Optional[torch.Tensor] = None   # [N,S,S,3] camera-space normals (+z towards the camera), 0 on background (CPU)
    background: float = 0.5
    meta: dict = field(default_factory=dict)


@torch.no_grad()
def render_views(vertices: torch.Tensor, faces: torch.Tensor, uvs: torch.Tensor, texture: torch.Tensor, cameras: List[Camera],
                 background: float = 0.5, supersample: int = 0) -> RenderedViews:
    """Unlit, anti-aliased renders of ``texture`` ([H,W,3] float, glTF UV convention) on all ``cameras``.

    ``supersample`` 0 = automatic (from the texel density)."""
    dev = vertices.device
    vertices = vertices.to(torch.float32)
    faces = faces.to(dev, torch.int64)
    tex = texture.to(dev, torch.float32)[..., :3].permute(2, 0, 1).contiguous()
    th, tw = tex.shape[1:]
    uv_px = uvs.to(dev, torch.float32) * torch.tensor([tw, th], device=dev, dtype=torch.float32)
    S = int(cameras[0].size)
    ratio = texel_ratio(vertices, faces, uv_px, cameras[0].scale)
    ss = int(supersample) if supersample else pick_supersample(ratio, S)
    if ratio / ss > 1.25:                                                  # still minified: pre-filter the texture (mip level)
        tex = blur(tex, 0.5 * ratio / ss)
    ds = 2 if ss % 2 == 0 else 1                                           # depth kept at 2x the view size (visibility test)
    vn = smooth_vertex_normals(vertices, faces)
    images, masks, depths, normals = [], [], [], []
    for cam in cameras:
        hr = S * ss
        xyz = cam.project(vertices, ss)
        tri, depth = rasterize_depth(xyz, faces, hr)
        ys, xs = torch.nonzero(tri >= 0, as_tuple=True)
        col_hr = torch.zeros((3, hr, hr), device=dev)
        nrm_hr = torch.zeros((3, hr, hr), device=dev)
        if ys.numel():
            fc = faces[tri[ys, xs].long()]
            centre = torch.stack([xs.to(torch.float32) + 0.5, ys.to(torch.float32) + 0.5], dim=1)
            bary = barycentric(xyz[fc][:, :, :2], centre).unsqueeze(2)
            col_hr[:, ys, xs] = sample_image(tex, (bary * uv_px[fc]).sum(dim=1)).t()
            R = torch.as_tensor(cam.rotation, dtype=torch.float32, device=dev)
            n = F.normalize((bary * vn[fc]).sum(dim=1), dim=1) @ R.T
            nrm_hr[:, ys, xs] = torch.where(n[:, 2:3] < 0, -n, n).t()             # two-sided
            del fc, centre, bary, n
        cov_hr = (tri >= 0).to(torch.float32)[None]
        if ss > 1:
            # antialiased tent filter: non-negative, so flat colours stay exact (no ringing at silhouettes or colour edges)
            col = F.interpolate(col_hr[None], size=(S, S), mode="bilinear", antialias=True)[0]
            cov = F.interpolate(cov_hr[None], size=(S, S), mode="bilinear", antialias=True)[0, 0].clamp(0, 1)
            nrm = F.avg_pool2d(nrm_hr[None], ss)[0]
            d = -F.max_pool2d(-depth[None, None], ss // ds)[0, 0] if ss // ds > 1 else depth
        else:
            col, cov, nrm, d = col_hr, cov_hr[0], nrm_hr, depth
        img = (col + float(background) * (1.0 - cov)[None]).clamp(0, 1)
        nrm = torch.where(cov[None] > 1e-3, F.normalize(nrm, dim=0), torch.zeros_like(nrm))
        images.append(img.permute(1, 2, 0).cpu())
        masks.append(cov.cpu())
        normals.append(nrm.permute(1, 2, 0).cpu())
        depths.append(d.cpu())
        del tri, depth, col_hr, nrm_hr, cov_hr
    meta = {"supersample": ss, "depth_scale": ds, "texel_ratio": ratio}
    return RenderedViews(cameras, torch.stack(images), torch.stack(masks), torch.stack(depths), torch.stack(normals),
                         float(background), meta)

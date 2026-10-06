"""Write what an image model changed on the rendered views back into the UV texture - its detail and the features it painted
(letters, symbols, ornaments), without baked light, colour casts or dot patterns.

Per view, the change (refined - rendered) is split into luminance (applied to the texture as a gain, so hues are kept) and
chroma (``color_features`` of the colour of painted features - letters, symbols, ornaments - and ``color_detail`` of the
finest colour change, where colour fringes and noise live).  Then:

1. ``remove_lighting``: the luminance change is fitted against the surface normals (first-order spherical harmonics: overall
   brightness + any directional light, robust fit) and that fit is removed - the light an image model paints in;
2. broad changes - wider than ``detail_radius`` pixels of a 1024 view: shadows, occlusion, colour shifts - are dropped
   (``broad_changes`` = how much of them to keep);
3. ``noise_guard``: the part of the change that only amplifies faint, orientation-less structure already in the texture
   (voxel lattices, compression noise - what an image model "enhances" into dots and bumps) or shades existing structure as
   relief is removed with a local least-squares fit.  New detail is not predictable from the texture and stays; so does the
   sharpening of real edges unless ``keep_sharpening`` is off.

Every texel then takes the change from the views that really see it: depth test against a 2x depth buffer with a
slope-scaled tolerance; full strength only within ~50 deg of a view, fading to nothing at 75 deg (no stretched streaks, no
flickering visibility); fading near silhouettes and occlusion edges (no background halos); views blended by cos^4.  Texels that
no view sees well stay bit-identical; gutters follow their island.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

from ..core.mesh import MeshData, normalize_uvs_to_unit
from ..core.raster import barycentric, nearest_valid_texel, nearest_valid_texel_cpu, rasterize_uv_coverage
from ..core.render import RenderedViews, blur, sample_image, smooth_vertex_normals

_CHUNK = 1 << 20
_LUMA = (0.2126, 0.7152, 0.0722)
_EPS = 0.04                      # log-luminance offset: keeps the gain stable on near-black texels
COS_FULL = math.cos(math.radians(50))
COS_ZERO = math.cos(math.radians(75))


def luma(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    w = torch.tensor(_LUMA, dtype=x.dtype, device=x.device)
    shape = [1] * x.ndim
    shape[dim] = 3
    return (x * w.view(shape)).sum(dim)


def normalized_blur(x: torch.Tensor, w: torch.Tensor, sigma: float) -> torch.Tensor:
    """Weighted local mean of [C,H,W] with weights [H,W]: background and unreliable pixels never leak in."""
    num = blur(x * w[None], sigma)
    den = blur(w[None], sigma)
    return num / den.clamp(min=1e-4)


def view_reliability(coverage: torch.Tensor, depth_hr: torch.Tensor, normals: torch.Tensor, px_world: float, fade_px: float):
    """[S,S] weight 0..1: 0 on partly covered pixels and depth discontinuities (silhouettes, occlusion edges), ramping up to 1
    ``fade_px`` pixels away from them. A refiner smears background into the object there and depth tests are least certain."""
    S = coverage.shape[0]
    k = depth_hr.shape[0] // S
    d = -F.max_pool2d(-depth_hr[None, None], k)[0, 0] if k > 1 else depth_hr
    d = torch.nan_to_num(d, posinf=1e9)
    nz = normals[..., 2].clamp(min=0.05)
    tan = (torch.sqrt((1 - nz * nz).clamp(min=0)) / nz).clamp(max=6.0)
    thr = px_world * (3.0 + 2.0 * tan)
    edge = torch.zeros_like(coverage, dtype=torch.bool)
    for dy, dx in ((0, 1), (1, 0)):
        diff = (d - torch.roll(d, (dy, dx), (0, 1))).abs()
        e = diff > torch.maximum(thr, torch.roll(thr, (dy, dx), (0, 1)))
        edge |= e | torch.roll(e, (-dy, -dx), (0, 1))
    bad = (coverage < 0.999) | edge
    from scipy import ndimage
    dist = torch.from_numpy(ndimage.distance_transform_edt(~bad.cpu().numpy()).astype(np.float32)).to(coverage.device)
    return ((dist - 0.5) / max(fade_px, 1e-3)).clamp(0, 1)


def fit_lighting(d_lum: torch.Tensor, normals: torch.Tensor, w: torch.Tensor, iterations: int = 4) -> torch.Tensor:
    """Robust weighted fit of d_lum ~ a + b nx + c ny + d nz (overall gain + one directional light) -> fitted field [S,S]."""
    sel = w > 0.05
    if int(sel.sum()) < 64:
        return torch.zeros_like(d_lum)
    A_all = torch.cat([torch.ones_like(d_lum)[..., None], normals], dim=-1)             # [S,S,4]
    A, y, w0 = A_all[sel].double(), d_lum[sel].double(), w[sel].double()
    h = torch.ones_like(w0)
    beta = torch.zeros(4, dtype=torch.float64, device=d_lum.device)
    for _ in range(iterations):
        ww = w0 * h
        M = A.T @ (A * ww[:, None]) + 1e-6 * torch.eye(4, dtype=torch.float64, device=d_lum.device)
        beta = torch.linalg.solve(M, A.T @ (y * ww))
        r = y - A @ beta
        s = 1.4826 * r.abs().median().clamp(min=1e-6)
        h = torch.clamp(1.345 * s / r.abs().clamp(min=1e-12), max=1.0)                  # Huber: real detail is an outlier
    return (A_all.double() @ beta).to(d_lum.dtype)


def _dx(x: torch.Tensor) -> torch.Tensor:
    d = torch.zeros_like(x)
    d[..., :, 1:-1] = (x[..., :, 2:] - x[..., :, :-2]) * 0.5
    return d


def _dy(x: torch.Tensor) -> torch.Tensor:
    d = torch.zeros_like(x)
    d[..., 1:-1, :] = (x[..., 2:, :] - x[..., :-2, :]) * 0.5
    return d


def _bands(base: torch.Tensor, w: torch.Tensor, radius: float):
    """Difference-of-Gaussians bands of ``base`` from the finest pixels up to ``radius``."""
    sig = [0.75, 1.75]
    sig = [x for x in sig if x < radius * 0.8] + [radius]
    out, prev = [], base
    for sgm in sig:
        low = normalized_blur(base[None], w, sgm)[0]
        out.append(prev - low)
        prev = low
    return out


def _local_least_squares(regs: torch.Tensor, target: torch.Tensor, w: torch.Tensor, window: float, ridge: float) -> torch.Tensor:
    """Per-pixel coefficients [n,S,S] of target ~ sum_i k_i regs_i in Gaussian windows; the (smooth) normal equations are
    accumulated and solved at half resolution and the coefficients upsampled."""
    n, S1, S2 = regs.shape
    half = S1 >= 256 and S2 >= 256
    shrink = (lambda x: F.avg_pool2d(x[None], 2)[0]) if half else (lambda x: x)
    win = window / 2 if half else window
    pairs = [(i, j) for i in range(n) for j in range(i, n)]
    prods = torch.stack([blur(shrink((w * regs[i] * regs[j])[None]), win)[0] for i, j in pairs])
    rhs = torch.stack([blur(shrink((w * regs[i] * target)[None]), win)[0] for i in range(n)])
    h1, h2 = prods.shape[1:]
    eye = torch.eye(n, device=regs.device, dtype=regs.dtype)
    k = torch.empty((n, h1, h2), device=regs.device, dtype=regs.dtype)
    rows = max(1, (1 << 17) // h2)
    for r0 in range(0, h1, rows):
        sl = slice(r0, r0 + rows)
        M = torch.empty(prods[:, sl].shape[1:] + (n, n), device=regs.device, dtype=regs.dtype)
        for (i, j), pm in zip(pairs, prods[:, sl]):
            M[..., i, j] = pm
            M[..., j, i] = pm
        tr = torch.diagonal(M, dim1=-2, dim2=-1).sum(-1).clamp(min=0)
        M += (ridge * tr / n + 1e-12)[..., None, None] * eye
        k[:, sl] = torch.linalg.solve(M, rhs[:, sl].permute(1, 2, 0)[..., None])[..., 0].permute(2, 0, 1)
    if half:
        k = F.interpolate(k[None], size=(S1, S2), mode="bilinear", align_corners=False)[0]
    return k


def guard_against_amplification(change: torch.Tensor, base: torch.Tensor, w: torch.Tensor, radius: float, keep_sharpening: bool = True,
                                ridge: float = 0.01) -> torch.Tensor:
    """Remove from ``change`` [S,S] what merely re-scales or relief-shades structure already in ``base`` [S,S].

    The structure of ``base`` up to ``radius`` is split into frequency bands, and each band into a *noise-like* part - faint and
    without a dominant orientation (structure-tensor coherence): voxel lattices, compression noise, what image models boost
    most - and the rest (real edges and lines, also soft ones).  These, plus the x / y
    derivatives of the structure (relief shading, sub-pixel shifts), are fitted to the change by local, Gaussian-weighted least
    squares.  Removed: amplified faint structure and all relief shading.  Kept: the refiner cleaning noise up, new detail (not
    predictable from ``base``) and - with ``keep_sharpening`` - the sharpening of real edges."""
    bands = _bands(base, w, radius)
    sel = w > 0.5
    weak = []
    for bd in bands:
        tau = 2.5 * float(bd[sel].abs().median()) + 1e-4 if bool(sel.any()) else 1e-3
        weak.append(bd * torch.exp(-bd.abs() / tau))
    faint_all = sum(weak)                                     # structure tensor of the faint structure: real (soft) edges and
    gx, gy = _dx(faint_all), _dy(faint_all)                   # lines have ONE orientation, lattices and noise have none
    jxx = blur((gx * gx)[None], 2.0)[0]
    jyy = blur((gy * gy)[None], 2.0)[0]
    jxy = blur((gx * gy)[None], 2.0)[0]
    coherence = (torch.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (jxx + jyy + 1e-12)).clamp(0, 1) ** 2
    faint = [wk * (1.0 - coherence) for wk in weak]           # faint AND without orientation: noise-like
    strong = [bd - f for bd, f in zip(bands, faint)]
    fine = sum(bands[:-1]) if len(bands) > 1 else bands[0]
    deriv = [_dx(fine), _dy(fine)] + ([_dx(bands[-1]), _dy(bands[-1])] if len(bands) > 1 else [])
    nb = len(bands)
    regs = torch.stack(faint + strong + deriv)
    k = _local_least_squares(regs, change, w, max(3.0 * radius, 6.0), ridge)
    remove = (k[:nb].clamp(min=0) * regs[:nb]).sum(0)                                     # amplified faint structure
    if not keep_sharpening:
        remove = remove + (k[nb:2 * nb].clamp(min=0) * regs[nb:2 * nb]).sum(0)
    remove = remove + (k[2 * nb:] * regs[2 * nb:]).sum(0)                                  # relief shading, shifts
    return change - remove


def guided_residual(guide: torch.Tensor, target: torch.Tensor, sigma: float, eps: float = 1e-3, robust: bool = True) -> torch.Tensor:
    """[3,H,W] guide, [C,H,W] target -> target minus its best local linear prediction from the guide's colours.

    A colour guided filter (local affine map guide -> target in Gaussian windows of ``sigma``, computed at reduced
    resolution and interpolated back): whatever the target does as a smooth function of the guide's colours - greying,
    tinting, a saturation or brightness change - is predicted and removed; what is not predictable from the colours that
    are there - new marks, letters, symbols - is the residual.  One robust pass keeps the new marks from biasing the fit."""
    C, H, W = target.shape
    r = max(1, int(sigma // 4))
    s = sigma / r
    pool = (lambda x: F.avg_pool2d(x[None], r, ceil_mode=True)[0]) if r > 1 else (lambda x: x)
    g, t = pool(guide), pool(target)
    h, w_ = g.shape[1:]
    dev, dt = guide.device, guide.dtype
    wts = torch.ones((1, h, w_), device=dev, dtype=dt)
    pairs = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
    for it in range(2 if robust else 1):
        sw = blur(wts, s).clamp(min=1e-6)
        m = lambda x: blur(wts * x, s) / sw
        mg, mt = m(g), m(t)
        cov = torch.zeros((h, w_, 3, 3), device=dev, dtype=dt)
        for i, j in pairs:
            v = m(g[i:i + 1] * g[j:j + 1])[0] - mg[i] * mg[j]
            cov[..., i, j] = v
            cov[..., j, i] = v
        cov = cov + eps * torch.eye(3, device=dev, dtype=dt)
        cgt = torch.stack([m(g[i:i + 1] * t)[:, ...] - mg[i:i + 1] * mt for i in range(3)], dim=0)   # [3(in), C, h, w]
        A = torch.linalg.solve(cov[None].expand(C, h, w_, 3, 3), cgt.permute(1, 2, 3, 0)[..., None])[..., 0]   # [C,h,w,3]
        b = mt - (A * mg.permute(1, 2, 0)[None]).sum(-1)                                                # [C,h,w]
        A = blur(A.permute(0, 3, 1, 2).reshape(C * 3, h, w_), s)
        b = blur(b, s)
        if it == 0 and robust:
            pred = (A.reshape(C, 3, h, w_) * g[None]).sum(1) + b
            res = (t - pred).norm(dim=0, keepdim=True)
            tau = 2.5 * float(res.median()) + 1e-4
            wts = 1.0 / (1.0 + (res / tau) ** 2)
    up = lambda x: F.interpolate(x[None], size=(H, W), mode="bilinear", align_corners=False)[0] if r > 1 else x
    A, b = up(A).reshape(C, 3, H, W), up(b)
    return target - ((A * guide[None]).sum(1) + b)


@torch.no_grad()
def view_change(refined: torch.Tensor, views: RenderedViews, v: int, device, detail_radius: float = 16.0, broad_changes: float = 0.0,
                remove_lighting: bool = True, noise_guard: bool = True, keep_sharpening: bool = True, color_detail: float = 0.25,
                color_features: float = 1.0):
    """-> [5,S,S]: luminance change as a log gain, chroma change (3, already weighted), reliability - for view ``v``.

    Colour: casts, greying and re-colouring of what is there are removed first (a local colour mapping of the render, see
    :func:`guided_residual`); of the rest ``color_detail`` of the finest change (a pixel or two: colour fringes where the
    refiner moved an edge, colour noise), ``color_features`` of larger painted features (marks, letters, ornaments) and
    ``broad_changes`` of what is wider than ``detail_radius``."""
    cam = views.cameras[v]
    S = cam.size
    ren = views.images[v].to(device, torch.float32)
    ref = refined.to(device, torch.float32)[..., :3]
    if ref.shape[0] != S or ref.shape[1] != S:
        big = ref.shape[0] > S
        ref = F.interpolate(ref.permute(2, 0, 1)[None], size=(S, S), mode="bicubic", antialias=big)[0].permute(1, 2, 0).clamp(0, 1)
    cov = views.masks[v].to(device, torch.float32)
    nrm = views.normals[v].to(device, torch.float32)
    rel = view_reliability(cov, views.depth[v].to(device), nrm, 1.0 / cam.scale, max(2.0, 6.0 * S / 1024))
    y_ren, y_ref = luma(ren), luma(ref)
    if remove_lighting:                                     # divide the painted light out of the refined luminance
        shade = fit_lighting(torch.log(y_ref + _EPS) - torch.log(y_ren + _EPS), nrm, rel)
        y_ref = (y_ref + _EPS) * torch.exp(-shade) - _EPS
    radius = max(1.0, float(detail_radius) * S / 1024)
    d_y = y_ref - y_ren
    d_chr = ((ref - luma(ref)[..., None]) - (ren - y_ren[..., None])).permute(2, 0, 1)
    if bool(d_chr.abs().max() > 0):                     # colour casts, greying, re-colouring of what is there: a colour mapping
        d_chr = guided_residual(ren.permute(2, 0, 1), d_chr, max(4.0, 1.5 * radius))          # of the render - not new features
    low_y = normalized_blur(d_y[None], rel, radius)[0]
    low_c = normalized_blur(d_chr, rel, radius)
    d_y = (d_y - low_y) + float(broad_changes) * low_y
    mid = normalized_blur(d_chr, rel, max(0.75, 1.5 * S / 1024))                 # colour of features larger than a pixel or two
    d_chr = (float(color_detail) * (d_chr - mid) + float(color_features) * (mid - low_c)
             + float(broad_changes) * low_c)
    if noise_guard:                                     # amplified noise is a fine-scale effect: the guard's own scale
        d_y = guard_against_amplification(d_y, y_ren, rel, max(1.0, min(radius, 6.0 * S / 1024)), keep_sharpening)
    gain = torch.log((y_ren + d_y).clamp(min=0) + _EPS) - torch.log(y_ren + _EPS)
    return torch.cat([gain[None], d_chr, rel[None]], dim=0)


@torch.no_grad()
def texel_geometry(mesh: MeshData, height: int, width: int, device: torch.device):
    """Covered texels of the UV layout with their 3-D position and smooth (seam-welded) normal."""
    v = torch.from_numpy(mesh.vertices).to(device, torch.float32)
    f = torch.from_numpy(mesh.faces).to(device, torch.int64)
    uv_px = torch.from_numpy(normalize_uvs_to_unit(mesh.uvs)).to(device) * torch.tensor([width, height], device=device, dtype=torch.float32)
    tri = rasterize_uv_coverage(uv_px, f, height, width)
    covered = tri >= 0
    ys, xs = torch.nonzero(covered, as_tuple=True)
    fc = f[tri[ys, xs].long()]
    bary = barycentric(uv_px[fc], torch.stack([xs.to(torch.float32) + 0.5, ys.to(torch.float32) + 0.5], dim=1)).unsqueeze(2)
    pos = (bary * v[fc]).sum(dim=1)
    nrm = F.normalize((bary * smooth_vertex_normals(v, f)[fc]).sum(dim=1), dim=1)
    return covered, ys, xs, pos, nrm


def uv_overlap(mesh: MeshData, height: int, width: int, n_covered: int) -> float:
    """Share of the UV area that lies on top of other UV triangles (mirrored / stacked parts share texels)."""
    uv = normalize_uvs_to_unit(mesh.uvs).astype(np.float64) * [width, height]
    f = mesh.faces
    e1, e2 = uv[f[:, 1]] - uv[f[:, 0]], uv[f[:, 2]] - uv[f[:, 0]]
    total = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]).sum()
    return float(max(0.0, 1.0 - n_covered / max(total, 1.0)))


def _texel_size_world(mesh: MeshData, n_covered: int) -> float:
    v, f = mesh.vertices.astype(np.float64), mesh.faces
    area = 0.5 * np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1).sum()
    return float(math.sqrt(area / max(n_covered, 1)))


def _smoothstep(e0: float, e1: float, x: torch.Tensor) -> torch.Tensor:
    t = ((x - e0) / (e1 - e0)).clamp(0, 1)
    return t * t * (3 - 2 * t)


@torch.no_grad()
def project_detail(mesh: MeshData, texture: torch.Tensor, views: RenderedViews, refined: torch.Tensor, detail_strength: float = 1.0,
                   color_detail: float = 0.25, broad_changes: float = 0.0, detail_radius: float = 16.0, remove_lighting: bool = True,
                   noise_guard: bool = True, device: torch.device = torch.device("cpu"), gutter: int = 16, blend_power: float = 4.0,
                   keep_sharpening: bool = True, color_features: float = 1.0):
    """texture [H,W,3] (as rendered), refined [N,S,S,3] -> (new texture [H,W,3], confidence mask [H,W]) on the CPU."""
    if mesh.uvs is None:
        raise ValueError("The mesh has no UVs: use the same mesh that was given to 'STM — Render Views'.")
    if refined.shape[0] != len(views.cameras):
        raise ValueError(f"{refined.shape[0]} refined views for {len(views.cameras)} rendered views: connect the matching batch.")
    if views.normals is None:
        raise ValueError("These views come from an older 'STM — Render Views': render them again.")
    H, W = texture.shape[:2]
    tex = texture.to(device, torch.float32)[..., :3]
    covered, ys, xs, pos, nrm = texel_geometry(mesh, H, W, device)
    n = ys.numel()
    overlap = uv_overlap(mesh, H, W, n)
    if overlap > 0.1:
        print(f"[STM] project views: {overlap:.0%} of the UV area overlaps (mirrored or stacked parts share texels); each shared "
              "texel takes the detail of one of its surfaces", flush=True)
    texel_world = _texel_size_world(mesh, n)
    acc = torch.zeros((n, 4), device=device)
    wsum = torch.zeros(n, device=device)
    conf = torch.zeros(n, device=device)
    ds = int(views.meta.get("depth_scale", views.depth.shape[1] // views.cameras[0].size))
    for v, cam in enumerate(views.cameras):
        if not bool((refined[v][..., :3].float() - views.images[v]).abs().gt(1.0 / 512).any()):
            continue                                                   # view left untouched: nothing to project
        S = cam.size
        maps = view_change(refined[v], views, v, device, detail_radius, broad_changes, remove_lighting, noise_guard, keep_sharpening,
                           color_detail, color_features)
        texel_px = texel_world * cam.scale
        if texel_px > 1.0:                                             # texels larger than view pixels: average their footprint
            maps = torch.cat([blur(maps[:4], 0.5 * texel_px), maps[4:]], dim=0)
        depth = views.depth[v].to(device)
        Sd = depth.shape[0]
        direction = torch.as_tensor(cam.direction, dtype=torch.float32, device=device)
        px_hr = 1.0 / (cam.scale * ds)
        for c0 in range(0, n, _CHUNK):
            sl = slice(c0, c0 + _CHUNK)
            xyz = cam.project(pos[sl])
            cos = nrm[sl] @ direction
            inside = (xyz[:, 0] >= 0) & (xyz[:, 0] < S) & (xyz[:, 1] >= 0) & (xyz[:, 1] < S) & (cos > COS_ZERO)
            ix = (xyz[:, 0] * ds).long().clamp(0, Sd - 1)
            iy = (xyz[:, 1] * ds).long().clamp(0, Sd - 1)
            zb = depth[iy, ix]
            tan = (torch.sqrt((1 - cos * cos).clamp(min=0)) / cos.clamp(min=1e-3)).clamp(max=6.0)
            visible = inside & torch.isfinite(zb) & (xyz[:, 2] <= zb + px_hr * (1.5 + 1.5 * tan))
            if not bool(visible.any()):
                continue
            s = torch.cat([sample_image(maps[:4], xyz[:, :2], "bicubic"),        # change: bicubic keeps fine detail
                           sample_image(maps[4:], xyz[:, :2])], dim=1)             # reliability: bilinear, no overshoot
            ramp = _smoothstep(COS_ZERO, COS_FULL, cos)
            c = torch.where(visible, s[:, 4].clamp(0, 1) * ramp, torch.zeros_like(cos))
            w = c * cos.clamp(min=0).pow(blend_power)
            acc[sl] += w[:, None] * s[:, :4]
            wsum[sl] += w
            conf[sl] = torch.maximum(conf[sl], c)
    seen = wsum > 1e-12
    mean = torch.where(seen[:, None], acc / wsum.clamp(min=1e-12)[:, None], torch.zeros_like(acc))
    f = conf * float(detail_strength)
    old = tex[ys, xs]
    gain = torch.exp((f * mean[:, 0]).clamp(-1.5, 1.5))
    new = (old * gain[:, None] + f[:, None] * mean[:, 1:4]).clamp(0, 1)
    delta_map = torch.zeros((H, W, 3), device=device)
    delta_map[ys, xs] = new - old
    src = nearest_valid_texel_cpu(covered.cpu(), gutter).to(device) if device.type == "cpu" else nearest_valid_texel(covered, gutter)
    gut = ((src >= 0) & ~covered).view(-1)
    flat = delta_map.view(-1, 3)
    flat[gut] = flat[src.view(-1)[gut].long()]
    mask = torch.zeros((H, W), device=device)
    mask[ys, xs] = conf
    return (tex + delta_map).clamp(0, 1).cpu(), mask.cpu()

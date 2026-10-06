"""Mesh loading: GLB / glTF / OBJ / PLY / STL file -> :class:`MeshData` plus its textures, in memory, without any model.

Textures: the glTF PBR base colour, metallic-roughness and normal maps (or an OBJ's MTL image) come along, ready for the
native ``MESH`` (``texture`` / ``metallic_roughness`` / ``normal_map``).  Assets whose parts use *different* textures - common
for older, hand-made models - are packed into one atlas (one cell per material, solid-colour parts share a swatch cell), with
each material's colour / metallic / roughness factors baked in, so the result is a single-texture mesh every node understands.
"""
import hashlib
import math
import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..core.mesh import MeshData
from ..core.normalize import compute_transform

MAX_ATLAS = 8192


@dataclass
class LoadedAsset:
    mesh: MeshData
    texture: Optional[np.ndarray] = None              # [H,W,3] float32 0..1, sRGB
    metallic_roughness: Optional[np.ndarray] = None   # [H,W,3] glTF packing (G = roughness, B = metallic)
    normal_map: Optional[np.ndarray] = None           # [H,W,3] tangent space
    material: Optional[dict] = None                   # factors for the native MESH (single-material assets only)
    notes: List[str] = field(default_factory=list)


def resolve_mesh_path(value: str, input_dir: Optional[str]) -> str:
    """Absolute path as given, or relative to ComfyUI's input folder (a missing extension defaults to .glb)."""
    value = (value or "").strip().strip('"').strip("'")
    if not value:
        raise ValueError("mesh_path is empty: give a file name from ComfyUI/input or an absolute path to a .glb file.")
    candidates: List[str] = []
    bases = [value] if os.path.isabs(value) else ([os.path.join(input_dir, value)] if input_dir else []) + [os.path.abspath(value)]
    for b in bases:
        candidates += [b, b + ".glb"] if not os.path.splitext(b)[1] else [b]
    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)
    raise FileNotFoundError("Mesh file not found. Looked for:\n  " + "\n  ".join(candidates))


# ------------------------------------------------------------------------------------------------ material reading
def _rgb(img) -> Optional[np.ndarray]:
    if img is None:
        return None
    return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def _factor(value, n: int, default: float) -> np.ndarray:
    if value is None:
        return np.full(n, default, np.float32)
    a = np.asarray(value, dtype=np.float64).reshape(-1)
    if a.dtype.kind in "ui" or a.max() > 1.0 + 1e-6:
        a = a / 255.0
    out = np.full(n, default, np.float32)
    out[:min(n, a.size)] = a[:n]
    return out


def _srgb_to_linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(x):
    x = np.clip(x, 0.0, None)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


@dataclass
class _Mat:
    base: Optional[np.ndarray]
    base_factor: np.ndarray            # RGBA
    mr: Optional[np.ndarray]
    metallic: float
    roughness: float
    normal: Optional[np.ndarray]
    key: tuple


def _digest(img) -> Optional[str]:
    if img is None:
        return None
    small = img.convert("RGB")
    return hashlib.sha1(small.tobytes() + repr(small.size).encode()).hexdigest()


def _read_material(visual, cache: dict) -> _Mat:
    from trimesh.visual.material import PBRMaterial
    mat = getattr(visual, "material", None)
    base_img = mr_img = nrm_img = None
    base_factor, metallic, roughness = np.ones(4, np.float32), 0.0, 1.0
    if isinstance(mat, PBRMaterial):
        base_img, mr_img, nrm_img = mat.baseColorTexture, mat.metallicRoughnessTexture, mat.normalTexture
        base_factor = _factor(mat.baseColorFactor, 4, 1.0)
        metallic = 1.0 if mat.metallicFactor is None else float(mat.metallicFactor)
        roughness = 1.0 if mat.roughnessFactor is None else float(mat.roughnessFactor)
    elif mat is not None:                                                   # OBJ / MTL (SimpleMaterial)
        base_img = getattr(mat, "image", None)
        if base_img is None and getattr(mat, "diffuse", None) is not None:
            base_factor = _factor(mat.diffuse, 4, 1.0)
        metallic, roughness = 0.0, 1.0 if getattr(mat, "glossiness", None) is None else float(np.clip(1 - mat.glossiness / 100.0, 0, 1))
    digests = tuple(cache.setdefault(id(i), _digest(i)) if i is not None else None for i in (base_img, mr_img, nrm_img))
    key = digests + (tuple(np.round(base_factor, 4)), round(metallic, 4), round(roughness, 4))
    return _Mat(_rgb(base_img), base_factor, _rgb(mr_img), metallic, roughness, _rgb(nrm_img), key)


def _baked(m: _Mat):
    """Base colour / metallic-roughness with the factors baked in (texture, or 1x1 solid colour when there is none)."""
    base = m.base if m.base is not None else np.ones((1, 1, 3), np.float32)
    if not np.allclose(m.base_factor[:3], 1.0):
        base = _linear_to_srgb(_srgb_to_linear(base) * m.base_factor[:3]).astype(np.float32)
    mr = m.mr if m.mr is not None else np.ones((1, 1, 3), np.float32)
    mr = mr * np.array([1.0, m.roughness, m.metallic], np.float32)
    return np.clip(base, 0, 1), np.clip(mr, 0, 1)


# ------------------------------------------------------------------------------------------------ atlas packing
def _resize(img: np.ndarray, h: int, w: int) -> np.ndarray:
    if img.shape[0] == h and img.shape[1] == w:
        return img
    from PIL import Image
    chans = [np.asarray(Image.fromarray(img[..., c].astype(np.float32), mode="F").resize((w, h), Image.BICUBIC)) for c in range(img.shape[2])]
    return np.clip(np.stack(chans, -1), 0, 1).astype(np.float32)


def _pack(groups: list, notes: List[str]):
    """groups: [(_Mat, [part index...], textured: bool)] -> (cell rectangles, per-group placer, atlas maps)."""
    textured = [g for g in groups if g[2]]
    solid = [g for g in groups if not g[2]]
    cell = max(max(max(g[0].base.shape[:2]) for g in textured), 64) if textured else 256
    n_cells = len(textured) + (1 if solid else 0)
    cols = math.ceil(math.sqrt(n_cells))
    rows = math.ceil(n_cells / cols)
    if max(cols, rows) * cell > MAX_ATLAS:
        new_cell = MAX_ATLAS // max(cols, rows)
        notes.append(f"{len(textured)} textures packed into one {cols * new_cell}x{rows * new_cell} atlas (each reduced from {cell} to {new_cell} px)")
        cell = new_cell
    else:
        notes.append(f"{len(textured)} texture(s){' + solid colours' if solid else ''} packed into one {cols * cell}x{rows * cell} atlas")
    H, W = rows * cell, cols * cell
    margin = max(2, cell // 128)
    has_mr = any(g[0].mr is not None or g[0].metallic != 0 or g[0].roughness != 1 for g in groups)
    has_n = any(g[0].normal is not None for g in groups)
    base = np.zeros((H, W, 3), np.float32)
    mr = np.ones((H, W, 3), np.float32) if has_mr else None
    nrm = np.tile(np.array([0.5, 0.5, 1.0], np.float32), (H, W, 1)) if has_n else None
    placers = {}

    def put(dst, img, y0, x0, size):
        inner = size - 2 * margin
        dst[y0 + margin:y0 + margin + inner, x0 + margin:x0 + margin + inner] = _resize(img, inner, inner)
        block = dst[y0:y0 + size, x0:x0 + size]                               # margin: edge texels repeated (no bleeding)
        block[:margin] = block[margin:margin + 1]
        block[size - margin:] = block[size - margin - 1:size - margin]
        block[:, :margin] = block[:, margin:margin + 1]
        block[:, size - margin:] = block[:, size - margin - 1:size - margin]

    for k, (m, parts, _) in enumerate(textured):
        r, c = divmod(k, cols)
        y0, x0 = r * cell, c * cell
        b, mrm = _baked(m)
        put(base, b, y0, x0, cell)
        if mr is not None:
            put(mr, mrm if m.mr is not None else np.broadcast_to(mrm, (1, 1, 3)).copy(), y0, x0, cell)
        if nrm is not None and m.normal is not None:
            put(nrm, m.normal, y0, x0, cell)
        inner = cell - 2 * margin
        placers[id(m)] = ("map", (x0 + margin) / W, (y0 + margin) / H, inner / W, inner / H)
    if solid:
        k = len(textured)
        r, c = divmod(k, cols)
        y0, x0 = r * cell, c * cell
        grid = max(1, math.ceil(math.sqrt(len(solid))))
        sw = cell // grid
        for j, (m, parts, _) in enumerate(solid):
            sr, sc = divmod(j, grid)
            b, mrm = _baked(m)
            base[y0 + sr * sw:y0 + (sr + 1) * sw, x0 + sc * sw:x0 + (sc + 1) * sw] = b.reshape(1, 1, 3)
            if mr is not None:
                mr[y0 + sr * sw:y0 + (sr + 1) * sw, x0 + sc * sw:x0 + (sc + 1) * sw] = mrm.reshape(1, 1, 3)
            placers[id(m)] = ("point", (x0 + (sc + 0.5) * sw) / W, (y0 + (sr + 0.5) * sw) / H)
    return placers, base, mr, nrm


# ------------------------------------------------------------------------------------------------ loading
def load_asset(path: str, keep_uvs: bool = True, normalize: bool = False, load_textures: bool = True) -> LoadedAsset:
    try:
        import trimesh
    except ImportError as e:
        raise ImportError("STM — Load Mesh needs the 'trimesh' package: python -m pip install trimesh") from e
    asset = trimesh.load(path, force="scene", process=False)       # process=False: keep every vertex (UV seams!)
    parts = asset.dump() if isinstance(asset, trimesh.Scene) else [asset]      # dump() applies the scene-graph transforms
    parts = [p for p in parts if isinstance(p, trimesh.Trimesh) and len(p.vertices) and len(p.faces)]
    if not parts:
        raise ValueError(f"{os.path.basename(path)} contains no triangle mesh geometry.")
    verts = np.concatenate([np.asarray(p.vertices, dtype=np.float32) for p in parts])
    offsets = np.cumsum([0] + [len(p.vertices) for p in parts[:-1]])
    faces = np.concatenate([np.asarray(p.faces, dtype=np.int64) + o for p, o in zip(parts, offsets)])
    if not np.isfinite(verts).all():
        raise ValueError("The mesh contains non-finite vertex coordinates.")
    if normalize:
        t = compute_transform(verts)
        verts = ((verts.astype(np.float64) - t.center) * t.scale).astype(np.float32)
    out = LoadedAsset(MeshData(verts, faces))

    def part_uv(p):
        uv = getattr(p.visual, "uv", None)
        if uv is None or len(uv) != len(p.vertices):
            return None
        uv = np.asarray(uv, dtype=np.float32).copy()
        uv[:, 1] = 1.0 - uv[:, 1]                    # trimesh (bottom-origin) -> glTF (top-origin) = texture-row convention
        return uv

    uvs = [part_uv(p) for p in parts]
    if not keep_uvs:
        return out
    cache: dict = {}
    mats = [_read_material(p.visual, cache) if load_textures and getattr(p.visual, "kind", None) == "texture" else None for p in parts]
    textured = [i for i, m in enumerate(mats) if m is not None and m.base is not None and uvs[i] is not None]
    if not textured:                                                     # geometry (+ UVs of a single part) only
        if len(parts) == 1 and uvs[0] is not None:
            out.mesh.uvs = uvs[0]
        if load_textures and any(m is not None for m in mats):
            out.notes.append("no base-colour texture found")
        return out
    groups: dict = {}
    for i, m in enumerate(mats):
        if i in textured:
            key, gm = m.key, m
        else:                                                            # no texture or no UVs: a solid-colour part
            gm = _solid_mat(parts[i], m)
            key = gm.key
        if key not in groups:
            groups[key] = [gm, [], i in textured]
        groups[key][1].append(i)
    glist = [tuple(g) for g in groups.values()]
    if len(glist) == 1:                                                  # one material: its texture as is, factors as material
        m = glist[0][0]
        out.mesh.uvs = np.concatenate(uvs).astype(np.float32)
        out.texture = m.base
        if m.mr is not None or m.metallic != 0.0 or m.roughness != 1.0:
            out.metallic_roughness = m.mr if m.mr is not None else None
        out.normal_map = m.normal
        mat = {}
        if not np.allclose(m.base_factor, 1.0):
            mat["base_color_factor"] = [float(c) for c in m.base_factor]
        if m.mr is not None:
            mat["metallic_factor"], mat["roughness_factor"] = float(m.metallic), float(m.roughness)
        elif m.metallic != 0.0 or m.roughness != 1.0:
            out.metallic_roughness = np.broadcast_to(np.array([1.0, m.roughness, m.metallic], np.float32), (4, 4, 3)).copy()
        out.material = mat or None
        return out
    for m, idx, is_tex in glist:                                         # several materials: pack them into one atlas
        if is_tex:
            u = np.concatenate([uvs[i] for i in idx])
            if u.min() < -0.01 or u.max() > 1.01:
                raise ValueError(f"{os.path.basename(path)}: its parts use {len([g for g in glist if g[2]])} different textures and some UVs "
                                 f"repeat the texture (range {u.min():.2f}..{u.max():.2f}), so they cannot be merged into one atlas. "
                                 "Export the asset with a single texture atlas (e.g. bake in Blender), or set keep_uvs off to load "
                                 "the geometry only.")
    placers, base, mr, nrm = _pack(glist, out.notes)
    all_uv = []
    for i, p in enumerate(parts):
        m = next(g[0] for g in glist if i in g[1])
        kind, *a = placers[id(m)]
        if kind == "map":
            x0, y0, sx, sy = a
            u = np.clip(uvs[i], 0.0, 1.0)
            all_uv.append(np.stack([x0 + u[:, 0] * sx, y0 + u[:, 1] * sy], 1))
        else:
            all_uv.append(np.tile(np.array(a, np.float32), (len(p.vertices), 1)))
    out.mesh.uvs = np.concatenate(all_uv).astype(np.float32)
    out.texture, out.metallic_roughness, out.normal_map = base, mr, nrm
    return out


def _part_colour(part) -> np.ndarray:
    vis = part.visual
    try:
        if getattr(vis, "kind", None) in ("vertex", "face"):
            c = np.asarray(vis.vertex_colors if vis.kind == "vertex" else vis.face_colors, dtype=np.float64)[:, :3].mean(0) / 255.0
            return c.astype(np.float32)
        mat = getattr(vis, "material", None)
        if mat is not None:
            from trimesh.visual.material import PBRMaterial
            if isinstance(mat, PBRMaterial):
                return _factor(mat.baseColorFactor, 4, 1.0)[:3]
            if getattr(mat, "diffuse", None) is not None:
                return _factor(mat.diffuse, 4, 1.0)[:3]
    except Exception:
        pass
    return np.full(3, 0.8, np.float32)


def _solid_mat(part, m: Optional[_Mat] = None) -> _Mat:
    if m is not None:                                                    # textured material on a part without UVs: mean colour
        c = (m.base.reshape(-1, 3).mean(0) if m.base is not None else np.ones(3, np.float32)) * m.base_factor[:3]
        metallic, roughness = m.metallic, m.roughness
    else:
        c, metallic, roughness = _part_colour(part), 0.0, 1.0
    c = np.clip(c, 0, 1).astype(np.float32)
    return _Mat(None, np.concatenate([c, [1.0]]).astype(np.float32), None, metallic, roughness, None,
                ("solid", tuple(np.round(c, 4)), round(metallic, 4), round(roughness, 4)))


def load_mesh_file(path: str, keep_uvs: bool = True, normalize: bool = False) -> MeshData:
    """Geometry (+ UVs) only - kept for callers that do not need the textures."""
    return load_asset(path, keep_uvs, normalize, load_textures=False).mesh

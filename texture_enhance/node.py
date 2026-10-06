"""Texture enhancement for ANY textured mesh: MESH (UVs + base-colour texture) -> unlit views -> image model (e.g. FLUX.2
[klein] 4B through ComfyUI's own nodes) -> detail written back into the texture -> MESH.

Works right after the STM texturing nodes, on a mesh from 'STM — Load Mesh' / ComfyUI's 'Get 3D Components', or on any other
node that outputs a textured MESH (old assets, other generators, a texture you want to refresh with a new seed).
"""
import torch

from ..core.mesh import mesh_data_from_comfy
from ..core.render import VIEW_SET_HELP, VIEW_SETS, orbit_cameras, render_views
from ..runtime.device import pick_device
from ..runtime.stage import GiB, log, stage_scope
from .project import project_detail
from .textures import mesh_appearance, store_texture, unit_uvs, with_texture
from .tiles import merge_tiles, split_tiles


class STMRenderViews:
    """Unlit (no light, no shadow, no shading), anti-aliased renders of a textured mesh from several directions."""

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "mesh": ("MESH", {"tooltip": "A mesh with UVs and a base-colour texture: from 'STM — Apply Texture', 'STM — Load Mesh', "
                                         "'Get 3D Components' or any node that outputs a textured MESH."}),
            "resolution": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64,
                                   "tooltip": "View size. Higher = finer detail reaches the texture (and more tiles / refiner time). "
                                              "1024 suits 2K textures and fast passes; 2048 (with tile_size 1088: 4 tiles per view) "
                                              "suits 4K textures - the biggest single step up in detail."}),
            "views": (sorted(VIEW_SETS), {"default": 8, "tooltip": VIEW_SET_HELP}),
            "background": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Grey level behind the object."}),
        }, "optional": {
            "base_color": ("IMAGE", {"tooltip": "Optional: use this texture instead of the mesh's own base colour."}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK", "STM_VIEWS")
    RETURN_NAMES = ("views", "view_masks", "view_data")
    FUNCTION = "run"
    CATEGORY = "STM/enhance"

    def run(self, mesh, resolution, views, background, base_color=None):
        data = unit_uvs(mesh_data_from_comfy(mesh))
        appearance, _ = mesh_appearance(mesh, base_color)
        dev = pick_device()
        with stage_scope("render views", dev, reserve=int(1.5 * GiB)):
            cams = orbit_cameras(data.vertices, int(resolution), int(views))
            out = render_views(torch.from_numpy(data.vertices).to(dev), torch.from_numpy(data.faces).to(dev),
                               torch.from_numpy(data.uvs).to(dev), appearance, cams, float(background))
        th, tw = appearance.shape[:2]
        log(f"render views: {len(cams)} unlit views of {resolution}px, {out.meta['supersample']}x supersampled "
            f"(texture {tw}x{th}, {out.meta['texel_ratio']:.1f} texels per view pixel)")
        return (out.images, out.masks, out)


class STMSplitTiles:
    """Views -> overlapping tiles for the refiner, one at a time (bounded VRAM). Background-only tiles are skipped."""

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "images": ("IMAGE",),
            "tile_size": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64,
                                  "tooltip": "Largest tile the refiner gets. Tiles are spread evenly; for 2048 views 1088 gives 4 tiles per view instead of 9."}),
            "overlap": ("INT", {"default": 128, "min": 0, "max": 1024, "step": 16, "tooltip": "Minimum feathered overlap between tiles (hides seams)."}),
            "cancel_lighting": ("BOOLEAN", {"default": True,
                                            "tooltip": "Refine every tile twice, the second time upside down, and average: the light the refiner "
                                                       "paints from above cancels out (flat, unlit albedo). Doubles refiner time."}),
        }, "optional": {"masks": ("MASK", {"tooltip": "Object masks (view_masks): tiles without object pixels are not sent to the refiner."})}}

    RETURN_TYPES = ("IMAGE", "STM_TILES")
    RETURN_NAMES = ("tiles", "tile_layout")
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "run"
    CATEGORY = "STM/enhance"

    def run(self, images, tile_size, overlap, cancel_lighting=True, masks=None):
        tiles, layout = split_tiles(images, int(tile_size), int(overlap), masks, cancel_lighting=bool(cancel_lighting))
        extra = " (each also upside down, to cancel painted light)" if cancel_lighting else ""
        log(f"split {images.shape[0]} image(s) of {images.shape[2]}x{images.shape[1]} into {len(tiles)} tile(s){extra}")
        return (tiles, layout)


class STMMergeTiles:
    """Refined tiles -> full views with feathered overlaps (the inverse of 'STM — Split Tiles').  With keep_colors the
    original colours and brightness are put back and what the refiner really added stays: new detail, sharper edges, new
    marks, letters and symbols (e.g. taken from a reference image) in their own colours."""

    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "tiles": ("IMAGE",), "tile_layout": ("STM_TILES",),
            "keep_colors": ("BOOLEAN", {"default": True,
                                        "tooltip": "On: greying, tinting, re-lighting and re-colouring of what is already there are taken out - "
                                                   "the original colours and brightness stay; new detail and new coloured features (letters, "
                                                   "symbols, ornaments) stay too. Off: the refiner's output as it is."}),
            "detail_size": ("FLOAT", {"default": 16.0, "min": 2.0, "max": 64.0, "step": 1.0,
                                      "tooltip": "In pixels of a 1024 view: the size of the features the refiner may add with keep_colors "
                                                 "(letters, symbols, ornaments up to about this size keep their own colour and brightness)."}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = "STM/enhance"

    def run(self, tiles, tile_layout, keep_colors=(True,), detail_size=(16.0,)):
        flat = []
        for t in tiles:
            flat += [t[i:i + 1] for i in range(t.shape[0])] if t.ndim == 4 else [t]
        first = lambda v: v[0] if isinstance(v, (list, tuple)) else v
        return (merge_tiles(flat, tile_layout[0], keep_colors=bool(first(keep_colors)), detail_size=float(first(detail_size))),)


class STMTileReferences:
    """One copy of a reference image per tile, turned upside down for the tiles 'STM — Split Tiles' turned upside down
    (cancel_lighting), so the refiner always sees the tile and the reference the same way up - letters and symbols it takes
    from the reference are not drawn upside down."""

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE", {"tooltip": "The reference: the picture the mesh was made from, or any concept art with the details "
                                           "(letters, symbols, ornaments) the texture should get."}),
            "tile_layout": ("STM_TILES",),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("references",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "run"
    CATEGORY = "STM/enhance"

    def run(self, image, tile_layout):
        ref = image[:1].float()
        flipped = torch.rot90(ref, 2, dims=(1, 2)).contiguous()
        out = [flipped if (len(b) > 5 and b[5]) else ref for b in tile_layout.boxes]
        log(f"tile references: {len(out)} ({sum(1 for b in tile_layout.boxes if len(b) > 5 and b[5])} upside down)")
        return (out,)


class STMProjectViews:
    """Writes what the refiner added to the views - detail and new features - back into the texture and returns the mesh
    with that texture.  Light, shadows, colour casts and amplified noise are left out; what no view sees well stays exactly
    as it was."""

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "mesh": ("MESH", {"tooltip": "The same mesh that went into 'STM — Render Views'."}),
            "view_data": ("STM_VIEWS",),
            "refined_views": ("IMAGE", {"tooltip": "The views after the refiner, same order and size (from 'STM — Merge Tiles')."}),
            "detail_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
            "color_detail": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05,
                                       "tooltip": "How much of the refiner's FINEST colour change to take (a pixel or two: colour fringes where it "
                                                  "moved an edge, colour noise). Larger coloured features: color_features."}),
            "broad_changes": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                        "tooltip": "How much of the refiner's broad changes to take (shadows, occlusion, colour shifts). 0 = detail only."}),
            "detail_radius": ("FLOAT", {"default": 16.0, "min": 1.0, "max": 64.0, "step": 0.5,
                                        "tooltip": "In pixels of a 1024 view: changes finer than this are detail and new features, broader ones "
                                                   "are 'broad changes' (shadows, overall colour). 6 = fine detail only (strictest)."}),
            "remove_lighting": ("BOOLEAN", {"default": True, "tooltip": "Remove light the refiner painted in (brightness that follows the surface direction)."}),
            "noise_guard": ("BOOLEAN", {"default": True,
                                        "tooltip": "Drop changes that only amplify faint pattern noise already in the texture (voxel lattices, "
                                                   "compression noise) or relief-shade it: the cause of dot patterns and bumps."}),
            "keep_sharpening": ("BOOLEAN", {"default": True,
                                            "tooltip": "Keep the refiner sharpening real (also soft) edges. Off = only new detail is added, "
                                                       "for the cleanest result on textures that are already sharp."}),
            "color_features": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                         "tooltip": "How much of the colour of features the refiner painted (letters, symbols, ornaments, e.g. "
                                                    "from a reference image) to take. Colour casts and greying of what is there are always "
                                                    "left out. 0 = keep the original hues everywhere."}),
        }, "optional": {
            "base_color": ("IMAGE", {"tooltip": "Only if the same override went into 'STM — Render Views'."}),
        }}

    RETURN_TYPES = ("MESH", "IMAGE", "MASK")
    RETURN_NAMES = ("mesh", "base_color", "detail_mask")
    FUNCTION = "run"
    CATEGORY = "STM/enhance"

    def run(self, mesh, view_data, refined_views, detail_strength, color_detail, broad_changes, detail_radius, remove_lighting,
            noise_guard, keep_sharpening=True, color_features=1.0, base_color=None):
        data = unit_uvs(mesh_data_from_comfy(mesh))
        appearance, factor = mesh_appearance(mesh, base_color)
        dev = pick_device()
        with stage_scope("project views", dev, reserve=int(2 * GiB)):
            new, mask = project_detail(data, appearance, view_data, refined_views.float().cpu(), float(detail_strength),
                                       float(color_detail), float(broad_changes), float(detail_radius), bool(remove_lighting),
                                       bool(noise_guard), dev, keep_sharpening=bool(keep_sharpening), color_features=float(color_features))
        log(f"project views: detail written to {float((mask > 0.5).float().mean()):.0%} of the atlas "
            f"(fading in on {float(((mask > 0) & (mask <= 0.5)).float().mean()):.0%}); the rest is unchanged")
        stored = store_texture(new, factor)
        return (with_texture(mesh, data, stored), stored[None], mask[None])


NODE_CLASS_MAPPINGS = {
    "STM_RenderViews": STMRenderViews, "STM_SplitTiles": STMSplitTiles, "STM_TileReferences": STMTileReferences,
    "STM_MergeTiles": STMMergeTiles, "STM_ProjectViews": STMProjectViews,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "STM_RenderViews": "STM — Render Views (unlit)", "STM_SplitTiles": "STM — Split Tiles",
    "STM_TileReferences": "STM — Tile References", "STM_MergeTiles": "STM — Merge Tiles",
    "STM_ProjectViews": "STM — Project Views (Detail)",
}

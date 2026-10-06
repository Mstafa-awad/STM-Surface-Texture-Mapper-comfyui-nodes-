"""STM - Surface Texture Mapper: modular TRELLIS.2 mesh texturing and texture enhancement for ComfyUI."""
from .apply_texture import node as _apply_texture
from .dino import node as _dino
from .geometry import node as _geometry
from .legacy import node as _legacy
from .load_mesh import node as _load_mesh_file
from .load_models import node as _load_models
from .shape_encoder import node as _shape_encoder
from .texture_decoder import node as _texture_decoder
from .texture_enhance import node as _texture_enhance
from .texture_generation import node as _texture_generation
from .texture_projection import node as _texture_projection
from .uv_unwrap import node as _uv_unwrap

_MODULES = (_load_models, _load_mesh_file, _dino, _uv_unwrap, _geometry, _shape_encoder, _texture_generation, _texture_decoder,
            _texture_projection, _texture_enhance, _apply_texture, _legacy)

NODE_CLASS_MAPPINGS = {k: v for m in _MODULES for k, v in m.NODE_CLASS_MAPPINGS.items()}
NODE_DISPLAY_NAME_MAPPINGS = {k: v for m in _MODULES for k, v in m.NODE_DISPLAY_NAME_MAPPINGS.items()}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

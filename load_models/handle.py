"""The shared handle every model stage receives (ComfyUI type ``TRELLIS2PIPELINE``).

It holds *paths and policy only*; weights are read by the stage that needs them, through one shared ``ModelStore``.
"""
import json
import os
from typing import Dict, Optional, Tuple

import torch

from .. import models
from ..modules.image_feature_extractor import DinoV3FeatureExtractor
from ..runtime.device import pick_device
from ..runtime.store import ModelStore
from ..runtime.vram import Plan

RESOLUTIONS = (512, 1024, 1536)
_DEFAULT_MODELS = {
    "shape_slat_encoder": "ckpts/shape_enc_next_dc_f16c32_fp16",
    "tex_slat_decoder": "ckpts/tex_dec_next_dc_f16c32_fp16",
    "tex_slat_flow_model_512": "ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16",
    "tex_slat_flow_model_1024": "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16",
}
_DEFAULT_SAMPLER = {"name": "FlowEulerGuidanceIntervalSampler", "args": {"sigma_min": 1e-5},
                    "params": {"steps": 12, "guidance_strength": 1.0, "guidance_rescale": 0.0,
                               "guidance_interval": [0.6, 0.9], "rescale_t": 3.0}}


class StmModels:
    def __init__(self, path: str, config: dict, dino_path: str, use_fp8: bool, keep_models_loaded: Optional[bool],
                 device_pref: Optional[str] = None, overrides: Optional[Dict[str, str]] = None):
        self.path = path
        self.config = config
        self.dino_path = dino_path
        self.use_fp8 = use_fp8
        self.keep_models_loaded = keep_models_loaded
        self.device_pref = device_pref
        self.overrides = dict(overrides or {})
        self._store: Optional[ModelStore] = None

    @classmethod
    def standalone(cls, overrides: Optional[Dict[str, str]] = None, dino_path: str = "", keep_models_loaded: Optional[bool] = None,
                   device: Optional[str] = None) -> "StmModels":
        """A handle for a single stage that is given its files directly (no TRELLIS.2 folder needed)."""
        return cls("", {}, dino_path, False, keep_models_loaded, device, overrides)

    @classmethod
    def from_folder(cls, path: str, dino_path: Optional[str] = None, use_fp8: bool = False,
                    keep_models_loaded: Optional[bool] = None, device: Optional[str] = None) -> "StmModels":
        if not os.path.isdir(path):
            raise FileNotFoundError(f"TRELLIS.2 model folder not found: {path}")
        names = (["pipeline_fp8.json"] if use_fp8 else []) + ["texturing_pipeline.json", "pipeline.json"]
        cfg_file = next((os.path.join(path, n) for n in names if os.path.exists(os.path.join(path, n))), None)
        if cfg_file is None:
            raise FileNotFoundError(f"No texturing_pipeline.json / pipeline.json in {path}")
        with open(cfg_file, "r") as f:
            config = json.load(f)["args"]
        if dino_path is None:
            dino_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(path.rstrip("/\\")))), "facebook",
                                     "dinov3-vitl16-pretrain-lvd1689m")
        return cls(path, config, dino_path, use_fp8, keep_models_loaded, device)

    # -------------------------------------------------------------------------------------------- configuration
    @property
    def device(self) -> torch.device:
        return pick_device(self.device_pref)

    def model_path(self, name: str) -> str:
        if name in self.overrides:
            return self.overrides[name]
        if name == "shape_slat_encoder" and self.use_fp8:
            fp8 = os.path.join(self.path, "ckpts_fp8", "shape_enc_next_dc_f16c32_fp8")
            if os.path.exists(fp8 + ".safetensors"):
                return fp8
        rel = self.config.get("models", {}).get(name) or _DEFAULT_MODELS[name]
        return rel if os.path.isabs(rel) else os.path.join(self.path, rel)

    def sampler_spec(self) -> dict:
        return self.config.get("tex_slat_sampler", _DEFAULT_SAMPLER)

    def normalization(self, key: str, device) -> Tuple[torch.Tensor, torch.Tensor]:
        n = self.config[key]
        return (torch.tensor(n["mean"], dtype=torch.float32)[None].to(device), torch.tensor(n["std"], dtype=torch.float32)[None].to(device))

    # -------------------------------------------------------------------------------------------- model access
    def store(self, plan: Plan) -> ModelStore:
        device = self.device
        if self._store is None or self._store.device != device or self._store.keep_in_ram != plan.keep_models_loaded:
            self._store = ModelStore(device, plan.keep_models_loaded)
        return self._store

    def acquire_network(self, name: str, plan: Plan):
        return self.store(plan).acquire(name, lambda: models.from_pretrained(self.model_path(name)))

    def acquire_dino(self, plan: Plan, bf16: bool = False):
        dtype = torch.bfloat16 if bf16 else {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(plan.dino_dtype)
        extractor = self.store(plan).acquire("dino", lambda: DinoV3FeatureExtractor(self.dino_path, image_size=1024, device="cpu", dtype=dtype),
                                             to_device=False)
        if dtype is not None and next(extractor.model.parameters()).dtype != dtype:
            extractor.model.to(dtype)                       # a cached extractor may have been loaded in another precision
        return extractor

    def release(self, name: str, obj, plan: Plan) -> None:
        self.store(plan).release(name, obj)


def coerce_models(handle) -> StmModels:
    """Accept our handle, or the object of the original ComfyUI-Trellis2 loader (only its model folder is reused)."""
    if isinstance(handle, StmModels):
        return handle
    path = getattr(handle, "path", None)
    if not path:
        raise TypeError("Unsupported TRELLIS2PIPELINE input. Connect the 'STM — Load Models' node.")
    key = (os.path.abspath(path), bool(getattr(handle, "use_fp8", False)), getattr(handle, "keep_models_loaded", None))
    if key not in _FOREIGN:
        _FOREIGN[key] = StmModels.from_folder(path, use_fp8=key[1], keep_models_loaded=key[2])
    return _FOREIGN[key]


_FOREIGN: dict = {}

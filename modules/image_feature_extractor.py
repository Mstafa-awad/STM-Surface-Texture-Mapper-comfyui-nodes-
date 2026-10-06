"""DINOv3 image conditioning for TRELLIS.2 (facebook/dinov3-vitl16-pretrain-lvd1689m).

DINOv3 weights are distributed by Meta under the *DINOv3 License*. Commercial use is permitted subject to that license's conditions; see docs/LICENSE_AUDIT.md. The model is loaded from a local folder only.
"""
import os
from typing import *

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class DinoV3FeatureExtractor:
    """Feature extractor for DINOv3 models (local folder, any torch device)."""

    def __init__(self, model_name: str, image_size: int = 1024, device=None, dtype: Optional[torch.dtype] = None):
        from transformers import DINOv3ViTModel          # Apache-2.0
        if not os.path.isdir(model_name):
            raise FileNotFoundError(
                f"DINOv3 folder not found: {model_name}\n"
                "Accept the DINOv3 license on Hugging Face and download facebook/dinov3-vitl16-pretrain-lvd1689m "
                "into ComfyUI/models/facebook/dinov3-vitl16-pretrain-lvd1689m (see docs/INSTALL_WINDOWS.md).")
        self.model_name = model_name
        if dtype is None:
            self.model = DINOv3ViTModel.from_pretrained(model_name)
        else:
            try:                                           # transformers >= 4.56
                self.model = DINOv3ViTModel.from_pretrained(model_name, dtype=dtype)
            except TypeError:                              # older transformers
                self.model = DINOv3ViTModel.from_pretrained(model_name, torch_dtype=dtype)
        self.model.eval()
        self.image_size = image_size
        self.device = torch.device(device) if device is not None else torch.device('cpu')
        self._mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)      # ImageNet statistics (no torchvision needed)
        self._std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.model.to(self.device)

    def to(self, device):
        self.device = torch.device(device)
        self.model.to(self.device)
        return self

    def cpu(self):
        return self.to('cpu')

    def extract_features(self, image: torch.Tensor) -> torch.Tensor:
        image = image.to(self.model.embeddings.patch_embeddings.weight.dtype)
        hidden_states = self.model.embeddings(image, bool_masked_pos=None)
        position_embeddings = self.model.rope_embeddings(image)
        layers = None
        if hasattr(self.model, 'layer'):                                  # transformers < 5
            layers = self.model.layer
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layer'):   # transformers >= 5
            layers = self.model.model.layer
        if layers is None:
            raise RuntimeError("Cannot extract DINOv3 features: unsupported transformers version")
        for layer_module in layers:
            hidden_states = layer_module(hidden_states, position_embeddings=position_embeddings)
        return F.layer_norm(hidden_states, hidden_states.shape[-1:])

    @torch.no_grad()
    def __call__(self, image: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:
        """Return ``[B, N, D]`` patch tokens for a list of PIL images (or a ``[B,3,H,W]`` tensor in 0..1)."""
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((self.image_size, self.image_size), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = torch.stack([torch.from_numpy(i).permute(2, 0, 1).float() for i in image])
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        outs = []
        for i in range(image.shape[0]):                                   # one view at a time: bounded VRAM
            x = (image[i:i + 1].to(self.device) - self._mean.to(self.device)) / self._std.to(self.device)
            outs.append(self.extract_features(x))
        return torch.cat(outs, dim=0)

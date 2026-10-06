"""Model definitions and a memory-lean checkpoint loader.

The original loader built every model in float32 (a 1.3 B-parameter flow model = 5.2 GB of
freshly *initialised* host RAM), then copied the bf16 checkpoint into it.  ``from_pretrained`` here
builds the module skeleton on the ``meta`` device (no allocation, no random init), loads the
safetensors file once and *assigns* the tensors, casting each to the dtype the constructor intended.
If anything is left uninitialised the loader transparently falls back to the original slow path, so
behaviour never differs from the reference loader -- it is only faster and uses ~3x less RAM.

Models are loaded from local files only (``<path>.json`` + ``<path>.safetensors``); nothing is
downloaded automatically.
"""
import contextlib
import importlib
import json
import os

import torch
import torch.nn as nn

__attributes = {
    'SLatFlowModel': 'structured_latent_flow',
    'SparseUnetVaeEncoder': 'sc_vaes.sparse_unet_vae',
    'SparseUnetVaeDecoder': 'sc_vaes.sparse_unet_vae',
    'FlexiDualGridVaeEncoder': 'sc_vaes.fdg_vae',
}
__all__ = list(__attributes.keys()) + ['from_pretrained']


def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module = importlib.import_module(f".{__attributes[name]}", __name__)
            globals()[name] = getattr(module, name)
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]


@contextlib.contextmanager
def init_empty_parameters():
    """Create ``nn.Parameter``s on the meta device (plain tensors / buffers stay real)."""
    original = nn.Module.register_parameter

    def register_parameter(self, name, param):
        original(self, name, param)
        if param is not None:
            p = self._parameters[name]
            if p is not None and p.device.type != 'meta':
                self._parameters[name] = type(p)(p.data.to('meta'), requires_grad=p.requires_grad)

    nn.Module.register_parameter = register_parameter
    try:
        yield
    finally:
        nn.Module.register_parameter = original


def _read_config(path):
    cfg_file, weights = f"{path}.json", f"{path}.safetensors"
    missing = [f for f in (cfg_file, weights) if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(
            "Missing model file(s):\n  " + "\n  ".join(missing) +
            "\nDownload microsoft/TRELLIS.2-4B (MIT) into ComfyUI/models/microsoft/TRELLIS.2-4B "
            "(see docs/INSTALL_WINDOWS.md).")
    with open(cfg_file, 'r') as f:
        return json.load(f), weights


def from_pretrained(path: str, device=None, fast: bool = True, **kwargs):
    """Load ``<path>.json`` / ``<path>.safetensors``; returns an eval-mode module on ``device`` (default CPU)."""
    from safetensors.torch import load_file
    config, weights = _read_config(path)
    cls = __getattr__(config['name'])
    args = config['args']
    # some published encoder configs name the encoder's down blocks 'up_block_type' (see HF discussion #3)
    if 'Encoder' in config['name'] and 'up_block_type' in args and 'down_block_type' not in args:
        args = dict(args)
        args['down_block_type'] = args.pop('up_block_type')
    config = dict(config, args=args)
    model = None
    if fast:
        try:
            with init_empty_parameters():
                model = cls(**config['args'], **kwargs)
            targets = {n: p.dtype for n, p in model.named_parameters()}
            state = load_file(weights, device='cpu')
            model.load_state_dict(state, strict=False, assign=True)
            del state
            left_on_meta = [n for n, p in model.named_parameters() if p.device.type == 'meta']
            left_on_meta += [n for n, b in model.named_buffers() if b.device.type == 'meta']
            if left_on_meta:
                model = None                                   # checkpoint did not cover everything -> reference path
            else:
                for n, p in model.named_parameters():
                    if p.dtype != targets[n]:
                        p.data = p.data.to(targets[n])
        except Exception:
            model = None
    if model is None:
        model = cls(**config['args'], **kwargs)
        model.load_state_dict(load_file(weights, device='cpu'), strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if device is not None:
        model.to(device)
    return model

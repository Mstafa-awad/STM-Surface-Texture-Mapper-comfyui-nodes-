import importlib
import torch
import torch.nn as nn
from .. import config
from .. import SparseTensor

_backends = {}
_flex_gemm_ok = None


def _flex_gemm_available() -> bool:
    global _flex_gemm_ok
    if _flex_gemm_ok is None:
        try:
            importlib.import_module('flex_gemm')        # optional MIT accelerator (Triton); never required
            _flex_gemm_ok = True
        except Exception:
            _flex_gemm_ok = False
    return _flex_gemm_ok


def resolve_backend(device: torch.device) -> str:
    """Pick the conv implementation for tensors living on ``device``."""
    want = config.CONV
    if want == 'torch':
        return 'torch'
    if want == 'flex_gemm':
        if device.type != 'cuda':
            return 'torch'
        return 'flex_gemm' if _flex_gemm_available() else 'torch'
    return 'flex_gemm' if (device.type == 'cuda' and _flex_gemm_available()) else 'torch'


def _backend(name: str):
    if name not in _backends:
        _backends[name] = importlib.import_module(f'..conv_{name}', __name__)
    return _backends[name]


class SparseConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=None, bias=True, indice_key=None):
        super(SparseConv3d, self).__init__()
        # both backends create the identical parameter layout (Co, Kx, Ky, Kz, Ci), so the
        # implementation can be chosen per call from the device of the input.
        _backend('torch').sparse_conv3d_init(self, in_channels, out_channels, kernel_size, stride, dilation, padding, bias, indice_key)

    def forward(self, x: SparseTensor) -> SparseTensor:
        return _backend(resolve_backend(x.feats.device)).sparse_conv3d_forward(self, x)


class SparseInverseConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, indice_key=None):
        super(SparseInverseConv3d, self).__init__()
        _backend('torch').sparse_inverse_conv3d_init(self, in_channels, out_channels, kernel_size, stride, dilation, bias, indice_key)

    def forward(self, x: SparseTensor) -> SparseTensor:
        return _backend('torch').sparse_inverse_conv3d_forward(self, x)

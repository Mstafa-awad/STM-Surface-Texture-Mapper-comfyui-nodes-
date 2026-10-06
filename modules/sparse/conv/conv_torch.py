"""Submanifold sparse 3-D convolution in plain PyTorch.

Drop-in alternative to ``conv_flex_gemm`` for machines where FlexGEMM/Triton is unavailable
(AMD, Intel, Apple, CPU, or Windows without ``triton-windows``).  Weight layout and offset
convention are identical to FlexGEMM and spconv (``(Co, Kx, Ky, Kz, Ci)``, cross-correlation,
``out[p] = sum_k W[k] * in[p + (k - K//2) * dilation]`` along the three coordinate columns), so
the same checkpoints load unchanged.  Verified against dense ``F.conv3d`` in
``tests/test_conv_torch.py``.

Implementation: a neighbour table ``[N, K^3]`` is built once per coordinate set (sorted linear
keys + ``searchsorted``; cached on the SparseTensor like the FlexGEMM backend does), then each
chunk of rows gathers its neighbour features into ``[rows, K^3 * Ci]`` and performs a *single*
GEMM, so accumulation happens in the GEMM's fp32 accumulator and the result is rounded once
(the fp16/bf16 behaviour of the fused kernels).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .. import SparseTensor
from .. import config


def sparse_conv3d_init(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=None, bias=True, indice_key=None):
    assert stride == 1 and (padding is None), 'torch backend only supports submanifold sparse convolution (stride=1, padding=None)'
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = tuple(kernel_size) if isinstance(kernel_size, (list, tuple)) else (kernel_size,) * 3
    self.stride = tuple(stride) if isinstance(stride, (list, tuple)) else (stride,) * 3
    self.dilation = tuple(dilation) if isinstance(dilation, (list, tuple)) else (dilation,) * 3

    weight = torch.empty((out_channels, in_channels, *self.kernel_size))
    if bias:
        self.bias = nn.Parameter(torch.empty(out_channels))
    else:
        self.register_parameter("bias", None)
    torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
    if self.bias is not None:
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weight)
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)
    # (Co, Ci, Kx, Ky, Kz) -> (Co, Kx, Ky, Kz, Ci): same stored layout as the flex_gemm backend / checkpoints
    self.weight = nn.Parameter(weight.permute(0, 2, 3, 4, 1).contiguous())


def build_neighbor_table(coords: torch.Tensor, spatial_shape, kernel_size, dilation) -> torch.Tensor:
    """Return ``[N, Kx*Ky*Kz]`` int32 row indices of the neighbours of every voxel (-1 = absent).

    Column ``v = (ix * Ky + iy) * Kz + iz`` holds the voxel at ``p + ((ix - Kx//2) * dx, ...)``.
    """
    n = coords.shape[0]
    dev = coords.device
    kx, ky, kz = kernel_size
    dx, dy, dz = dilation
    c = coords.to(torch.int64)
    b, x, y, z = c[:, 0], c[:, 1], c[:, 2], c[:, 3]
    sx, sy, sz = (int(s) for s in spatial_shape)
    nb = int(b.max()) + 1 if n else 1
    keys = ((b * sx + x) * sy + y) * sz + z
    sorted_keys, order = torch.sort(keys)
    table = torch.full((n, kx * ky * kz), -1, dtype=torch.int32, device=dev)
    for ix in range(kx):
        for iy in range(ky):
            for iz in range(kz):
                v = (ix * ky + iy) * kz + iz
                ox, oy, oz = (ix - kx // 2) * dx, (iy - ky // 2) * dy, (iz - kz // 2) * dz
                if ox == 0 and oy == 0 and oz == 0:
                    table[:, v] = torch.arange(n, dtype=torch.int32, device=dev)
                    continue
                qx, qy, qz = x + ox, y + oy, z + oz
                inb = (qx >= 0) & (qx < sx) & (qy >= 0) & (qy < sy) & (qz >= 0) & (qz < sz)
                q = ((b * sx + qx.clamp(0, sx - 1)) * sy + qy.clamp(0, sy - 1)) * sz + qz.clamp(0, sz - 1)
                pos = torch.searchsorted(sorted_keys, q).clamp_(max=n - 1)
                hit = inb & (sorted_keys[pos] == q)
                table[:, v] = torch.where(hit, order[pos].to(torch.int32), torch.full_like(pos, -1, dtype=torch.int32))
    return table


def _conv_with_table(feats: torch.Tensor, table: torch.Tensor, weight: torch.Tensor, bias, chunk_bytes: int) -> torch.Tensor:
    n, ci = feats.shape
    co = weight.shape[0]
    v = table.shape[1]
    w2 = weight.reshape(co, v * ci).t().contiguous()                      # [V*Ci, Co]
    pad = torch.zeros((1, ci), dtype=feats.dtype, device=feats.device)
    fpad = torch.cat([feats, pad], dim=0)                                 # row n == zeros (absent neighbour)
    rows = max(1, int(chunk_bytes // max(1, v * ci * feats.element_size())))
    out = torch.empty((n, co), dtype=feats.dtype, device=feats.device)
    for s in range(0, n, rows):
        t = table[s:s + rows].to(torch.int64)
        t = torch.where(t < 0, torch.full_like(t, n), t)
        g = fpad[t].reshape(t.shape[0], v * ci)                           # [rows, V*Ci]
        o = g @ w2
        if bias is not None:
            o = o + bias
        out[s:s + rows] = o
    return out


def sparse_conv3d_forward(self, x: SparseTensor) -> SparseTensor:
    Co, Kx, Ky, Kz, Ci = self.weight.shape
    key = f'SubMConv3d_torch_neighbor_{Kx}x{Ky}x{Kz}_dilation{self.dilation}'
    table = x.get_spatial_cache(key)
    if table is None:
        table = build_neighbor_table(x.coords, x.spatial_shape, (Kx, Ky, Kz), self.dilation)
        x.register_spatial_cache(key, table)
    feats = x.feats
    weight = self.weight if self.weight.dtype == feats.dtype else self.weight.to(feats.dtype)
    bias = self.bias if (self.bias is None or self.bias.dtype == feats.dtype) else self.bias.to(feats.dtype)
    out = _conv_with_table(feats, table, weight, bias, int(config.TORCH_CONV_CHUNK_BYTES))
    return x.replace(out)


def sparse_inverse_conv3d_init(self, *args, **kwargs):
    raise NotImplementedError('SparseInverseConv3d is not implemented for the torch backend')


def sparse_inverse_conv3d_forward(self, x: SparseTensor) -> SparseTensor:
    raise NotImplementedError('SparseInverseConv3d is not implemented for the torch backend')

"""Backends of the sparse modules.

* ``CONV``: ``'auto'`` (default) uses FlexGEMM (MIT, Triton) on CUDA when it is importable and the
  built-in pure-PyTorch implementation everywhere else; ``'flex_gemm'`` / ``'torch'`` force one.
  Override with the environment variable ``TRELLIS2_SPARSE_CONV``.
* ``ATTN``: ``'auto'`` (default) uses PyTorch ``scaled_dot_product_attention``; ``'flash_attn'`` or
  ``'xformers'`` are opt-in via ``TRELLIS2_ATTN``.
"""
import os
from typing import *

CONV = 'auto'
DEBUG = False
ATTN = 'sdpa'
TORCH_CONV_CHUNK_BYTES = 1 << 28                     # gather buffer of the torch conv backend (256 MiB)


def _from_env():
    global CONV, DEBUG, ATTN, TORCH_CONV_CHUNK_BYTES
    conv = os.environ.get('TRELLIS2_SPARSE_CONV') or os.environ.get('SPARSE_CONV_BACKEND')
    if conv in ('auto', 'flex_gemm', 'torch'):
        CONV = conv
    if os.environ.get('SPARSE_DEBUG') is not None:
        DEBUG = os.environ.get('SPARSE_DEBUG') == '1'
    attn = os.environ.get('TRELLIS2_ATTN') or os.environ.get('SPARSE_ATTN_BACKEND') or os.environ.get('ATTN_BACKEND')
    if attn and attn != 'auto' and attn in ('xformers', 'flash_attn', 'flash_attn_3', 'sdpa'):
        ATTN = attn
    chunk = os.environ.get('TRELLIS2_TORCH_CONV_CHUNK_MB')
    if chunk and chunk.isdigit():
        TORCH_CONV_CHUNK_BYTES = int(chunk) << 20


_from_env()


def set_conv_backend(backend: Literal['auto', 'flex_gemm', 'torch']):
    global CONV
    CONV = backend


def set_debug(debug: bool):
    global DEBUG
    DEBUG = debug


def set_attn_backend(backend: Literal['xformers', 'flash_attn', 'flash_attn_3', 'sdpa']):
    global ATTN
    ATTN = backend

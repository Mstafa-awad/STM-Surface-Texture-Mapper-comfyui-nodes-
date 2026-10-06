"""Attention backend selection (dense attention).

``TRELLIS2_ATTN`` / ``ATTN_BACKEND`` may be ``auto`` (default), ``flash_attn``, ``xformers``,
``sdpa`` or ``naive``.  ``auto`` uses PyTorch's built-in ``scaled_dot_product_attention``
(``sdpa``): it ships with PyTorch, has fused flash / memory-efficient kernels on CUDA, ROCm and
XPU, and needs no extra wheel.  ``flash_attn`` / ``xformers`` (both BSD-3-Clause) are only used
when explicitly requested.
"""
import os
from typing import *

BACKEND = 'sdpa'
DEBUG = False


def _from_env():
    global BACKEND, DEBUG
    env = os.environ.get('TRELLIS2_ATTN') or os.environ.get('ATTN_BACKEND')
    if env and env != 'auto' and env in ['xformers', 'flash_attn', 'flash_attn_3', 'sdpa', 'naive']:
        BACKEND = env
    if os.environ.get('ATTN_DEBUG') is not None:
        DEBUG = os.environ.get('ATTN_DEBUG') == '1'


_from_env()


def set_backend(backend: Literal['xformers', 'flash_attn', 'flash_attn_3', 'sdpa', 'naive']):
    global BACKEND
    BACKEND = backend


def set_debug(debug: bool):
    global DEBUG
    DEBUG = debug

"""Chunk-size control and OOM retry for the sparse VAEs."""
from typing import Callable

import torch.nn as nn

from ..modules.sparse import config as sparse_config
from ..runtime.device import StmMemoryError, is_oom, soft_empty_cache
from ..runtime.stage import log
from ..runtime.vram import Plan


def set_chunk_size(module: nn.Module, n: int, low_vram: bool) -> None:
    for m in module.modules():
        if hasattr(m, "chunk_size"):
            m.chunk_size = int(n)
    if hasattr(module, "low_vram"):
        module.low_vram = bool(low_vram)


def configure_vae(model: nn.Module, plan: Plan) -> None:
    set_chunk_size(model, plan.sampler_chunk_size, plan.tier in ("low", "8GB", "12GB", "cpu"))


def run_chunked(fn: Callable[[], object], model: nn.Module, plan: Plan, what: str):
    """Run ``fn``; on out-of-memory halve every chunk knob and retry (5 attempts), then raise :class:`StmMemoryError`."""
    last = None
    previous_backend = sparse_config.CONV
    try:
        for _ in range(5):
            try:
                return fn()
            except BaseException as e:                     # noqa: BLE001
                if not is_oom(e):
                    raise
                last = e
                soft_empty_cache()
                plan.shrink()
                set_chunk_size(model, plan.sampler_chunk_size, True)
                note = ""
                if sparse_config.CONV != "torch":          # FlexGEMM builds whole-tensor neighbour tables: use the chunked PyTorch conv
                    sparse_config.CONV = "torch"
                    note = "; sparse convolutions switched to the PyTorch backend"
                log(f"{what}: out of memory, retrying with chunk_size={plan.sampler_chunk_size}{note}")
        raise StmMemoryError(f"{what} ran out of memory even with the smallest chunk size. Use resolution 1024/512, "
                             f"the fp8 model files, or close other GPU applications.") from last
    finally:
        sparse_config.CONV = previous_backend

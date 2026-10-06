"""Stage scope: memory hand-over between nodes, timing and memory log.

Each stage starts and ends by returning cached allocator blocks and asks ComfyUI to unload its own models up to the memory the
stage is expected to need.  No allocator cap is applied: ComfyUI's dynamic-VRAM manager already handles memory pressure, and a cap
below the physical VRAM made large stages fail that otherwise fit.
"""
import contextlib
import time

import torch

from .device import make_room, soft_empty_cache

GiB = 1024 ** 3


def log(msg: str) -> None:
    print(f"[STM] {msg}", flush=True)


@contextlib.contextmanager
def stage_scope(name: str, device: torch.device, reserve: int = 0):
    """``reserve``: bytes this stage is expected to need; ComfyUI is asked to unload its own models to make that much room."""
    soft_empty_cache()                           # raw CUDA allocations (FlexGEMM) cannot use blocks cached by the previous stage
    if reserve > 0 and device.type != "cpu":
        make_room(int(reserve), device)
    cuda = device.type == "cuda"
    free_at_start = 0.0
    if cuda:
        free_at_start = torch.cuda.mem_get_info(device)[0] / GiB
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    try:
        yield
    finally:
        info = f", peak VRAM {torch.cuda.max_memory_allocated(device) / GiB:.1f} GiB (free at start {free_at_start:.1f} GiB)" if cuda else ""
        log(f"{name}: {time.time() - t0:.1f}s{info}")
        soft_empty_cache()                       # stage boundary: hand everything back before the next node runs

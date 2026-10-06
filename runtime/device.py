"""Device, memory and ComfyUI integration helpers.

Everything here degrades gracefully when ComfyUI (``comfy.model_management``) or ``psutil`` is not
importable, so the baking / geometry code can be unit-tested outside ComfyUI.
"""
from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

GiB = 1024 ** 3


class StmMemoryError(RuntimeError):
    """Raised when a stage cannot fit in memory even after all automatic fallbacks."""


def _comfy_mm():
    try:
        import comfy.model_management as mm            # only available inside ComfyUI
        return mm
    except Exception:
        return None


def pick_device(prefer: Optional[str] = None) -> torch.device:
    """ComfyUI's device if available, else CUDA/XPU/MPS, else CPU. ``prefer='cpu'`` forces CPU."""
    if prefer and prefer != "auto":
        return torch.device(prefer)
    mm = _comfy_mm()
    if mm is not None:
        try:
            return torch.device(mm.get_torch_device())
        except Exception:
            pass
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class MemInfo:
    kind: str                 # 'cuda' | 'xpu' | 'mps' | 'cpu'
    name: str
    total: int                # bytes
    free: int                 # bytes (device memory, or system RAM for cpu/mps)

    @property
    def total_gb(self) -> float:
        return self.total / GiB

    @property
    def free_gb(self) -> float:
        return self.free / GiB


def system_ram() -> Tuple[int, int]:
    """(available, total) system RAM in bytes."""
    try:
        import psutil                                    # BSD-3, part of ComfyUI's requirements
        vm = psutil.virtual_memory()
        return int(vm.available), int(vm.total)
    except Exception:
        try:
            pages = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
            avail = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
            return int(avail), int(pages)
        except Exception:
            return 8 * GiB, 16 * GiB


def device_memory(device: torch.device) -> MemInfo:
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        return MemInfo("cuda", torch.cuda.get_device_name(device), int(total), int(free))
    if device.type == "xpu":
        try:
            free, total = torch.xpu.mem_get_info(device)
            return MemInfo("xpu", torch.xpu.get_device_name(device), int(total), int(free))
        except Exception:
            pass
    avail, total = system_ram()
    return MemInfo(device.type, "system RAM", total, avail)


def soft_empty_cache() -> None:
    """Release cached allocator blocks (call after freeing large tensors, not in hot loops)."""
    gc.collect()
    mm = _comfy_mm()
    if mm is not None:
        try:
            mm.soft_empty_cache()
            return
        except Exception:
            pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def make_room(nbytes: int, device: torch.device) -> None:
    """Ask ComfyUI to unload other models so that ``nbytes`` of VRAM are free."""
    mm = _comfy_mm()
    if mm is not None and device.type != "cpu":
        try:
            mm.free_memory(int(nbytes), device)
        except Exception:
            pass


# "allocation on device" is what ComfyUI's dynamic-VRAM allocator (comfy-aimdo) raises instead of torch.OutOfMemoryError
_OOM_MARKERS = ("out of memory", "allocation on device", "failed to allocate", "can't allocate memory", "cannot allocate memory",
                "not enough memory", "alloc_cpu", "bad_alloc", "cuda_error_out_of_memory", "cublas_status_alloc_failed")


def is_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return any(s in msg for s in _OOM_MARKERS)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)

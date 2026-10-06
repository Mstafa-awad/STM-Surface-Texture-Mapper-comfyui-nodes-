"""One-model-at-a-time storage shared by every stage node.

Each stage acquires the model it needs, runs, and releases it before the next stage loads its own: only the model of the
currently executing stage is resident on the GPU.  Released models are dropped, or parked in system RAM when the machine has
plenty of it (``Plan.keep_models_loaded``).
"""
import time
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn

from .device import make_room, soft_empty_cache
from .stage import log


def _nbytes(model: nn.Module) -> int:
    return int(sum(p.numel() * p.element_size() for p in model.parameters()))


class ModelStore:
    def __init__(self, device: torch.device, keep_in_ram: bool):
        self.device = device
        self.keep_in_ram = keep_in_ram
        self._ram: Dict[str, object] = {}

    def acquire(self, name: str, loader: Callable[[], object], to_device: bool = True):
        obj = self._ram.pop(name, None)
        if obj is None:
            t0 = time.time()
            obj = loader()
            log(f"loaded {name} in {time.time() - t0:.1f}s")
        if to_device and self.device.type != "cpu":
            size = _nbytes(obj) if isinstance(obj, nn.Module) else 0
            make_room(int(size * 1.3) + (512 << 20), self.device)      # ask ComfyUI to unload its own models first
            obj.to(self.device)
        return obj

    def release(self, name: str, obj, keep: Optional[bool] = None) -> None:
        keep = self.keep_in_ram if keep is None else keep
        if keep:
            obj.to("cpu")
            self._ram[name] = obj
        else:
            module = obj if isinstance(obj, nn.Module) else getattr(obj, "model", None)
            if isinstance(module, nn.Module):
                module.to("meta")                          # frees the weights now, whatever else still references the module
        soft_empty_cache()

    def clear(self) -> None:
        self._ram.clear()
        soft_empty_cache()

"""Device selection helpers shared by the inference paths.

Detection order is XPU, then CUDA, then CPU. An explicit device always wins and
is never swapped for another backend.
"""

from typing import Union

import torch

DeviceLike = Union[str, torch.device, None]

_DIST_BACKENDS = {"cuda": "nccl", "xpu": "xccl", "cpu": "gloo"}


def is_available(device_type: str) -> bool:
    """Whether ``device_type`` is usable in this torch build and on this host."""
    if device_type == "xpu":
        return hasattr(torch, "xpu") and torch.xpu.is_available()
    if device_type == "cuda":
        return torch.cuda.is_available()
    return device_type == "cpu"


def default_device_type() -> str:
    """Best available backend: ``xpu`` > ``cuda`` > ``cpu``."""
    for device_type in ("xpu", "cuda"):
        if is_available(device_type):
            return device_type
    return "cpu"


def resolve_device(device: DeviceLike = None) -> torch.device:
    """Resolve ``None`` to the first device of the best backend, else honour ``device``."""
    if device is None:
        device_type = default_device_type()
        return torch.device(device_type, 0) if device_type != "cpu" else torch.device("cpu")
    device = torch.device(device)
    if device.type in ("xpu", "cuda") and device.index is None:
        device = torch.device(device.type, 0)
    return device


def set_device(device: Union[str, torch.device, int]) -> None:
    """Make ``device`` the current device of its backend; no-op on CPU."""
    if isinstance(device, int):
        device = torch.device(default_device_type(), device)
    device = torch.device(device)
    if device.type == "xpu":
        torch.xpu.set_device(device)
    elif device.type == "cuda":
        torch.cuda.set_device(device)


def dist_backend(device_type: str = None) -> str:
    """Collective backend for the backend: nccl for CUDA, xccl for XPU, gloo otherwise."""
    return _DIST_BACKENDS[device_type or default_device_type()]

"""Shared torch device and dtype resolution for reranker wrappers."""

from __future__ import annotations


def resolve_device(device: str) -> str:
    """Resolve reranker ``device: auto`` using the CUDA > MPS > CPU policy."""
    if device != "auto":
        return device

    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(dtype: str, device: str):
    """Resolve ``dtype: auto`` to BF16/FP16 on CUDA and FP32 elsewhere."""
    import torch

    named = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
    }
    if dtype != "auto":
        key = dtype.lower()
        if key not in named:
            raise ValueError(f"Unknown dtype={dtype!r}; allowed: auto + {sorted(named)}")
        return named[key]

    if device.startswith("cuda") and torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        return torch.bfloat16 if cap[0] >= 8 else torch.float16
    return torch.float32

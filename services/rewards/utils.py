from __future__ import annotations

from typing import Any

import torch


def _parse_device(value: Any) -> torch.device | None:
    if value is None:
        return None
    if isinstance(value, torch.device):
        return value
    return torch.device(str(value))


def _parse_dtype(value: Any) -> torch.dtype | None:
    if value is None:
        return None
    if isinstance(value, torch.dtype):
        return value
    text = str(value).lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    return mapping.get(text)

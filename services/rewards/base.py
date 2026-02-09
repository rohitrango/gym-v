from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch


class BaseReward(ABC):
    def __init__(self, device: torch.device | str = "cpu", **kwargs: Any) -> None:
        self.device = torch.device(device) if isinstance(device, str) else device

    @abstractmethod
    def __call__(self, samples):
        raise NotImplementedError

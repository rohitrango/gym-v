"""
Reward API.

- Individual reward implementations live in `gym_v/rewards`.
- `build_rm_model_manager(...)` builds a callable that computes per-reward results.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from typing import Any

from services.rewards.base import BaseReward
from services.rewards.geneval.geneval_reward import GenevalReward
from services.rewards.qwen3vl.qwen3vl_reward import Qwen3VLReward
from services.rewards.registry import REWARD_REGISTRY, register_reward

_REWARD_MODULES = (
    "services.rewards.geneval.geneval_reward",
    "services.rewards.qwen3vl.qwen3vl_reward",
)


@lru_cache(maxsize=1)
def _auto_register_rewards() -> None:
    for module in _REWARD_MODULES:
        import_module(module)


def build_local_reward(name: str, **kwargs: Any) -> BaseReward:
    _auto_register_rewards()
    return REWARD_REGISTRY[name](**kwargs)


@dataclass(frozen=True)
class _RewardComponent:
    name: str
    reward: BaseReward


class RMModelManager:
    """Callable that computes multiple rewards and returns per-reward results.

    `init_kwargs` supports:
    - `{"geneval": {}}` (defaults)
    - `{"geneval": {"torch_device": "cuda:0", "torch_dtype": "float16"}}`
    - `{"clipscore": {"model_name_or_path": "...", "processor_name_or_path": "..."}}`
    - list-of-dicts per reward: `{"qwen3vl": [{...}, {...}]}`
    """

    def __init__(self, init_kwargs: Mapping[str, Any]) -> None:
        self.components: list[_RewardComponent] = []
        for name, spec in init_kwargs.items():
            specs = spec if isinstance(spec, list | tuple) else [spec]
            for idx, item in enumerate(specs):
                reward = build_local_reward(name, **dict(item))
                comp_name = name if len(specs) == 1 else f"{name}_{idx}"
                self.components.append(_RewardComponent(name=comp_name, reward=reward))

    def __call__(self, samples):
        score_details: dict[str, Any] = {}

        for component in self.components:
            name = component.name
            result = component.reward(samples)
            score_details[name] = result
        return score_details, {}

    def get_model_info(self) -> dict[str, Any]:
        return {
            "rewards": [
                {"name": component.name, "device": str(component.reward.device)}
                for component in self.components
            ],
        }


def build_rm_model_manager(init_kwargs: Mapping[str, Any]) -> RMModelManager:
    """Alias for building the reward model manager."""
    return RMModelManager(init_kwargs=init_kwargs)


__all__ = [
    "GenevalReward",
    "Qwen3VLReward",
    "RMModelManager",
    "build_rm_model_manager",
    "build_local_reward",
    "register_reward",
    "REWARD_REGISTRY",
]

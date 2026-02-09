from __future__ import annotations

from collections.abc import Callable

from services.rewards.base import BaseReward

REWARD_REGISTRY: dict[str, Callable[..., BaseReward]] = {}


def register_reward(name: str):
    def _decorator(factory: Callable[..., BaseReward]):
        REWARD_REGISTRY[name] = factory
        return factory

    return _decorator

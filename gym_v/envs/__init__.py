"""Registers the internal gym-v envs then loads the env plugins for module using the entry point."""

from typing import Any

from gym_v.envs.registration import make, pprint_registry, register, registry, spec

register(
    id="TextArena/Sokoban-v0",
    entry_point="gym_v.envs.textarena.sokoban:TextArenaSokobanEnv",
    max_episode_steps=100,
    kwargs=dict(
        dim_room=(6, 6),
        num_boxes=3,
        tile_size=48,
    ),
)

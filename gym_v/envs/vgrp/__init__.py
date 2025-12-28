"""VGRP-Bench puzzle environments for gym-v."""

from gym_v.envs.vgrp.battleships import VGRPBattleshipsEnv
from gym_v.envs.vgrp.binairo import VGRPBinairoEnv
from gym_v.envs.vgrp.thermometers import VGRPThermometersEnv
from gym_v.envs.vgrp.treesandtents import VGRPTreesAndTentsEnv

__all__ = [
    "VGRPBinairoEnv",
    "VGRPThermometersEnv",
    "VGRPTreesAndTentsEnv",
    "VGRPBattleshipsEnv",
]

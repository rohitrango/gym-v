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
from gym_v.envs.vgrp.fieldexplore import VGRPFieldExploreEnv
from gym_v.envs.vgrp.futoshiki import VGRPFutoshikiEnv
from gym_v.envs.vgrp.hitori import VGRPHitoriEnv
from gym_v.envs.vgrp.renzoku import VGRPRenzokuEnv
from gym_v.envs.vgrp.starbattle import VGRPStarBattleEnv

__all__ += [
    "VGRPRenzokuEnv",
    "VGRPFieldExploreEnv",
    "VGRPFutoshikiEnv",
    "VGRPHitoriEnv",
    "VGRPStarBattleEnv",
]

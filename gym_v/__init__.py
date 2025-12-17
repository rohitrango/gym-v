from importlib.metadata import version

from gym_v import utils
from gym_v.core import Env, Observation
from gym_v.envs.registration import (
    make,
    pprint_registry,
    register,
    register_envs,
    registry,
    spec,
)
from gym_v.logger import get_logger, set_level

__version__ = version("gym-v")

__all__ = [
    "Observation",
    "Env",
    "get_logger",
    "set_level",
    "make",
    "spec",
    "register",
    "registry",
    "pprint_registry",
    "register_envs",
    "utils",
]

from __future__ import annotations

from copy import deepcopy
import inspect
from typing import Any

import numpy as np
from PIL import Image

from gym_v import Observation
from gym_v.logger import get_logger

logger = get_logger()


def np_random(seed: int | None = None) -> tuple[np.random.Generator, int]:
    """Returns a NumPy random number generator (RNG) along with seed value from the inputted seed."""
    if seed is not None and not (isinstance(seed, int) and 0 <= seed):
        if isinstance(seed, int) is False:
            raise ValueError(
                f"Seed must be a python integer, actual type: {type(seed)}"
            )
        else:
            raise ValueError(
                f"Seed must be greater or equal to zero, actual value: {seed}"
            )

    seed_seq = np.random.SeedSequence(seed)
    np_seed = seed_seq.entropy
    rng = np.random.Generator(np.random.PCG64(seed_seq))
    return rng, np_seed


class RecordConstructorArgs:
    """Records all arguments passed to constructor to `_saved_kwargs`."""

    def __init__(self, *, _disable_deepcopy: bool = False, **kwargs: Any):
        """Records all arguments passed to constructor to `_saved_kwargs`."""
        # See class docstring for explanation
        if not hasattr(self, "_saved_kwargs"):
            if _disable_deepcopy is False:
                kwargs = deepcopy(kwargs)
            self._saved_kwargs: dict[str, Any] = kwargs


def check_obs(obs, method_name: str):
    """Check that the observation returned by the environment correspond to the declared one.

    Args:
        obs: The observation to check
        method_name: The method name that generated the observation
    """
    pre = f"The obs returned by the `{method_name}()` method"
    if not isinstance(obs, Observation):
        logger.warn(f"{pre} should be an Observation, actual type: {type(obs)}")


def env_step_passive_checker(env, action):
    """A passive check for the environment step, investigating the returning data then returning the data unchanged."""
    # We don't check the action as for some environments then out-of-bounds values can be given
    result = env.step(action)
    assert isinstance(
        result, tuple
    ), f"Expects step result to be a tuple, actual type: {type(result)}"
    if len(result) == 4:
        logger.deprecation(
            "Core environment is written in old step API which returns one bool instead of two. "
            "It is recommended to rewrite the environment with new step API. "
        )
        obs, reward, done, info = result

        if not isinstance(done, bool | np.bool_):
            logger.warn(
                f"Expects `done` signal to be a boolean, actual type: {type(done)}"
            )
    elif len(result) == 5:
        obs, reward, terminated, truncated, info = result

        # np.bool is actual python bool not np boolean type, therefore bool_ or bool8
        if not isinstance(terminated, bool | np.bool_):
            logger.warn(
                f"Expects `terminated` signal to be a boolean, actual type: {type(terminated)}"
            )
        if not isinstance(truncated, bool | np.bool_):
            logger.warn(
                f"Expects `truncated` signal to be a boolean, actual type: {type(truncated)}"
            )
    else:
        raise RuntimeError(
            f"Expected `Env.step` to return a four or five element tuple, actual number of elements returned: {len(result)}."
        )

    check_obs(obs, "step")

    if not (
        np.issubdtype(type(reward), np.integer)
        or np.issubdtype(type(reward), np.floating)
    ):
        logger.warn(
            f"The reward returned by `step()` must be a float, int, np.integer or np.floating, actual type: {type(reward)}"
        )
    else:
        if np.isnan(reward):
            logger.warn("The reward is a NaN value.")
        if np.isinf(reward):
            logger.warn("The reward is an inf value.")

    assert isinstance(
        info, dict
    ), f"The `info` returned by `step()` must be a python dictionary, actual type: {type(info)}"

    return result


def env_reset_passive_checker(env, **kwargs):
    """A passive check of the `Env.reset` function investigating the returning reset information and returning the data unchanged."""
    signature = inspect.signature(env.reset)
    if "seed" not in signature.parameters and "kwargs" not in signature.parameters:
        logger.deprecation(
            "Current gymnasium version requires that `Env.reset` can be passed a `seed` instead of using `Env.seed` for resetting the environment random number generator."
        )
    else:
        seed_param = signature.parameters.get("seed")
        # Check the default value is None
        if seed_param is not None and seed_param.default is not None:
            logger.warn(
                "The default seed argument in `Env.reset` should be `None`, otherwise the environment will by default always be deterministic. "
                f"Actual default: {seed_param}"
            )

    if "options" not in signature.parameters and "kwargs" not in signature.parameters:
        logger.deprecation(
            "Current gymnasium version requires that `Env.reset` can be passed `options` to allow the environment initialisation to be passed additional information."
        )

    # Checks the result of env.reset with kwargs
    result = env.reset(**kwargs)

    if not isinstance(result, tuple):
        logger.warn(
            f"The result returned by `env.reset()` was not a tuple of the form `(obs, info)`, where `obs` is a observation and `info` is a dictionary containing additional information. Actual type: `{type(result)}`"
        )
    elif len(result) != 2:
        logger.warn(
            "The result returned by `env.reset()` should be `(obs, info)` by default, , where `obs` is a observation and `info` is a dictionary containing additional information."
        )
    else:
        obs, info = result
        check_obs(obs, "reset")
        assert isinstance(
            info, dict
        ), f"The second element returned by `env.reset()` was not a dictionary, actual type: {type(info)}"
    return result


def _check_render_return(render_mode, render_return):
    """Produces warning if `render_return` doesn't match `render_mode`."""

    if not isinstance(render_return, Image.Image):
        logger.warn(
            f"Render return should be an PIL Image, actual type: {type(render_return)}"
        )


def env_render_passive_checker(env):
    """A passive check of the `Env.render` that the declared render modes/fps in the metadata of the environment is declared."""

    result = env.render()
    if env.render_mode is not None:
        _check_render_return(env.render_mode, result)

    return result

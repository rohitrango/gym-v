"""A set of functions for passively checking environment implementations."""

import inspect

import numpy as np
from PIL import Image

from gym_v.core import Observation
from gym_v.logger import get_logger

logger = get_logger()


def check_obs(obs_dict, method_name: str):
    """Check that the observation returned by the environment correspond to the declared one.

    Args:
        obs_dict: The observation dictionary to check
        method_name: The method name that generated the observation
    """
    pre = f"The obs returned by the `{method_name}()` method"
    if not isinstance(obs_dict, dict):
        logger.warn(f"{pre} should be a dictionary, actual type: {type(obs_dict)}")
        return

    for agent_id, obs in obs_dict.items():
        if not isinstance(obs, Observation):
            logger.warn(
                f"{pre} for agent '{agent_id}' should be an Observation, actual type: {type(obs)}"
            )


def env_step_passive_checker(env, action):
    """A passive check for the environment step, investigating the returning data then returning the data unchanged."""
    result = env.step(action)
    assert isinstance(
        result, tuple
    ), f"Expects step result to be a tuple, actual type: {type(result)}"

    if len(result) != 5:
        raise RuntimeError(
            f"Expected `Env.step` to return a five element tuple, actual number of elements returned: {len(result)}."
        )

    obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict = result

    check_obs(obs_dict, "step")

    if not isinstance(reward_dict, dict):
        logger.warn(
            f"Expects `reward` to be a dictionary, actual type: {type(reward_dict)}"
        )
    else:
        for agent_id, reward in reward_dict.items():
            if not (
                np.issubdtype(type(reward), np.integer)
                or np.issubdtype(type(reward), np.floating)
            ):
                logger.warn(
                    f"The reward for agent '{agent_id}' must be a float/int, actual type: {type(reward)}"
                )
            elif np.isnan(reward):
                logger.warn(f"The reward for agent '{agent_id}' is a NaN value.")
            elif np.isinf(reward):
                logger.warn(f"The reward for agent '{agent_id}' is an inf value.")

    if not isinstance(terminated_dict, dict):
        logger.warn(
            f"Expects `terminated` to be a dictionary, actual type: {type(terminated_dict)}"
        )
    else:
        for agent_id, term in terminated_dict.items():
            if not isinstance(term, bool | np.bool_):
                logger.warn(
                    f"Expects `terminated` for agent '{agent_id}' to be a boolean, actual type: {type(term)}"
                )

    if not isinstance(truncated_dict, dict):
        logger.warn(
            f"Expects `truncated` to be a dictionary, actual type: {type(truncated_dict)}"
        )
    else:
        for agent_id, trunc in truncated_dict.items():
            if not isinstance(trunc, bool | np.bool_):
                logger.warn(
                    f"Expects `truncated` for agent '{agent_id}' to be a boolean, actual type: {type(trunc)}"
                )

    assert isinstance(
        info_dict, dict
    ), f"The `info` returned by `step()` must be a dictionary, actual type: {type(info_dict)}"

    return result


def env_reset_passive_checker(env, **kwargs):
    """A passive check of the `Env.reset` function investigating the returning reset information and returning the data unchanged."""
    signature = inspect.signature(env.reset)
    if "seed" not in signature.parameters and "kwargs" not in signature.parameters:
        logger.deprecation(
            "Current gym_v version requires that `Env.reset` can be passed a `seed` instead of using `Env.seed` for resetting the environment random number generator."
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
            "Current gym_v version requires that `Env.reset` can be passed `options` to allow the environment initialisation to be passed additional information."
        )

    # Checks the result of env.reset with kwargs
    result = env.reset(**kwargs)

    if not isinstance(result, tuple):
        logger.warn(
            f"The result returned by `env.reset()` was not a tuple of the form `(obs_dict, info_dict)`. Actual type: `{type(result)}`"
        )
    elif len(result) != 2:
        logger.warn(
            "The result returned by `env.reset()` should be `(obs_dict, info_dict)` by default."
        )
    else:
        obs_dict, info = result
        check_obs(obs_dict, "reset")
        assert isinstance(
            info, dict
        ), f"The second element returned by `env.reset()` was not a dictionary, actual type: {type(info)}"
    return result


def _check_render_return(render_return):
    """Produces warning if `render_return` doesn't match the expected type."""

    if not isinstance(render_return, Image.Image):
        logger.warn(
            f"Render return should be an PIL Image, actual type: {type(render_return)}"
        )


def env_render_passive_checker(env):
    """A passive check for the environment render, investigating the returning data then returning the data unchanged."""

    result = env.render()
    _check_render_return(result)

    return result

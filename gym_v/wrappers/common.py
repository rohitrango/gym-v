from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, SupportsFloat

from PIL import Image

from gym_v import Env, Observation, Wrapper
from gym_v.logger import get_logger
from gym_v.utils import (
    RecordConstructorArgs,
    env_render_passive_checker,
    env_reset_passive_checker,
    env_step_passive_checker,
)

if TYPE_CHECKING:
    from gymnasium.envs.registration import EnvSpec


logger = get_logger()


class PassiveEnvChecker(Wrapper, RecordConstructorArgs):
    """A passive wrapper that surrounds the ``step``, ``reset`` and ``render`` functions to check they follow Gymnasium's API.

    This wrapper is automatically applied during make and can be disabled with `disable_env_checker`.
    """

    def __init__(self, env: Env):
        """Initialises the wrapper with the environments, run the observation and action space tests."""
        RecordConstructorArgs.__init__(self)
        Wrapper.__init__(self, env)

        if not isinstance(env, Env):
            raise TypeError(
                f"The environment must inherit from the gym_v.Env class, actual class: {type(env)}. "
            )

        self.checked_reset: bool = False
        self.checked_step: bool = False
        self.checked_render: bool = False
        self.close_called: bool = False

    def step(
        self, action: str
    ) -> tuple[Observation, SupportsFloat, bool, bool, dict[str, Any]]:
        """Steps through the environment that on the first call will run the `passive_env_step_check`."""
        if self.checked_step is False:
            self.checked_step = True
            return env_step_passive_checker(self.env, action)
        else:
            return self.env.step(action)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        """Resets the environment that on the first call will run the `passive_env_reset_check`."""
        if self.checked_reset is False:
            self.checked_reset = True
            return env_reset_passive_checker(self.env, seed=seed, options=options)
        else:
            return self.env.reset(seed=seed, options=options)

    def render(self) -> Image.Image | list[Image.Image] | None:
        """Renders the environment that on the first call will run the `passive_env_render_check`."""
        if self.checked_render is False:
            self.checked_render = True
            return env_render_passive_checker(self.env)
        else:
            return self.env.render()

    @property
    def spec(self) -> EnvSpec | None:
        """Modifies the environment spec to such that `disable_env_checker=False`."""
        if self._cached_spec is not None:
            return self._cached_spec

        env_spec = self.env.spec
        if env_spec is not None:
            try:
                env_spec = deepcopy(env_spec)
                env_spec.disable_env_checker = False
            except Exception as e:
                logger.warn(
                    f"An exception occurred ({e}) while copying the environment spec={env_spec}"
                )
                return None

        self._cached_spec = env_spec
        return env_spec

    def close(self):
        """Warns if calling close on a closed environment fails."""
        if not self.close_called:
            self.close_called = True
            return self.env.close()
        else:
            try:
                return self.env.close()
            except Exception as e:
                logger.warn(
                    "Calling `env.close()` on the closed environment should be allowed, but it raised the following exception."
                )
                raise e


class OrderEnforcing(Wrapper, RecordConstructorArgs):
    """Will produce an error if ``step`` or ``render`` is called before ``reset``."""

    def __init__(
        self,
        env: Env,
        disable_render_order_enforcing: bool = False,
    ):
        """A wrapper that will produce an error if `step` is called before an initial `reset`.

        Args:
            env: The environment to wrap
            disable_render_order_enforcing: If to disable render order enforcing
        """
        RecordConstructorArgs.__init__(
            self, disable_render_order_enforcing=disable_render_order_enforcing
        )
        Wrapper.__init__(self, env)

        self._has_reset: bool = False
        self._disable_render_order_enforcing: bool = disable_render_order_enforcing

    def step(
        self, action: dict[str, str]
    ) -> tuple[
        dict[str, Observation],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, Any],
    ]:
        """Steps through the environment."""
        if not self._has_reset:
            raise RuntimeError("Cannot call env.step() before calling env.reset()")
        return super().step(action)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        """Resets the environment with `kwargs`."""
        self._has_reset = True
        return super().reset(seed=seed, options=options)

    def render(self) -> Image.Image | list[Image.Image] | None:
        """Renders the environment with `kwargs`."""
        if not self._disable_render_order_enforcing and not self._has_reset:
            raise RuntimeError(
                "Cannot call `env.render()` before calling `env.reset()`, if this is an intended action, "
                "set `disable_render_order_enforcing=True` on the OrderEnforcer wrapper."
            )
        return super().render()

    @property
    def has_reset(self):
        """Returns if the environment has been reset before."""
        return self._has_reset

    @property
    def spec(self) -> EnvSpec | None:
        """Modifies the environment spec to add the `order_enforce=True`."""
        if self._cached_spec is not None:
            return self._cached_spec

        env_spec = self.env.spec
        if env_spec is not None:
            try:
                env_spec = deepcopy(env_spec)
                env_spec.order_enforce = True
            except Exception as e:
                logger.warn(
                    f"An exception occurred ({e}) while copying the environment spec={env_spec}"
                )
                return None

        self._cached_spec = env_spec
        return env_spec

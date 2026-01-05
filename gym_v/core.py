"""Core components for gym-v."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, SupportsFloat

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, field_validator

from gym_v.logger import get_logger
from gym_v.utils import RecordConstructorArgs, np_random

if TYPE_CHECKING:
    from gym_v.envs.registration import EnvSpec, WrapperSpec

logger = get_logger()


class Observation(BaseModel):
    """Standard observation structure combining visual and textual information.

    Attributes:
        image: A PIL Image object representing the visual observation
        text: A string containing textual description or context
        metadata: Optional dictionary for additional information
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: Image.Image = Field(..., description="Visual observation as a PIL Image")
    text: str | None = Field(default=None, description="Textual observation or context")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    @field_validator("image", mode="before")
    @classmethod
    def convert_image(cls, v: Any) -> Image.Image:
        """Convert numpy array to PIL Image if needed."""
        if isinstance(v, np.ndarray):
            return Image.fromarray(v)
        return v


class Env:
    """Base environment class for vision-language environments.

    The main API methods:
    - step(action): Execute action, return (observation, reward, terminated, truncated, info)
    - reset(seed, options): Reset environment, return (observation, info)
    - render(): Render the environment
    - close(): Clean up resources
    """

    metadata: dict[str, Any] = dict()
    spec: EnvSpec | None = None

    _np_random: np.random.Generator | None = None
    _np_random_seed: int | None = None

    def __init__(self, max_episode_steps: int | None = None):
        super().__init__()
        if max_episode_steps is not None and max_episode_steps > 0:
            self._max_episode_steps = max_episode_steps
        else:
            self._max_episode_steps = float("inf")
        self._current_episode_steps = 0

    @property
    def description(self) -> str:
        """Return description of the environment."""
        raise NotImplementedError

    def step(
        self, action: str
    ) -> tuple[Observation, SupportsFloat, bool, bool, dict[str, Any]]:
        """Run one timestep of the environment's dynamics.

        Args:
            action: Action string to execute

        Returns:
            observation: Next observation
            reward: Reward for this step
            terminated: Whether the agent reaches terminal state
            truncated: Whether truncation condition is satisfied
            info: Additional diagnostic information
        """
        self._current_episode_steps += 1
        obs, reward, terminated, truncated, info = self.inner_step(action)

        if self._current_episode_steps >= self._max_episode_steps:
            truncated = True

        return obs, reward, terminated, truncated, info

    def inner_step(
        self, action: str
    ) -> tuple[Observation, SupportsFloat, bool, bool, dict[str, Any]]:
        """Internal step implementation."""
        raise NotImplementedError

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        """Reset environment to initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional reset options

        Returns:
            observation: Initial observation
            info: Additional information
        """
        self._current_episode_steps = 0
        if seed is not None:
            self._np_random, self._np_random_seed = np_random(seed)

    def render(self) -> Image.Image:
        """Render the environment.

        Returns:
            Rendered frame as a PIL Image
        """
        raise NotImplementedError

    def close(self):
        """Clean up environment resources."""
        pass

    @property
    def unwrapped(self) -> Env:
        """Returns the base non-wrapped environment."""
        return self

    @property
    def np_random_seed(self) -> int:
        """Returns the environment's random seed."""
        if self._np_random_seed is None:
            self._np_random, self._np_random_seed = np_random()
        return self._np_random_seed

    @property
    def np_random(self) -> np.random.Generator:
        """Returns the environment's random number generator."""
        if self._np_random is None:
            self._np_random, self._np_random_seed = np_random()
        return self._np_random

    @np_random.setter
    def np_random(self, value: np.random.Generator):
        """Sets the environment's random number generator."""
        self._np_random = value
        self._np_random_seed = -1

    def __str__(self) -> str:
        """Returns a string of the environment with `spec` id's if `spec."""
        if self.spec is None:
            return f"<{type(self).__name__} instance>"
        else:
            return f"<{type(self).__name__}<{self.spec.id}>>"

    def __enter__(self) -> Env:
        """Support with-statement for the environment."""
        return self

    def __exit__(self, *args: Any) -> bool:
        """Close environment when exiting with-statement."""
        self.close()
        return False

    def has_wrapper_attr(self, name: str) -> bool:
        """Check if the named attribute is present in the environment."""
        return hasattr(self, name)

    def get_wrapper_attr(self, name: str) -> Any:
        """Get the named attribute from the environment."""
        return getattr(self, name)

    def set_wrapper_attr(self, name: str, value: Any, *, force: bool = True) -> bool:
        """Set the named attribute on the environment."""
        if force or hasattr(self, name):
            setattr(self, name, value)
            return True
        return False


class Wrapper(Env):
    """Wraps a `gym_v.Env` to allow a modular transformation of the `description`, `step` and `reset` methods."""

    def __init__(self, env: Env):
        """Wraps an environment to allow a modular transformation of the `description`, `step` and `reset` methods.

        Args:
            env: The environment to wrap
        """
        self.env = env
        assert isinstance(
            env, Env
        ), f"Expected env to be a `gym_v.Env` but got {type(env)}"

        self._metadata: dict[str, Any] | None = None

        self._cached_spec: EnvSpec | None = None

    @property
    def description(self) -> str:
        """Returns the `description` of the `env` that can be overwritten to change the returned data."""
        return self.env.description

    def step(
        self, action: str
    ) -> tuple[Observation, SupportsFloat, bool, bool, dict[str, Any]]:
        """Uses the `step` of the `env` that can be overwritten to change the returned data."""
        return self.env.step(action)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        """Uses the `reset` of the `env` that can be overwritten to change the returned data."""
        return self.env.reset(seed=seed, options=options)

    def render(self) -> Image.Image:
        """Uses the `render` of the `env` that can be overwritten to change the returned data."""
        return self.env.render()

    def close(self) -> None:
        """Closes the wrapper and `env`."""
        return self.env.close()

    @property
    def np_random_seed(self) -> int | None:
        """Returns the base environment's `np_random_seed`."""
        return self.env.np_random_seed

    @property
    def unwrapped(self) -> Env:
        """Returns the base environment of the wrapper.

        This will be the bare `gym_v.Env` environment, underneath all layers of wrappers.
        """
        return self.env.unwrapped

    @property
    def spec(self) -> EnvSpec | None:
        """Returns the `Env` `spec` attribute with the `WrapperSpec`."""
        if self._cached_spec is not None:
            return self._cached_spec

        env_spec = self.env.spec
        if env_spec is not None:
            # See if the wrapper inherits from `RecordConstructorArgs` then add the kwargs otherwise use `None` for the wrapper kwargs. This will raise an error in `make`
            if isinstance(self, RecordConstructorArgs):
                kwargs = self._saved_kwargs
                if "env" in kwargs:
                    kwargs = deepcopy(kwargs)
                    kwargs.pop("env")
            else:
                kwargs = None

            wrapper_spec = WrapperSpec(
                name=self.class_name(),
                entry_point=f"{self.__module__}:{type(self).__name__}",
                kwargs=kwargs,
            )

            try:
                env_spec = deepcopy(env_spec)
                env_spec.additional_wrappers += (wrapper_spec,)
            except Exception as e:
                logger.warn(
                    f"An exception occurred ({e}) while copying the environment spec={env_spec}"
                )
                return None

        self._cached_spec = env_spec
        return env_spec

    @classmethod
    def wrapper_spec(cls, **kwargs: Any) -> WrapperSpec:
        """Generates a `WrapperSpec` for the wrappers."""

        return WrapperSpec(
            name=cls.class_name(),
            entry_point=f"{cls.__module__}:{cls.__name__}",
            kwargs=kwargs,
        )

    def has_wrapper_attr(self, name: str) -> bool:
        """Checks if the given attribute is within the wrapper or its environment."""
        if hasattr(self, name):
            return True
        else:
            return self.env.has_wrapper_attr(name)

    def get_wrapper_attr(self, name: str) -> Any:
        """Gets an attribute from the wrapper and lower environments if `name` doesn't exist in this object."""
        if hasattr(self, name):
            return getattr(self, name)
        else:
            try:
                return self.env.get_wrapper_attr(name)
            except AttributeError as e:
                raise AttributeError(
                    f"wrapper {self.class_name()} has no attribute {name!r}"
                ) from e

    def set_wrapper_attr(self, name: str, value: Any, *, force: bool = True) -> bool:
        """Sets an attribute on this wrapper or lower environment if `name` is already defined.

        Args:
            name: The variable name
            value: The new variable value
            force: Whether to create the attribute on this wrapper if it does not exists on the
               lower environment instead of raising an exception

        Returns:
            If the variable has been set in this or a lower wrapper.
        """
        if hasattr(self, name):
            setattr(self, name, value)
            return True
        else:
            already_set = self.env.set_wrapper_attr(name, value, force=False)
            if already_set:
                return True
            elif force:
                setattr(self, name, value)
                return True
            else:
                return False

    def __str__(self):
        """Returns the wrapper name and the :attr:`env` representation string."""
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        """Returns the string representation of the wrapper."""
        return str(self)

    @classmethod
    def class_name(cls) -> str:
        """Returns the class name of the wrapper."""
        return cls.__name__

    @property
    def metadata(self) -> dict[str, Any]:
        """Returns the `Env` `metadata`."""
        if self._metadata is None:
            return self.env.metadata
        return self._metadata

    @metadata.setter
    def metadata(self, value: dict[str, Any]):
        self._metadata = value

    @property
    def np_random(self) -> np.random.Generator:
        """Returns the `Env` `np_random` attribute."""
        return self.env.np_random

    @np_random.setter
    def np_random(self, value: np.random.Generator):
        self.env.np_random = value

    @property
    def _np_random(self):
        """This code will never be run due to __getattr__ being called prior this.

        It seems that @property overwrites the variable (`_np_random`) meaning that __getattr__ gets called with the missing variable.
        """
        raise AttributeError(
            "Can't access `_np_random` of a wrapper, use `.unwrapped._np_random` or `.np_random`."
        )


class DescriptionWrapper(Wrapper):
    """Superclass of wrappers that can modify the `description`."""

    def __init__(self, env: Env):
        """Constructor for the description wrapper.

        Args:
            env: Environment to be wrapped.
        """
        Wrapper.__init__(self, env)

    @property
    def description(self) -> str:
        """Returns a modified description from `self.description`."""
        return self._description(self.env.description)

    def _description(self, description: str) -> str:
        """Returns a modified description"""
        raise NotImplementedError


class ObservationWrapper(Wrapper):
    """Modify observations from `Env.reset` and `Env.step` using `observation` function."""

    def __init__(self, env: Env):
        """Constructor for the observation wrapper.

        Args:
            env: Environment to be wrapped.
        """
        Wrapper.__init__(self, env)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        """Modifies the `env` after calling `reset`, returning a modified observation using `self.observation`."""
        obs, info = self.env.reset(seed=seed, options=options)
        return self.observation(obs), info

    def step(
        self, action: str
    ) -> tuple[Observation, SupportsFloat, bool, bool, dict[str, Any]]:
        """Modifies the `env` after calling `step` using `self.observation` on the returned observations."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(observation), reward, terminated, truncated, info

    def observation(self, observation: Observation) -> Observation:
        """Returns a modified observation.

        Args:
            observation: The `env` observation

        Returns:
            The modified observation
        """
        raise NotImplementedError


class RewardWrapper(Wrapper):
    """Superclass of wrappers that can modify the returning reward from a step."""

    def __init__(self, env: Env):
        """Constructor for the Reward wrapper.

        Args:
            env: Environment to be wrapped.
        """
        Wrapper.__init__(self, env)

    def step(
        self, action: str
    ) -> tuple[Observation, SupportsFloat, bool, bool, dict[str, Any]]:
        """Modifies the `env` `step` reward using `self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, self.reward(reward), terminated, truncated, info

    def reward(self, reward: SupportsFloat) -> SupportsFloat:
        """Returns a modified environment ``reward``.

        Args:
            reward: The `env` `step` reward

        Returns:
            The modified `reward`
        """
        raise NotImplementedError


class ActionWrapper(Wrapper):
    """Superclass of wrappers that can modify the action before `step`."""

    def __init__(self, env: Env):
        """Constructor for the action wrapper.

        Args:
            env: Environment to be wrapped.
        """
        Wrapper.__init__(self, env)

    def step(
        self, action: str
    ) -> tuple[Observation, SupportsFloat, bool, bool, dict[str, Any]]:
        """Runs the `env` `env.step` using the modified ``action`` from `self.action`."""
        return self.env.step(self.action(action))

    def action(self, action: str) -> str:
        """Returns a modified action before `step` is called.

        Args:
            action: The original `step` actions

        Returns:
            The modified actions
        """
        raise NotImplementedError

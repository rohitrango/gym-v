"""Core components for gym-v."""

from __future__ import annotations

from typing import Any, SupportsFloat

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, field_validator

from gym_v.utils import np_random


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
        """Return string representation of the environment."""
        return f"<{type(self).__name__} instance>"

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

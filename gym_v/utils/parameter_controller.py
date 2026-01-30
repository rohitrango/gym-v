"""Parameter controllers for difficulty-based environment configuration.

This module provides base classes for managing environment parameters that
change based on difficulty levels. Controllers implement progressive
difficulty scaling through different strategies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ParameterController(ABC):
    """Base class for difficulty-based parameter control.

    A ParameterController manages environment parameters that change based on
    difficulty level. Subclasses implement specific progression strategies
    (linear, staged, etc.).

    Attributes:
        current_difficulty: The current difficulty level (0 = initial state).
    """

    def __init__(self, initial_difficulty: int = 0) -> None:
        """Initialize the controller.

        Args:
            initial_difficulty: Starting difficulty level. If > 0, the
                controller will advance to this level during initialization.
        """
        self.current_difficulty = 0
        self._initialize_parameters()
        if initial_difficulty > 0:
            self.reset_to_difficulty(initial_difficulty)

    @abstractmethod
    def _initialize_parameters(self) -> None:
        """Initialize parameters to their starting values (difficulty 0).

        Subclasses must implement this to set all controlled parameters
        to their initial state.
        """

    @abstractmethod
    def update(self) -> None:
        """Advance parameters by one difficulty level.

        This method is called when difficulty increases. Subclasses implement
        the specific logic for how parameters change at each level.
        """

    @abstractmethod
    def get_parameters(self) -> dict[str, Any]:
        """Return current parameter values.

        Returns:
            Dictionary mapping parameter names to their current values.
        """

    def reset_to_difficulty(self, difficulty: int) -> None:
        """Reset and advance to a specific difficulty level.

        Args:
            difficulty: Target difficulty level (must be >= 0).

        Raises:
            ValueError: If difficulty is negative.
        """
        if difficulty < 0:
            raise ValueError(f"Difficulty must be non-negative, got {difficulty}")
        self._initialize_parameters()
        self.current_difficulty = 0
        for _ in range(difficulty):
            self.update()
            self.current_difficulty += 1


class LinearController(ParameterController):
    """Controller with linear parameter progression.

    Parameters increase linearly with difficulty according to configured
    min/max values and step sizes.

    Example:
        >>> class GridSizeController(LinearController):
        ...     def _get_param_configs(self):
        ...         return {
        ...             'grid_size': {'min': 3, 'max': 10, 'step': 1},
        ...             'obstacles': {'min': 0, 'max': 20, 'step': 2},
        ...         }
    """

    def _initialize_parameters(self) -> None:
        """Initialize parameters to their minimum values."""
        self._params = {}
        for name, config in self._get_param_configs().items():
            self._params[name] = config["min"]

    @abstractmethod
    def _get_param_configs(self) -> dict[str, dict[str, Any]]:
        """Return configuration for each parameter.

        Returns:
            Dictionary mapping parameter names to config dicts with keys:
                - 'min': Minimum value (at difficulty 0)
                - 'max': Maximum value (difficulty cap)
                - 'step': Increment per difficulty level
        """

    def update(self) -> None:
        """Increase each parameter by its step, up to max."""
        for name, config in self._get_param_configs().items():
            current = self._params[name]
            new_value = current + config["step"]
            self._params[name] = min(new_value, config["max"])

    def get_parameters(self) -> dict[str, Any]:
        """Return current parameter values."""
        return dict(self._params)


class StageController(ParameterController):
    """Controller with staged parameter progression.

    Parameters change at specific difficulty thresholds according to
    predefined stages.

    Example:
        >>> class GameModeController(StageController):
        ...     def _get_stages(self):
        ...         return [
        ...             (0, {'mode': 'easy', 'hints': 5}),
        ...             (3, {'mode': 'medium', 'hints': 3}),
        ...             (6, {'mode': 'hard', 'hints': 1}),
        ...         ]
    """

    def _initialize_parameters(self) -> None:
        """Initialize parameters from the first stage."""
        stages = self._get_stages()
        if stages:
            _, params = stages[0]
            self._params = dict(params)
        else:
            self._params = {}

    @abstractmethod
    def _get_stages(self) -> list[tuple[int, dict[str, Any]]]:
        """Return list of (threshold, parameters) tuples.

        Each tuple defines parameters that take effect when
        current_difficulty >= threshold. Stages should be ordered
        by threshold.

        Returns:
            List of (difficulty_threshold, parameter_dict) tuples.
        """

    def update(self) -> None:
        """Update parameters based on the new difficulty level."""
        next_difficulty = self.current_difficulty + 1
        stages = self._get_stages()
        # Find the highest stage that applies
        for threshold, params in reversed(stages):
            if next_difficulty >= threshold:
                self._params = dict(params)
                break

    def get_parameters(self) -> dict[str, Any]:
        """Return current parameter values."""
        return dict(self._params)


class CompositeController(ParameterController):
    """Controller that combines multiple sub-controllers.

    Useful when an environment has parameters with different progression
    strategies that need to be managed together.
    """

    def __init__(
        self, controllers: dict[str, ParameterController], initial_difficulty: int = 0
    ) -> None:
        """Initialize with named sub-controllers.

        Args:
            controllers: Dictionary mapping names to ParameterController instances.
            initial_difficulty: Starting difficulty level.
        """
        self._controllers = controllers
        super().__init__(initial_difficulty)

    def _initialize_parameters(self) -> None:
        """Initialize all sub-controllers."""
        for controller in self._controllers.values():
            controller._initialize_parameters()
            controller.current_difficulty = 0

    def update(self) -> None:
        """Advance all sub-controllers by one level."""
        for controller in self._controllers.values():
            controller.update()
            controller.current_difficulty += 1

    def get_parameters(self) -> dict[str, Any]:
        """Merge parameters from all sub-controllers."""
        params = {}
        for controller in self._controllers.values():
            params.update(controller.get_parameters())
        return params

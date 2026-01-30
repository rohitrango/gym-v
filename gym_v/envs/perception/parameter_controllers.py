"""Parameter controllers for Perception environments.

Perception environments test visual understanding capabilities.
"""

from __future__ import annotations

from typing import Any

from gym_v.utils.parameter_controller import ParameterController


class PerceptionDefaultController(ParameterController):
    """Default controller for Perception environments."""

    def _initialize_parameters(self) -> None:
        self._max_categories = 3
        self._grid_size = 3

    def update(self) -> None:
        d = self.current_difficulty + 1
        self._max_categories = min(10, 3 + d)
        self._grid_size = min(8, 3 + d // 2)

    def get_parameters(self) -> dict[str, Any]:
        return {
            "max_categories": self._max_categories,
            "grid_size": self._grid_size,
        }


def get_controller_for_env(
    env_class_name: str, difficulty: int = 0
) -> ParameterController:
    """Get the appropriate controller for a Perception environment."""
    return PerceptionDefaultController(difficulty)

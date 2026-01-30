"""Parameter controllers for Sphinx environments."""

from __future__ import annotations

from typing import Any

from gym_v.utils.parameter_controller import ParameterController


class SphinxDefaultController(ParameterController):
    """Default controller for Sphinx environments."""

    def _initialize_parameters(self) -> None:
        pass

    def update(self) -> None:
        pass

    def get_parameters(self) -> dict[str, Any]:
        return {}


def get_controller_for_env(
    env_class_name: str, difficulty: int = 0
) -> ParameterController:
    """Get the appropriate controller for a Sphinx environment."""
    return SphinxDefaultController(difficulty)

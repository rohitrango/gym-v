"""Parameter controllers for TextArena environments.

TextArena environments have various game-specific parameters.
Controllers map difficulty levels to these parameters.
"""

from __future__ import annotations

from typing import Any

from gym_v.utils.parameter_controller import ParameterController


class TextArenaDefaultController(ParameterController):
    """Default controller for TextArena environments."""

    def _initialize_parameters(self) -> None:
        pass

    def update(self) -> None:
        pass

    def get_parameters(self) -> dict[str, Any]:
        return {}


class TextArenaSudokuController(ParameterController):
    """Controller for TextArena Sudoku environment.

    Controls number of clues (fewer clues = harder puzzle).
    """

    def __init__(
        self,
        initial_difficulty: int = 0,
        max_clues: int = 40,
        min_clues: int = 17,
    ) -> None:
        self._max_clues = max_clues
        self._min_clues = min_clues
        super().__init__(initial_difficulty)

    def _initialize_parameters(self) -> None:
        self._clues = self._max_clues

    def update(self) -> None:
        # Decrease clues (harder puzzles)
        self._clues = max(self._min_clues, self._clues - 2)

    def get_parameters(self) -> dict[str, Any]:
        return {"clues": self._clues}


def get_controller_for_env(
    env_class_name: str, difficulty: int = 0
) -> ParameterController:
    """Get the appropriate controller for a TextArena environment."""
    controllers = {
        "TextArenaSudokuEnv": lambda d: TextArenaSudokuController(d),
    }

    factory = controllers.get(env_class_name)
    if factory:
        return factory(difficulty)
    return TextArenaDefaultController(difficulty)

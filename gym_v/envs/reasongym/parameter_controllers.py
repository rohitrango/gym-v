"""Parameter controllers for ReasoningGym environments.

ReasoningGym environments use dataset_kwargs for configuration.
Controllers map difficulty levels to appropriate dataset_kwargs values.
"""

from __future__ import annotations

from typing import Any

from gym_v.utils.parameter_controller import ParameterController


class ReasoningGymDefaultController(ParameterController):
    """Default controller for ReasoningGym environments.

    Most ReasoningGym environments control difficulty through dataset_kwargs.
    This controller tracks difficulty level and provides suggested kwargs.
    """

    def _initialize_parameters(self) -> None:
        self._dataset_kwargs: dict[str, Any] = {}

    def update(self) -> None:
        # Default: increase 'size' parameter if supported
        pass

    def get_parameters(self) -> dict[str, Any]:
        return {"dataset_kwargs": self._dataset_kwargs}


class ReasoningGymSudokuController(ParameterController):
    """Controller for Sudoku environment.

    Sudoku difficulty is controlled through dataset generation.
    """

    def _initialize_parameters(self) -> None:
        self._min_empty = 20
        self._max_empty = 30

    def update(self) -> None:
        d = self.current_difficulty + 1
        # Increase number of empty cells (harder puzzles)
        self._min_empty = min(45, 20 + d * 2)
        self._max_empty = min(55, 30 + d * 2)

    def get_parameters(self) -> dict[str, Any]:
        return {
            "dataset_kwargs": {
                "min_empty": self._min_empty,
                "max_empty": self._max_empty,
            }
        }


class ReasoningGymMazeController(ParameterController):
    """Controller for Maze environment."""

    def __init__(
        self,
        initial_difficulty: int = 0,
        min_size: int = 5,
        max_size: int = 15,
    ) -> None:
        self._min_size = min_size
        self._max_size = max_size
        super().__init__(initial_difficulty)

    def _initialize_parameters(self) -> None:
        self._grid_size = self._min_size

    def update(self) -> None:
        self._grid_size = min(self._max_size, self._grid_size + 1)

    def get_parameters(self) -> dict[str, Any]:
        return {
            "dataset_kwargs": {
                "min_grid_size": self._grid_size,
                "max_grid_size": self._grid_size,
            }
        }


class ReasoningGymNQueensController(ParameterController):
    """Controller for N-Queens environment."""

    def __init__(
        self,
        initial_difficulty: int = 0,
        min_n: int = 4,
        max_n: int = 12,
    ) -> None:
        self._min_n = min_n
        self._max_n = max_n
        super().__init__(initial_difficulty)

    def _initialize_parameters(self) -> None:
        self._n = self._min_n

    def update(self) -> None:
        self._n = min(self._max_n, self._n + 1)

    def get_parameters(self) -> dict[str, Any]:
        # Set min_remove and max_remove to valid values for the given n
        # max_remove must be between min_remove and n
        return {
            "dataset_kwargs": {"n": self._n, "min_remove": 1, "max_remove": self._n - 1}
        }


class ReasoningGymTowerOfHanoiController(ParameterController):
    """Controller for Tower of Hanoi environment."""

    def __init__(
        self,
        initial_difficulty: int = 0,
        min_disks: int = 3,
        max_disks: int = 8,
    ) -> None:
        self._min_disks_val = min_disks
        self._max_disks_val = max_disks
        super().__init__(initial_difficulty)

    def _initialize_parameters(self) -> None:
        self._num_disks = self._min_disks_val

    def update(self) -> None:
        self._num_disks = min(self._max_disks_val, self._num_disks + 1)

    def get_parameters(self) -> dict[str, Any]:
        return {
            "dataset_kwargs": {
                "min_disks": self._num_disks,
                "max_disks": self._num_disks,
            }
        }


class ReasoningGymGameOfLifeController(ParameterController):
    """Controller for Game of Life environment."""

    def __init__(
        self,
        initial_difficulty: int = 0,
        min_size: int = 5,
        max_size: int = 15,
    ) -> None:
        self._min_size = min_size
        self._max_size = max_size
        super().__init__(initial_difficulty)

    def _initialize_parameters(self) -> None:
        self._grid_size = self._min_size

    def update(self) -> None:
        self._grid_size = min(self._max_size, self._grid_size + 1)

    def get_parameters(self) -> dict[str, Any]:
        return {
            "dataset_kwargs": {
                "grid_size_x": self._grid_size,
                "grid_size_y": self._grid_size,
            }
        }


def get_controller_for_env(
    env_class_name: str, difficulty: int = 0
) -> ParameterController:
    """Get the appropriate controller for a ReasoningGym environment.

    Args:
        env_class_name: Name of the environment class.
        difficulty: Initial difficulty level.

    Returns:
        ParameterController instance.
    """
    controllers = {
        "ReasoningGymSudokuEnv": lambda d: ReasoningGymSudokuController(d),
        "ReasoningGymMazeEnv": lambda d: ReasoningGymMazeController(d),
        "ReasoningGymNQueensEnv": lambda d: ReasoningGymNQueensController(d),
        "ReasoningGymTowerOfHanoiEnv": lambda d: ReasoningGymTowerOfHanoiController(d),
        "ReasoningGymGameOfLifeEnv": lambda d: ReasoningGymGameOfLifeController(d),
    }

    factory = controllers.get(env_class_name)
    if factory:
        return factory(difficulty)
    return ReasoningGymDefaultController(difficulty)

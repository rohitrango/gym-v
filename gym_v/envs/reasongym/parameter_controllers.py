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


# =============================================================================
# Grid-based Linear Controllers (PR-1)
# =============================================================================


class ReasoningGymBinaryMatrixController(ParameterController):
    """Controller for Binary Matrix environment.

    Controls grid_size parameter (3→10).
    """

    def __init__(
        self,
        initial_difficulty: int = 0,
        min_size: int = 3,
        max_size: int = 10,
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


class ReasoningGymRotateMatrixController(ParameterController):
    """Controller for Rotate Matrix environment.

    Controls matrix_size parameter (2→8).
    """

    def __init__(
        self,
        initial_difficulty: int = 0,
        min_size: int = 2,
        max_size: int = 8,
    ) -> None:
        self._min_size = min_size
        self._max_size = max_size
        super().__init__(initial_difficulty)

    def _initialize_parameters(self) -> None:
        self._matrix_size = self._min_size

    def update(self) -> None:
        self._matrix_size = min(self._max_size, self._matrix_size + 1)

    def get_parameters(self) -> dict[str, Any]:
        return {
            "dataset_kwargs": {
                "min_size": self._matrix_size,
                "max_size": self._matrix_size,
            }
        }


class ReasoningGymSpiralMatrixController(ParameterController):
    """Controller for Spiral Matrix environment.

    Controls matrix_size parameter (3→10).
    """

    def __init__(
        self,
        initial_difficulty: int = 0,
        min_size: int = 3,
        max_size: int = 10,
    ) -> None:
        self._min_size = min_size
        self._max_size = max_size
        super().__init__(initial_difficulty)

    def _initialize_parameters(self) -> None:
        self._matrix_size = self._min_size

    def update(self) -> None:
        self._matrix_size = min(self._max_size, self._matrix_size + 1)

    def get_parameters(self) -> dict[str, Any]:
        return {
            "dataset_kwargs": {
                "min_rows": self._matrix_size,
                "max_rows": self._matrix_size,
                "min_cols": self._matrix_size,
                "max_cols": self._matrix_size,
            }
        }


class ReasoningGymLargestIslandController(ParameterController):
    """Controller for Largest Island environment.

    Controls grid_size parameter (4→12).
    """

    def __init__(
        self,
        initial_difficulty: int = 0,
        min_size: int = 4,
        max_size: int = 12,
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
                "min_rows": self._grid_size,
                "max_rows": self._grid_size,
                "min_cols": self._grid_size,
                "max_cols": self._grid_size,
            }
        }


class ReasoningGymRectangleCountController(ParameterController):
    """Controller for Rectangle Count environment.

    Controls grid_size parameter (4→10).
    """

    def __init__(
        self,
        initial_difficulty: int = 0,
        min_size: int = 4,
        max_size: int = 10,
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


class ReasoningGymRottenOrangesController(ParameterController):
    """Controller for Rotten Oranges environment.

    Controls grid_size parameter (3→10).
    """

    def __init__(
        self,
        initial_difficulty: int = 0,
        min_size: int = 3,
        max_size: int = 10,
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
                "min_n": self._grid_size,
                "max_n": self._grid_size,
            }
        }


class ReasoningGymShortestPathController(ParameterController):
    """Controller for Shortest Path environment.

    Controls grid_size parameter (4→15).
    """

    def __init__(
        self,
        initial_difficulty: int = 0,
        min_size: int = 4,
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


class ReasoningGymKakurasuController(ParameterController):
    """Controller for Kakurasu environment.

    Controls n_rows/n_cols parameter (3→8).
    """

    def __init__(
        self,
        initial_difficulty: int = 0,
        min_size: int = 3,
        max_size: int = 8,
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
                "min_n_rows": self._grid_size,
                "max_n_rows": self._grid_size,
                "min_n_cols": self._grid_size,
                "max_n_cols": self._grid_size,
            }
        }


class ReasoningGymSurvoController(ParameterController):
    """Controller for Survo environment.

    Controls grid_size parameter (3→8).
    """

    def __init__(
        self,
        initial_difficulty: int = 0,
        min_size: int = 3,
        max_size: int = 8,
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
                "min_board_size": self._grid_size,
                "max_board_size": self._grid_size,
            }
        }


# =============================================================================
# Stage-based Controllers (PR-2)
# =============================================================================


class ReasoningGymCircuitLogicController(ParameterController):
    """Controller for Circuit Logic environment.

    Uses staged difficulty with num_inputs and num_gates.
    Stage 0: 2 inputs, 3 gates
    Stage 1: 3 inputs, 5 gates
    Stage 2: 5 inputs, 8 gates
    """

    def __init__(self, initial_difficulty: int = 0) -> None:
        super().__init__(initial_difficulty)

    def _initialize_parameters(self) -> None:
        self._num_inputs = 2
        self._num_gates = 3

    def update(self) -> None:
        d = self.current_difficulty + 1
        if d < 3:
            self._num_inputs = 2
            self._num_gates = 3
        elif d < 6:
            self._num_inputs = 3
            self._num_gates = 5
        else:
            self._num_inputs = 5
            self._num_gates = 8

    def get_parameters(self) -> dict[str, Any]:
        return {
            "dataset_kwargs": {
                "min_inputs": self._num_inputs,
                "max_inputs": self._num_inputs,
                "min_gates": self._num_gates,
                "max_gates": self._num_gates,
            }
        }


class ReasoningGymKnightSwapController(ParameterController):
    """Controller for Knight Swap environment.

    Uses staged difficulty with board_size and num_knights.
    Stage 0: 3x3 board, 2 knights
    Stage 1: 4x4 board, 3 knights
    Stage 2: 5x5 board, 4 knights
    """

    def __init__(self, initial_difficulty: int = 0) -> None:
        super().__init__(initial_difficulty)

    def _initialize_parameters(self) -> None:
        self._board_size = 3
        self._num_knights = 2

    def update(self) -> None:
        d = self.current_difficulty + 1
        if d < 3:
            self._board_size = 3
            self._num_knights = 2
        elif d < 6:
            self._board_size = 4
            self._num_knights = 3
        else:
            self._board_size = 5
            self._num_knights = 4

    def get_parameters(self) -> dict[str, Any]:
        return {
            "dataset_kwargs": {
                "min_board_size": self._board_size,
                "max_board_size": self._board_size,
                "min_knights": self._num_knights,
                "max_knights": self._num_knights,
            }
        }


class ReasoningGymMiniSudokuController(ParameterController):
    """Controller for Mini Sudoku environment.

    Uses staged difficulty with puzzle size.
    Stage 0: 4x4 puzzles
    Stage 1: 6x6 puzzles
    Stage 2: 9x9 puzzles (full Sudoku)

    Note: MiniSudoku in reasoning-gym typically only supports 4x4.
    This controller reflects the conceptual stages.
    """

    def __init__(self, initial_difficulty: int = 0) -> None:
        super().__init__(initial_difficulty)

    def _initialize_parameters(self) -> None:
        self._min_empty = 4
        self._max_empty = 6

    def update(self) -> None:
        d = self.current_difficulty + 1
        # For 4x4 MiniSudoku, increase empty cells (harder puzzles)
        self._min_empty = min(10, 4 + d)
        self._max_empty = min(12, 6 + d)

    def get_parameters(self) -> dict[str, Any]:
        return {
            "dataset_kwargs": {
                "min_empty": self._min_empty,
                "max_empty": self._max_empty,
            }
        }


class ReasoningGymTsumegoController(ParameterController):
    """Controller for Tsumego (Go problems) environment.

    Uses staged difficulty with board_size and problem difficulty.
    Stage 0: 9x9 board, easy problems
    Stage 1: 13x13 board, medium problems
    Stage 2: 19x19 board, hard problems
    """

    def __init__(self, initial_difficulty: int = 0) -> None:
        super().__init__(initial_difficulty)

    def _initialize_parameters(self) -> None:
        self._board_size = 9
        self._problem_difficulty = "easy"

    def update(self) -> None:
        d = self.current_difficulty + 1
        if d < 3:
            self._board_size = 9
            self._problem_difficulty = "easy"
        elif d < 6:
            self._board_size = 13
            self._problem_difficulty = "medium"
        else:
            self._board_size = 19
            self._problem_difficulty = "hard"

    def get_parameters(self) -> dict[str, Any]:
        return {
            "dataset_kwargs": {
                "board_size": self._board_size,
                "difficulty": self._problem_difficulty,
            }
        }


class ReasoningGymArc1DController(ParameterController):
    """Controller for ARC 1D environment.

    ARC 1D uses dataset_kwargs but has no easily controllable difficulty
    parameters. This controller provides default behavior.
    """

    def __init__(self, initial_difficulty: int = 0) -> None:
        super().__init__(initial_difficulty)

    def _initialize_parameters(self) -> None:
        self._dataset_kwargs: dict[str, Any] = {}

    def update(self) -> None:
        # ARC 1D difficulty is inherent to the dataset
        pass

    def get_parameters(self) -> dict[str, Any]:
        return {"dataset_kwargs": self._dataset_kwargs}


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
        # Existing controllers
        "ReasoningGymSudokuEnv": lambda d: ReasoningGymSudokuController(d),
        "ReasoningGymMazeEnv": lambda d: ReasoningGymMazeController(d),
        "ReasoningGymNQueensEnv": lambda d: ReasoningGymNQueensController(d),
        "ReasoningGymTowerOfHanoiEnv": lambda d: ReasoningGymTowerOfHanoiController(d),
        "ReasoningGymGameOfLifeEnv": lambda d: ReasoningGymGameOfLifeController(d),
        # Grid-based Linear Controllers (PR-1)
        "ReasoningGymBinaryMatrixEnv": lambda d: ReasoningGymBinaryMatrixController(d),
        "ReasoningGymRotateMatrixEnv": lambda d: ReasoningGymRotateMatrixController(d),
        "ReasoningGymSpiralMatrixEnv": lambda d: ReasoningGymSpiralMatrixController(d),
        "ReasoningGymLargestIslandEnv": lambda d: ReasoningGymLargestIslandController(
            d
        ),
        "ReasoningGymRectangleCountEnv": lambda d: ReasoningGymRectangleCountController(
            d
        ),
        "ReasoningGymRottenOrangesEnv": lambda d: ReasoningGymRottenOrangesController(
            d
        ),
        "ReasoningGymShortestPathEnv": lambda d: ReasoningGymShortestPathController(d),
        "ReasoningGymKakurasuEnv": lambda d: ReasoningGymKakurasuController(d),
        "ReasoningGymSurvoEnv": lambda d: ReasoningGymSurvoController(d),
        # Stage-based Controllers (PR-2)
        "ReasoningGymCircuitLogicEnv": lambda d: ReasoningGymCircuitLogicController(d),
        "ReasoningGymKnightSwapEnv": lambda d: ReasoningGymKnightSwapController(d),
        "ReasoningGymMiniSudokuEnv": lambda d: ReasoningGymMiniSudokuController(d),
        "ReasoningGymTsumegoEnv": lambda d: ReasoningGymTsumegoController(d),
        "ReasoningGymArc1DEnv": lambda d: ReasoningGymArc1DController(d),
    }

    factory = controllers.get(env_class_name)
    if factory:
        return factory(difficulty)
    return ReasoningGymDefaultController(difficulty)

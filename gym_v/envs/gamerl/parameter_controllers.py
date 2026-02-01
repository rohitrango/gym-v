"""Parameter controllers for GameRL environments.

Each controller maps difficulty levels to environment-specific parameters.
GameRL environments are Q&A-based games where difficulty controls:
- Game board/grid size
- Game-specific difficulty settings
- Question type complexity
"""

from __future__ import annotations

from typing import Any

from gym_v.utils.parameter_controller import ParameterController, StageController


class GameRLDefaultController(ParameterController):
    """Default controller for GameRL environments without specific mappings.

    Simply tracks difficulty level without modifying parameters.
    """

    def _initialize_parameters(self) -> None:
        pass

    def update(self) -> None:
        pass

    def get_parameters(self) -> dict[str, Any]:
        return {}


class GameRLMinesweeperController(StageController):
    """Controller for Minesweeper Q&A difficulty.

    Maps unified difficulty to Minesweeper's game difficulty levels.
    """

    def _get_stages(self) -> list[tuple[int, dict[str, Any]]]:
        return [
            (0, {"game_difficulty": "Easy"}),
            (3, {"game_difficulty": "Medium"}),
            (6, {"game_difficulty": "Hard"}),
        ]


class GameRLRhythmGameController(StageController):
    """Controller for Rhythm Game Q&A difficulty."""

    def _get_stages(self) -> list[tuple[int, dict[str, Any]]]:
        return [
            (0, {"difficulty": "Easy"}),
            (3, {"difficulty": "Medium"}),
            (6, {"difficulty": "Hard"}),
        ]


class GameRLSudokuController(ParameterController):
    """Controller for Sudoku Q&A difficulty.

    Note: Sudoku difficulty is primarily controlled through puzzle generation.
    """

    def _initialize_parameters(self) -> None:
        self._game_difficulty = "Easy"

    def update(self) -> None:
        d = self.current_difficulty + 1
        if d <= 3:
            self._game_difficulty = "Easy"
        elif d <= 6:
            self._game_difficulty = "Medium"
        else:
            self._game_difficulty = "Hard"

    def get_parameters(self) -> dict[str, Any]:
        return {"game_difficulty": self._game_difficulty}


class GameRLMazeController(ParameterController):
    """Controller for Maze Q&A difficulty.

    Controls maze size and complexity.
    """

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
        self._size = self._min_size

    def update(self) -> None:
        self._size = min(self._max_size, self._size + 1)

    def get_parameters(self) -> dict[str, Any]:
        return {"maze_size": self._size}


class GameRLSnakeController(ParameterController):
    """Controller for Snake Q&A difficulty."""

    def __init__(
        self,
        initial_difficulty: int = 0,
        min_size: int = 6,
        max_size: int = 12,
    ) -> None:
        self._min_size = min_size
        self._max_size = max_size
        super().__init__(initial_difficulty)

    def _initialize_parameters(self) -> None:
        self._size = self._min_size

    def update(self) -> None:
        self._size = min(self._max_size, self._size + 1)

    def get_parameters(self) -> dict[str, Any]:
        return {"grid_size": self._size}


class GameRLTetrisController(ParameterController):
    """Controller for Tetris Q&A difficulty."""

    def __init__(
        self,
        initial_difficulty: int = 0,
        min_cols: int = 6,
        max_cols: int = 12,
    ) -> None:
        self._min_cols = min_cols
        self._max_cols = max_cols
        super().__init__(initial_difficulty)

    def _initialize_parameters(self) -> None:
        self._cols = self._min_cols

    def update(self) -> None:
        self._cols = min(self._max_cols, self._cols + 1)

    def get_parameters(self) -> dict[str, Any]:
        return {"cols": self._cols}


class GameRLRubiksCubeController(StageController):
    """Controller for Rubik's Cube Q&A difficulty.

    Controls cube size (2x2, 3x3, etc.) and scramble depth.
    """

    def _get_stages(self) -> list[tuple[int, dict[str, Any]]]:
        return [
            (0, {"cube_size": 2, "scramble_depth": 3}),
            (3, {"cube_size": 2, "scramble_depth": 5}),
            (5, {"cube_size": 3, "scramble_depth": 3}),
            (7, {"cube_size": 3, "scramble_depth": 5}),
            (10, {"cube_size": 3, "scramble_depth": 8}),
        ]


class GameRLSokobanController(ParameterController):
    """Controller for Sokoban Q&A difficulty."""

    def __init__(
        self,
        initial_difficulty: int = 0,
        min_level: int = 1,
        max_level: int = 20,
    ) -> None:
        self._min_level = min_level
        self._max_level = max_level
        super().__init__(initial_difficulty)

    def _initialize_parameters(self) -> None:
        self._level = self._min_level

    def update(self) -> None:
        self._level = min(self._max_level, self._level + 1)

    def get_parameters(self) -> dict[str, Any]:
        return {"level": self._level}


def get_controller_for_env(
    env_class_name: str, difficulty: int = 0
) -> ParameterController:
    """Get the appropriate controller for a GameRL environment.

    Args:
        env_class_name: Name of the environment class.
        difficulty: Initial difficulty level.

    Returns:
        ParameterController instance.
    """
    controllers = {
        "GameRLMinesweeperQAEnv": lambda d: GameRLMinesweeperController(d),
        "GameRLSudokuQAEnv": lambda d: GameRLSudokuController(d),
        "GameRLMazeQAEnv": lambda d: GameRLMazeController(d),
        "GameRLSnakeQAEnv": lambda d: GameRLSnakeController(d),
        "GameRLTetrisQAEnv": lambda d: GameRLTetrisController(d),
        "GameRLRubiksCubeQAEnv": lambda d: GameRLRubiksCubeController(d),
        "GameRLSokobanQAEnv": lambda d: GameRLSokobanController(d),
        "GameRLRhythmGameQAEnv": lambda d: GameRLRhythmGameController(d),
    }

    factory = controllers.get(env_class_name)
    if factory:
        return factory(difficulty)
    return GameRLDefaultController(difficulty)

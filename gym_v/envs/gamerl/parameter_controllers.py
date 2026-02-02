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


class GameRLGridSizeController(ParameterController):
    """Controller for grid_size-based GameRL QA envs."""

    def __init__(
        self, initial_difficulty: int = 0, min_size: int = 4, max_size: int = 12
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


class GameRLBoardSizeController(ParameterController):
    """Controller for board_size-based GameRL QA envs."""

    def __init__(
        self, initial_difficulty: int = 0, min_size: int = 4, max_size: int = 12
    ) -> None:
        self._min_size = min_size
        self._max_size = max_size
        super().__init__(initial_difficulty)

    def _initialize_parameters(self) -> None:
        self._size = self._min_size

    def update(self) -> None:
        self._size = min(self._max_size, self._size + 1)

    def get_parameters(self) -> dict[str, Any]:
        return {"board_size": self._size}


class GameRLSizeController(ParameterController):
    """Controller for size-based GameRL QA envs."""

    def __init__(
        self, initial_difficulty: int = 0, min_size: int = 4, max_size: int = 12
    ) -> None:
        self._min_size = min_size
        self._max_size = max_size
        super().__init__(initial_difficulty)

    def _initialize_parameters(self) -> None:
        self._size = self._min_size

    def update(self) -> None:
        self._size = min(self._max_size, self._size + 1)

    def get_parameters(self) -> dict[str, Any]:
        return {"size": self._size}


class GameRLFreecellController(ParameterController):
    """Controller for Freecell Q&A (cascade_number)."""

    def _initialize_parameters(self) -> None:
        self._cascade_number = 4

    def update(self) -> None:
        self._cascade_number = min(10, self._cascade_number + 1)

    def get_parameters(self) -> dict[str, Any]:
        return {"cascade_number": self._cascade_number}


class GameRLHueController(ParameterController):
    """Controller for Hue Q&A (board_size + num_lines)."""

    def __init__(
        self,
        initial_difficulty: int = 0,
        min_size: int = 4,
        max_size: int = 10,
        min_lines: int = 4,
        max_lines: int = 12,
    ) -> None:
        self._min_size = min_size
        self._max_size = max_size
        self._min_lines = min_lines
        self._max_lines = max_lines
        super().__init__(initial_difficulty)

    def _initialize_parameters(self) -> None:
        self._size = self._min_size
        self._lines = self._min_lines

    def update(self) -> None:
        self._size = min(self._max_size, self._size + 1)
        self._lines = min(self._max_lines, self._lines + 1)

    def get_parameters(self) -> dict[str, Any]:
        return {"board_size": self._size, "num_lines": self._lines}


class GameRLSpiderSolitaireController(ParameterController):
    """Controller for SpiderSolitaire Q&A."""

    def _initialize_parameters(self) -> None:
        self._num_waste = 1

    def update(self) -> None:
        self._num_waste = min(5, self._num_waste + 1)

    def get_parameters(self) -> dict[str, Any]:
        return {"num_waste": self._num_waste}


class GameRLTangramController(ParameterController):
    """Controller for Tangram Q&A."""

    def _initialize_parameters(self) -> None:
        self._grid_size = 6
        self._num_seeds = 2
        self._num_remove = 1

    def update(self) -> None:
        self._grid_size = min(12, self._grid_size + 1)
        self._num_seeds = min(6, self._num_seeds + 1)
        self._num_remove = min(4, self._num_remove + 1)

    def get_parameters(self) -> dict[str, Any]:
        return {
            "grid_size": self._grid_size,
            "num_seeds": self._num_seeds,
            "num_pieces_to_remove": self._num_remove,
        }


class GameRLTentsController(ParameterController):
    """Controller for Tents Q&A (grid_size + num_trees)."""

    def __init__(
        self,
        initial_difficulty: int = 0,
        min_grid: int = 6,
        max_grid: int = 12,
        min_trees: int = 5,
        max_trees: int = 20,
    ) -> None:
        self._min_grid = min_grid
        self._max_grid = max_grid
        self._min_trees = min_trees
        self._max_trees = max_trees
        super().__init__(initial_difficulty)

    def _initialize_parameters(self) -> None:
        self._grid = self._min_grid
        self._trees = self._min_trees

    def update(self) -> None:
        self._grid = min(self._max_grid, self._grid + 1)
        self._trees = min(self._max_trees, self._trees + 2)

    def get_parameters(self) -> dict[str, Any]:
        return {"grid_size": (self._grid, self._grid), "num_trees": self._trees}


class GameRLTuringMachineController(ParameterController):
    """Controller for TuringMachine2d Q&A."""

    def __init__(
        self,
        initial_difficulty: int = 0,
        min_grid: int = 5,
        max_grid: int = 12,
        min_states: int = 2,
        max_states: int = 6,
        min_symbols: int = 2,
        max_symbols: int = 4,
        min_steps: int = 10,
        max_steps: int = 40,
    ) -> None:
        self._min_grid = min_grid
        self._max_grid = max_grid
        self._min_states = min_states
        self._max_states = max_states
        self._min_symbols = min_symbols
        self._max_symbols = max_symbols
        self._min_steps = min_steps
        self._max_steps = max_steps
        super().__init__(initial_difficulty)

    def _initialize_parameters(self) -> None:
        self._grid = self._min_grid
        self._states = self._min_states
        self._symbols = self._min_symbols
        self._steps = self._min_steps

    def update(self) -> None:
        self._grid = min(self._max_grid, self._grid + 1)
        self._states = min(self._max_states, self._states + 1)
        self._symbols = min(self._max_symbols, self._symbols + 1)
        self._steps = min(self._max_steps, self._steps + 5)

    def get_parameters(self) -> dict[str, Any]:
        return {
            "grid_size": (self._grid, self._grid),
            "num_states": self._states,
            "num_symbols": self._symbols,
            "max_steps": self._steps,
        }


class GameRLSpaceInvadersController(ParameterController):
    """Controller for SpaceInvaders Q&A."""

    def _initialize_parameters(self) -> None:
        self._enemy_rows = 3
        self._enemy_cols = 4
        self._enemy_area_rows = 6

    def update(self) -> None:
        self._enemy_rows = min(5, self._enemy_rows + 1)
        self._enemy_cols = min(8, self._enemy_cols + 1)
        self._enemy_area_rows = min(8, self._enemy_area_rows + 1)

    def get_parameters(self) -> dict[str, Any]:
        return {
            "enemy_rows": self._enemy_rows,
            "enemy_cols": self._enemy_cols,
            "enemy_area_rows": self._enemy_area_rows,
        }


class GameRLStarBattleController(ParameterController):
    """Controller for StarBattle Q&A."""

    def _initialize_parameters(self) -> None:
        self._grid_size = 5
        self._stars_per_region = 1

    def update(self) -> None:
        if self._grid_size == 5:
            self._grid_size = 6
        elif self._grid_size == 6:
            self._grid_size = 8
        if self._grid_size >= 8:
            self._stars_per_region = 2

    def get_parameters(self) -> dict[str, Any]:
        return {
            "grid_size": self._grid_size,
            "stars_per_region": self._stars_per_region,
        }


class GameRLPacmanController(ParameterController):
    """Controller for Pacman Q&A."""

    def _initialize_parameters(self) -> None:
        self._grid_size = 12
        self._wall_ratio = 0.08

    def update(self) -> None:
        self._grid_size = min(20, self._grid_size + 1)
        self._wall_ratio = min(0.2, self._wall_ratio + 0.01)

    def get_parameters(self) -> dict[str, Any]:
        return {"grid_size": self._grid_size, "wall_ratio": self._wall_ratio}


class GameRLZumaController(ParameterController):
    """Controller for Zuma Q&A."""

    def _initialize_parameters(self) -> None:
        self._num_balls = 15

    def update(self) -> None:
        self._num_balls = min(40, self._num_balls + 3)

    def get_parameters(self) -> dict[str, Any]:
        return {"num_balls": self._num_balls}


class GameRLPlotLevelController(ParameterController):
    """Controller for plot_level-based GameRL QA envs."""

    def _initialize_parameters(self) -> None:
        self._plot_level = "Easy"

    def update(self) -> None:
        if self._plot_level == "Easy":
            self._plot_level = "Medium"
        elif self._plot_level == "Medium":
            self._plot_level = "Hard"

    def get_parameters(self) -> dict[str, Any]:
        return {"plot_level": self._plot_level}


class GameRLChessRangerController(ParameterController):
    """Controller for ChessRanger Q&A."""

    def _initialize_parameters(self) -> None:
        self._num_pieces = 4

    def update(self) -> None:
        self._num_pieces = min(10, self._num_pieces + 1)

    def get_parameters(self) -> dict[str, Any]:
        return {"num_pieces": self._num_pieces}


class GameRL3dMazeController(ParameterController):
    """Controller for 3D Maze QA (grid_size tuple)."""

    def _initialize_parameters(self) -> None:
        self._size = 5
        self._height = 5

    def update(self) -> None:
        self._size = min(10, self._size + 1)
        self._height = min(8, self._height + 1)

    def get_parameters(self) -> dict[str, Any]:
        return {"grid_size": (self._size, self._size, self._height)}


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
        "GameRL3dMazeQAEnv": lambda d: GameRL3dMazeController(d),
        "GameRLLifegameQAEnv": lambda d: GameRLGridSizeController(d, 5, 12),
        "GameRLLangtonAntQAEnv": lambda d: GameRLGridSizeController(d, 5, 12),
        "GameRLFreecellQAEnv": lambda d: GameRLFreecellController(d),
        "GameRLHueQAEnv": lambda d: GameRLHueController(d),
        "GameRLJewel2QAEnv": lambda d: GameRLSizeController(d, 4, 10),
        "GameRLPacmanQAEnv": lambda d: GameRLPacmanController(d),
        "GameRLSpiderSolitaireQAEnv": lambda d: GameRLSpiderSolitaireController(d),
        "GameRLTangramQAEnv": lambda d: GameRLTangramController(d),
        "GameRLTentsQAEnv": lambda d: GameRLTentsController(d),
        "GameRL2dTuringMachineQAEnv": lambda d: GameRLTuringMachineController(d),
        "GameRLSpaceInvadersQAEnv": lambda d: GameRLSpaceInvadersController(d),
        "GameRLStarBattleQAEnv": lambda d: GameRLStarBattleController(d),
        "GameRLWordSearchQAEnv": lambda d: GameRLGridSizeController(d, 8, 16),
        "GameRLZumaQAEnv": lambda d: GameRLZumaController(d),
        "GameRL3DReconstructionQAEnv": lambda d: GameRLPlotLevelController(d),
        "GameRLPyramidChessQAEnv": lambda d: GameRLPlotLevelController(d),
        "GameRLUltraTicTacToeQAEnv": lambda d: GameRLPlotLevelController(d),
        "GameRLChessRangerQAEnv": lambda d: GameRLChessRangerController(d),
    }

    factory = controllers.get(env_class_name)
    if factory:
        return factory(difficulty)
    return GameRLDefaultController(difficulty)

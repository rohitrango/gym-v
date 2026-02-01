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


# =============================================================================
# Linear Controllers (PR-3)
# =============================================================================


class TextArenaLightsOutController(ParameterController):
    """Controller for LightsOut environment.

    Controls size parameter (3→7).
    """

    def __init__(
        self,
        initial_difficulty: int = 0,
        min_size: int = 3,
        max_size: int = 7,
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


class TextArenaGame2048Controller(ParameterController):
    """Controller for Game2048 environment.

    Controls target_tile parameter (128→2048→8192).
    Higher target tiles are harder.
    """

    def __init__(
        self,
        initial_difficulty: int = 0,
    ) -> None:
        # Target tiles: 128, 256, 512, 1024, 2048, 4096, 8192
        self._target_tiles = [128, 256, 512, 1024, 2048, 4096, 8192]
        super().__init__(initial_difficulty)

    def _initialize_parameters(self) -> None:
        self._target_tile_idx = 0

    def update(self) -> None:
        self._target_tile_idx = min(
            len(self._target_tiles) - 1, self._target_tile_idx + 1
        )

    def get_parameters(self) -> dict[str, Any]:
        return {"target_tile": self._target_tiles[self._target_tile_idx]}


class TextArenaSokobanController(ParameterController):
    """Controller for Sokoban environment.

    Controls room size and num_boxes.
    Starts with (6,6) room and 2 boxes, increases to larger rooms with more boxes.
    """

    def __init__(
        self,
        initial_difficulty: int = 0,
        min_room_size: int = 6,
        max_room_size: int = 10,
        min_boxes: int = 2,
        max_boxes: int = 5,
    ) -> None:
        self._min_room_size = min_room_size
        self._max_room_size = max_room_size
        self._min_boxes = min_boxes
        self._max_boxes = max_boxes
        super().__init__(initial_difficulty)

    def _initialize_parameters(self) -> None:
        self._room_size = self._min_room_size
        self._num_boxes = self._min_boxes

    def update(self) -> None:
        d = self.current_difficulty + 1
        # Increase room size every 2 difficulty levels
        self._room_size = min(self._max_room_size, self._min_room_size + d // 2)
        # Increase boxes every 3 difficulty levels
        self._num_boxes = min(self._max_boxes, self._min_boxes + d // 3)

    def get_parameters(self) -> dict[str, Any]:
        return {
            "dim_room": (self._room_size, self._room_size),
            "num_boxes": self._num_boxes,
        }


class TextArenaTowerOfHanoiController(ParameterController):
    """Controller for Tower of Hanoi environment.

    Controls num_disks parameter (3→8).
    """

    def __init__(
        self,
        initial_difficulty: int = 0,
        min_disks: int = 3,
        max_disks: int = 8,
    ) -> None:
        self._min_disks = min_disks
        self._max_disks = max_disks
        super().__init__(initial_difficulty)

    def _initialize_parameters(self) -> None:
        self._num_disks = self._min_disks

    def update(self) -> None:
        self._num_disks = min(self._max_disks, self._num_disks + 1)

    def get_parameters(self) -> dict[str, Any]:
        return {"num_disks": self._num_disks}


class TextArenaWordleController(ParameterController):
    """Controller for Wordle environment.

    Controls word_length parameter (4→7).
    Longer words are harder.
    """

    def __init__(
        self,
        initial_difficulty: int = 0,
        min_length: int = 4,
        max_length: int = 7,
    ) -> None:
        self._min_length = min_length
        self._max_length = max_length
        super().__init__(initial_difficulty)

    def _initialize_parameters(self) -> None:
        self._word_length = self._min_length

    def update(self) -> None:
        self._word_length = min(self._max_length, self._word_length + 1)

    def get_parameters(self) -> dict[str, Any]:
        return {"word_length": self._word_length}


class TextArenaFifteenPuzzleController(ParameterController):
    """Controller for 15-Puzzle environment.

    The 15-puzzle has fixed 4x4 size, so difficulty is controlled by
    the initial shuffle complexity. We don't have direct control over
    shuffle_moves in TextArena, so this provides default parameters.
    """

    def __init__(self, initial_difficulty: int = 0) -> None:
        super().__init__(initial_difficulty)

    def _initialize_parameters(self) -> None:
        pass

    def update(self) -> None:
        pass

    def get_parameters(self) -> dict[str, Any]:
        return {}


# =============================================================================
# Composite Controllers (PR-4)
# =============================================================================


class TextArenaFrozenLakeController(ParameterController):
    """Controller for FrozenLake environment.

    Controls size (4→8) and num_holes (2→6) parameters.
    """

    def __init__(
        self,
        initial_difficulty: int = 0,
        min_size: int = 4,
        max_size: int = 8,
        min_holes: int = 2,
        max_holes: int = 6,
    ) -> None:
        self._min_size = min_size
        self._max_size = max_size
        self._min_holes = min_holes
        self._max_holes = max_holes
        super().__init__(initial_difficulty)

    def _initialize_parameters(self) -> None:
        self._size = self._min_size
        self._num_holes = self._min_holes

    def update(self) -> None:
        d = self.current_difficulty + 1
        # Increase size every 2 levels
        self._size = min(self._max_size, self._min_size + d // 2)
        # Increase holes every difficulty level
        self._num_holes = min(self._max_holes, self._min_holes + d)

    def get_parameters(self) -> dict[str, Any]:
        return {"size": self._size, "num_holes": self._num_holes}


class TextArenaMinesweeperController(ParameterController):
    """Controller for Minesweeper environment.

    Controls rows, cols (5→12) and num_mines parameters.
    """

    def __init__(
        self,
        initial_difficulty: int = 0,
        min_size: int = 5,
        max_size: int = 12,
        min_mines: int = 5,
        max_mines: int = 20,
    ) -> None:
        self._min_size = min_size
        self._max_size = max_size
        self._min_mines = min_mines
        self._max_mines = max_mines
        super().__init__(initial_difficulty)

    def _initialize_parameters(self) -> None:
        self._size = self._min_size
        self._num_mines = self._min_mines

    def update(self) -> None:
        d = self.current_difficulty + 1
        # Increase size every 2 levels
        self._size = min(self._max_size, self._min_size + d // 2)
        # Increase mines proportionally
        self._num_mines = min(self._max_mines, self._min_mines + d * 2)

    def get_parameters(self) -> dict[str, Any]:
        return {
            "rows": self._size,
            "cols": self._size,
            "num_mines": self._num_mines,
        }


class TextArenaWordSearchController(ParameterController):
    """Controller for WordSearch environment.

    Controls hardcore mode (easier vs harder word sets).
    """

    def __init__(self, initial_difficulty: int = 0) -> None:
        super().__init__(initial_difficulty)

    def _initialize_parameters(self) -> None:
        self._hardcore = False

    def update(self) -> None:
        d = self.current_difficulty + 1
        # Switch to hardcore after difficulty 3
        self._hardcore = d >= 3

    def get_parameters(self) -> dict[str, Any]:
        return {"hardcore": self._hardcore}


# =============================================================================
# Stage-based Controllers (PR-4)
# =============================================================================


class TextArenaCrosswordsController(ParameterController):
    """Controller for Crosswords environment.

    Controls hardcore mode and num_words.
    """

    def __init__(self, initial_difficulty: int = 0) -> None:
        super().__init__(initial_difficulty)

    def _initialize_parameters(self) -> None:
        self._hardcore = False
        self._num_words = 3

    def update(self) -> None:
        d = self.current_difficulty + 1
        if d < 3:
            self._hardcore = False
            self._num_words = 3
        elif d < 6:
            self._hardcore = False
            self._num_words = 5
        else:
            self._hardcore = True
            self._num_words = 7

    def get_parameters(self) -> dict[str, Any]:
        return {"hardcore": self._hardcore, "num_words": self._num_words}


class TextArenaPegJumpController(ParameterController):
    """Controller for PegJump environment.

    Controls initial_empty position.
    Different starting positions have different difficulty levels.
    """

    def __init__(self, initial_difficulty: int = 0) -> None:
        # Different initial empty positions (1-15)
        # Position 1 (apex) is easiest, middle positions are harder
        self._positions_by_difficulty = [1, 4, 5, 13, 8, 7, 11, 12, 14, 15]
        super().__init__(initial_difficulty)

    def _initialize_parameters(self) -> None:
        self._initial_empty = 1

    def update(self) -> None:
        d = self.current_difficulty + 1
        idx = min(d, len(self._positions_by_difficulty) - 1)
        self._initial_empty = self._positions_by_difficulty[idx]

    def get_parameters(self) -> dict[str, Any]:
        return {"initial_empty": self._initial_empty}


class TextArenaRushHourController(ParameterController):
    """Controller for RushHour environment.

    Controls difficulty level (easy/medium/hard).
    """

    def __init__(self, initial_difficulty: int = 0) -> None:
        super().__init__(initial_difficulty)

    def _initialize_parameters(self) -> None:
        self._difficulty = "easy"

    def update(self) -> None:
        d = self.current_difficulty + 1
        if d < 3:
            self._difficulty = "easy"
        elif d < 6:
            self._difficulty = "medium"
        else:
            self._difficulty = "hard"

    def get_parameters(self) -> dict[str, Any]:
        return {"difficulty": self._difficulty}


def get_controller_for_env(
    env_class_name: str, difficulty: int = 0
) -> ParameterController:
    """Get the appropriate controller for a TextArena environment."""
    controllers = {
        # Existing controller
        "TextArenaSudokuEnv": lambda d: TextArenaSudokuController(d),
        # Linear Controllers (PR-3)
        "TextArenaLightsOutEnv": lambda d: TextArenaLightsOutController(d),
        "TextArenaGame2048Env": lambda d: TextArenaGame2048Controller(d),
        "TextArenaSokobanEnv": lambda d: TextArenaSokobanController(d),
        "TextArenaTowerOfHanoiEnv": lambda d: TextArenaTowerOfHanoiController(d),
        "TextArenaWordleEnv": lambda d: TextArenaWordleController(d),
        "TextArenaFifteenPuzzleEnv": lambda d: TextArenaFifteenPuzzleController(d),
        # Composite Controllers (PR-4)
        "TextArenaFrozenLakeEnv": lambda d: TextArenaFrozenLakeController(d),
        "TextArenaMinesweeperEnv": lambda d: TextArenaMinesweeperController(d),
        "TextArenaWordSearchEnv": lambda d: TextArenaWordSearchController(d),
        # Stage-based Controllers (PR-4)
        "TextArenaCrosswordsEnv": lambda d: TextArenaCrosswordsController(d),
        "TextArenaPegJumpEnv": lambda d: TextArenaPegJumpController(d),
        "TextArenaRushHourEnv": lambda d: TextArenaRushHourController(d),
    }

    factory = controllers.get(env_class_name)
    if factory:
        return factory(difficulty)
    return TextArenaDefaultController(difficulty)

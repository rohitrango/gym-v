from __future__ import annotations

import random
from typing import Any


class Constraint:
    def __init__(self) -> None:
        self.name = ""

    def check(self, game_state: dict[str, Any]) -> bool:
        pass


class PuzzleFactory:
    def __init__(self) -> None:
        self.constraints = []
        self.game_name = "unknown"
        self.size = 0

    def sample_hints(
        self, board: list[list[Any]], num_sample_hints: int
    ) -> list[list[Any]]:
        # Create a new board filled with zeros
        new_board = [[0 for _ in range(len(board[0]))] for _ in range(len(board))]
        # Sample num_sample_hints cells to keep from the original board
        sampled_cells = random.sample(
            range(len(board) * len(board[0])), num_sample_hints
        )
        for cell in sampled_cells:
            row = cell // len(board[0])
            col = cell % len(board[0])
            new_board[row][col] = board[row][
                col
            ]  # Copy only the sampled cells from original board
        return new_board

    def check(self, game_state: dict[str, Any]) -> bool:
        for constraint in self.constraints:
            if not constraint.check(game_state):
                return False
        return True

    def get_possible_values(
        self, game_state: dict[str, Any], row: int, col: int
    ) -> list[Any]:
        pass


class ConstraintRowNoRepeat(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_row_no_repeat"

    def check(self, game_state: dict[str, Any]) -> bool:
        board = game_state["board"]
        for row in board:
            values = [x for x in row if x != 0]
            if len(set(values)) != len(values):
                return False
        return True


class ConstraintColNoRepeat(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_col_no_repeat"

    def check(self, game_state: dict[str, Any]) -> bool:
        board = game_state["board"]
        size = len(board)
        for col in range(size):
            values = [board[row][col] for row in range(size) if board[row][col] != 0]
            if len(set(values)) != len(values):
                return False
        return True

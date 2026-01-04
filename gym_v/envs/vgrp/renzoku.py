"""Renzoku single-turn environment backed by VGRP-Bench."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from gym_v import Env, Observation, get_logger

from .utils import generate_puzzle
from .vgrp_base import (
    Constraint,
    ConstraintColNoRepeat,
    ConstraintRowNoRepeat,
    PuzzleFactory,
)

logger = get_logger()


class ConstraintAdjacency(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_adjacency"

    def check(self, game_state: dict[str, Any]) -> bool:
        board = game_state["board"]
        size = len(board)
        hints = game_state.get("hints")
        if not hints:
            return True

        if len(hints.get("row", [])) < size:
            hints["row"] = [["0" for _ in range(size - 1)] for _ in range(size)]
        if len(hints.get("col", [])) < size - 1:
            hints["col"] = [["0" for _ in range(size)] for _ in range(size - 1)]

        # Check row adjacency
        for row in range(size):
            for col in range(size - 1):
                if hints["row"][row][col] == "1":
                    val1 = board[row][col]
                    val2 = board[row][col + 1]
                    if val1 == 0 or val2 == 0:
                        continue
                    if abs(val1 - val2) != 1:
                        return False

        # Check col adjacency
        for row in range(size - 1):
            for col in range(size):
                if hints["col"][row][col] == "1":
                    val1 = board[row][col]
                    val2 = board[row + 1][col]
                    if val1 == 0 or val2 == 0:
                        continue
                    if abs(val1 - val2) != 1:
                        return False
        return True


class RenzokuPuzzleFactory(PuzzleFactory):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.game_name = "renzoku"
        self.size = size
        self.constraints = [
            ConstraintRowNoRepeat(),
            ConstraintColNoRepeat(),
            ConstraintAdjacency(),
        ]
        self.all_possible_values = [i for i in range(1, size + 1)]

    def get_possible_values(
        self, game_state: dict[str, Any], row: int, col: int
    ) -> list[int]:
        possible_values = []
        board = game_state["board"]
        original_value = board[row][col]
        for value in self.all_possible_values:
            board[row][col] = value
            if self.check(game_state):
                possible_values.append(value)
        board[row][col] = original_value
        return possible_values


class VGRPRenzokuEnv(Env):
    """Renzoku puzzle using VGRP-Bench's factory.

    Fill the grid with numbers 1-N (like Sudoku).
    Constraint:
    - Dots between cells indicate the two numbers are consecutive (difference is 1).
    - If there is NO dot, the numbers must NOT be consecutive.

    Args:
        size: Grid size (default 9)
        cell_px: Size of each cell in pixels for rendering
        padding: Padding around the grid
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    def __init__(
        self,
        size: int = 9,
        cell_px: int = 50,
        padding: int = 30,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._size = size
        self._cell_px = cell_px
        self._padding = padding

        self._seed: int | None = None
        # State
        self._solution_board: list[list[int]] | None = None
        self._hints: dict[str, list[list[str]]] | None = (
            None  # {'row': ..., 'col': ...}
        )
        self._puzzle_board: list[list[int]] | None = None
        self._factory = RenzokuPuzzleFactory(size)

    @property
    def description(self) -> str:
        return dedent(f"""
            Solve this {self._size}x{self._size} Renzoku puzzle.

            Rules:
            1. Fill the grid with numbers 1 to {self._size}.
            2. Each row and column must contain each number exactly once. (Standard Latin Square / Sudoku rules apply).
            3. A dot between two cells means the numbers are consecutive (difference is 1).
            4. No dot means the numbers are NOT consecutive.

            Output format: A {self._size}x{self._size} grid with numbers separated by spaces.
            separated by spaces within rows, and newlines separating rows.
            Example for 9x9:
            5 3 4 6 7 8 9 1 2
            6 7 2 1 9 5 3 4 8
            1 9 8 3 4 2 5 6 7
            8 5 9 7 6 1 4 2 3
            4 2 6 8 5 3 7 9 1
            7 1 3 9 2 4 8 5 6
            9 6 1 5 3 7 2 8 4
            2 8 7 4 1 9 6 3 5
            3 4 5 2 8 6 1 7 9
        """).strip()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)
        self._seed = seed
        if seed is not None:
            np.random.seed(seed)

        # 1. Generate Sudoku solution using generate_puzzle
        # Renzoku without hints is just a Latin Square (due to current factory constraints)
        # However, to be a valid SUDOKU (blocks), Renzoku factory doesn't enforce blocks?
        # Let's check RenzokuPuzzleFactory constraints.
        # It has RowNoRepeat, ColNoRepeat, Adjacency. No Subgrid.
        # So it generates a Latin Square that respects adjacency.
        # If we pass empty hints, it generates a Latin Square.

        hints = {
            "row": [["0" for _ in range(self._size - 1)] for _ in range(self._size)],
            "col": [["0" for _ in range(self._size)] for _ in range(self._size - 1)],
        }

        result = generate_puzzle(
            self._factory, self._size, num_hints=0, hints=hints, max_attempts=1000
        )
        if result is None:
            raise RuntimeError("Failed to generate Renzoku solution")

        _, self._solution_board = result

        # 2. Generate hints (dots)
        self._hints = self._generate_hints(self._solution_board)

        # 3. Puzzle board (empty)
        self._puzzle_board = [[0 for _ in range(self._size)] for _ in range(self._size)]

        logger.info("Reset VGRP Renzoku.")

        obs = Observation(
            image=self.render(),
            text=self._board_to_text_with_hints(),
            metadata={"size": self._size},
        )
        info = {
            "oracle_answer": self._board_to_text(self._solution_board),
        }
        return obs, info

    def inner_step(
        self, action: str
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        try:
            answer_board = self._text_to_board(action)
            reward = 1.0 if self._check_solution(answer_board) else 0.0
        except Exception as e:
            logger.warning(f"Failed to parse answer: {e}")
            reward = 0.0

        obs = Observation(
            image=self.render(),
            text=self._board_to_text_with_hints(),
            metadata={"size": self._size},
        )
        info = {
            "oracle_answer": self._board_to_text(self._solution_board),
        }
        return obs, reward, True, False, info

    def _generate_sudoku_solution(self) -> list[list[int]]:
        """Generate a valid Sudoku/Latin Square."""
        # For Renzoku, usually it's just Sudoku
        board = [[0 for _ in range(self._size)] for _ in range(self._size)]
        # Just use random Latin Square generator or Sudoku if size is square
        # For simplicity and speed, let's use a simple valid generator
        # Or reuse backtracking with factory (but need to disable adjacency constraints for generation first)

        # Temporary disable adjacency constraint logic by passing empty hints
        # dummy_hints = {
        #     "row": [["0" for _ in range(self._size - 1)] for _ in range(self._size)],
        #     "col": [["0" for _ in range(self._size)] for _ in range(self._size - 1)],
        # }
        # Actually Renzoku factory checks hints if they exist. If we pass hints that match any board (e.g. no dots and ignore "no dot" rule?),
        # But wait, Renzoku rules say "No dot means NOT consecutive".
        # So we can't easily use the factory to generate the BOARD, because the board defines the hints.
        # We should generate a board first, THEN compute hints.

        # Generate a random Latin Square (or Sudoku)
        # Simple shuffling of rows of a canonical solution often works for Latin Square, but Sudoku needs block constraints.
        # Let's generate a valid Sudoku.

        sub_size = int(self._size**0.5)

        # Canonical pattern
        def pattern(r, c):
            return (sub_size * (r % sub_size) + r // sub_size + c) % self._size

        from random import sample

        def shuffle(s):
            return sample(s, len(s))

        rBase = range(sub_size)
        rows = [g * sub_size + r for g in shuffle(rBase) for r in shuffle(rBase)]
        cols = [g * sub_size + c for g in shuffle(rBase) for c in shuffle(rBase)]
        nums = shuffle(range(1, self._size + 1))

        board = [[nums[pattern(r, c)] for c in cols] for r in rows]
        return board

    def _generate_hints(self, board: list[list[int]]) -> dict:
        row_hints = [["0" for _ in range(self._size - 1)] for _ in range(self._size)]
        col_hints = [["0" for _ in range(self._size)] for _ in range(self._size - 1)]

        for r in range(self._size):
            for c in range(self._size - 1):
                if abs(board[r][c] - board[r][c + 1]) == 1:
                    row_hints[r][c] = "1"

        for r in range(self._size - 1):
            for c in range(self._size):
                if abs(board[r][c] - board[r + 1][c]) == 1:
                    col_hints[r][c] = "1"

        return {"row": row_hints, "col": col_hints}

    def _check_solution(self, answer_board: list[list[int]]) -> bool:
        """Check if the answer matches the solution or satisfies constraints."""
        if len(answer_board) != self._size:
            return False
        for i in range(self._size):
            if len(answer_board[i]) != self._size:
                return False

        # 1. Exact match
        matches_solution = True
        for i in range(self._size):
            for j in range(self._size):
                if answer_board[i][j] != self._solution_board[i][j]:
                    matches_solution = False
                    break
            if not matches_solution:
                break

        if matches_solution:
            return True

        # 2. VGRP check
        # Must be complete (no 0s)
        for row in answer_board:
            if 0 in row:
                return False

        game_state = {"board": answer_board, "hints": self._hints}
        return self._factory.check(game_state)

    def _board_to_text(self, board: list[list[int]]) -> str:
        return "\n".join(" ".join(str(x) for x in row) for row in board)

    def _board_to_text_with_hints(self) -> str:
        # Renzoku text representation is tricky. We can just say "look at the image".
        return "Please refer to the image for connectivity dots (Renzoku constraints)."

    def _text_to_board(self, text: str) -> list[list[int]]:
        lines = text.strip().split("\n")
        board = []
        for line in lines:
            line = line.strip()
            if not line or "refer" in line:
                continue
            row = []
            for val in line.split():
                try:
                    row.append(int(val))
                except Exception:
                    row.append(0)
            if row:
                board.append(row)
        return board

    def render(self) -> Image.Image:
        size_px = self._cell_px * self._size + 2 * self._padding
        img = Image.new("RGB", (size_px, size_px), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Grid
        for i in range(self._size + 1):
            w = 2 if i % int(self._size**0.5) == 0 else 1
            x = self._padding + i * self._cell_px
            draw.line(
                [(x, self._padding), (x, size_px - self._padding)],
                fill=(0, 0, 0),
                width=w,
            )
            y = self._padding + i * self._cell_px
            draw.line(
                [(self._padding, y), (size_px - self._padding, y)],
                fill=(0, 0, 0),
                width=w,
            )

        # Dots
        dot_radius = 4

        # Row dots (between columns)
        for r in range(self._size):
            for c in range(self._size - 1):
                if self._hints["row"][r][c] == "1":
                    x = self._padding + (c + 1) * self._cell_px
                    y = self._padding + r * self._cell_px + self._cell_px // 2
                    draw.ellipse(
                        [
                            x - dot_radius,
                            y - dot_radius,
                            x + dot_radius,
                            y + dot_radius,
                        ],
                        fill=(0, 0, 0),
                    )

        # Col dots (between rows)
        for r in range(self._size - 1):
            for c in range(self._size):
                if self._hints["col"][r][c] == "1":
                    x = self._padding + c * self._cell_px + self._cell_px // 2
                    y = self._padding + (r + 1) * self._cell_px
                    draw.ellipse(
                        [
                            x - dot_radius,
                            y - dot_radius,
                            x + dot_radius,
                            y + dot_radius,
                        ],
                        fill=(0, 0, 0),
                    )

        return img

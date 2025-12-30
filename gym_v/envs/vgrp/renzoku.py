"""Renzoku single-turn environment backed by VGRP-Bench."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from gym_v import Env, Observation, get_logger

from .vgrp_factories import RenzokuPuzzleFactory

logger = get_logger()


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
        self._factory = RenzokuPuzzleFactory(size)

        # State
        self._solution_board: list[list[int]] | None = None
        self._hints: dict[str, list[list[str]]] | None = (
            None  # {'row': ..., 'col': ...}
        )
        self._puzzle_board: list[list[int]] | None = None

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
        """).strip()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)
        self._seed = seed
        if seed is not None:
            np.random.seed(seed)

        # 1. Generate Sudoku solution
        self._solution_board = self._generate_sudoku_solution()

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
        if len(answer_board) != self._size:
            return False
        for i in range(self._size):
            if len(answer_board[i]) != self._size:
                return False
            for j in range(self._size):
                if answer_board[i][j] != self._solution_board[i][j]:
                    return False
        return True

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

"""Futoshiki single-turn environment backed by VGRP-Bench."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation, get_logger

from .vgrp_factories import FutoshikiPuzzleFactory

logger = get_logger()


class VGRPFutoshikiEnv(Env):
    """Futoshiki puzzle using VGRP-Bench's factory.

    Fill the grid with numbers 1-N.
    Inequality signs (<, >) between cells must be respected.
    Each row and column must contain each number exactly once.

    Args:
        size: Grid size (default 5)
        cell_px: Size of each cell in pixels for rendering
        padding: Padding around the grid
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    def __init__(
        self,
        size: int = 5,
        cell_px: int = 60,
        padding: int = 30,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._size = size
        self._cell_px = cell_px
        self._padding = padding

        self._seed: int | None = None
        self._factory = FutoshikiPuzzleFactory(size)

        self._solution_board: list[list[int]] | None = None
        self._inequalities: dict[str, list[list[str]]] | None = None
        self._puzzle_board: list[list[int]] | None = None

    @property
    def description(self) -> str:
        return dedent(f"""
            Solve this {self._size}x{self._size} Futoshiki puzzle.

            Rules:
            1. Fill the grid with numbers 1 to {self._size}.
            2. Each row and column must contain each number exactly once.
            3. Respect the inequality signs between cells (if any).
               - '<' or '>' between horizontal neighbors.
               - '^' or 'v' between vertical neighbors.

            Output format: A {self._size}x{self._size} grid with numbers separated by spaces.
        """).strip()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)
        self._seed = seed
        if seed is not None:
            np.random.seed(seed)

        # 1. Generate Latin Square
        self._solution_board = self._generate_latin_square()

        # 2. Generate Inequalities
        self._inequalities = self._generate_inequalities(self._solution_board)

        # 3. Create Puzzle (hide numbers, keep some hints?)
        # Standard Futoshiki has some initial numbers. Let's keep ~20%.
        self._puzzle_board = [[0 for _ in range(self._size)] for _ in range(self._size)]
        num_hints = max(1, int(self._size * self._size * 0.2))
        cells = [(r, c) for r in range(self._size) for c in range(self._size)]
        np.random.shuffle(cells)
        for i in range(num_hints):
            r, c = cells[i]
            self._puzzle_board[r][c] = self._solution_board[r][c]

        logger.info("Reset VGRP Futoshiki.")

        obs = Observation(
            image=self.render(),
            text=self._board_to_text_puzzle(),
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
            text=self._board_to_text_puzzle(),
            metadata={"size": self._size},
        )
        info = {
            "oracle_answer": self._board_to_text(self._solution_board),
        }
        return obs, reward, True, False, info

    def _generate_latin_square(self) -> list[list[int]]:
        # Simple shuffling for Latin Square
        # nums = list(range(1, self._size + 1))
        # board = [[0] * self._size for _ in range(self._size)]

        # Shifted rows method
        # first_row = list(range(1, self._size + 1))
        # np.random.shuffle(first_row)
        # for r in range(self._size):
        #     shift = r
        #     # Shuffle columns mapping? No, that preserves LS property?
        #     # Simple cyclic shift of a random permutation is a Latin Square.
        #     # But we want more randomness.
        #     # Let's use backtracking on empty board (it's fast for small sizes like 5-9)
        #     pass

        # Use backtracking for guaranteed valid LS
        empty_board = [[0] * self._size for _ in range(self._size)]
        # We can use FutoshikiFactory with empty constraints
        # But we don't have direct access to "LatinSquareFactory".
        # Reusing Sudoku logic or just writing a simple backtrack here.

        def solve(bo):
            for r in range(self._size):
                for c in range(self._size):
                    if bo[r][c] == 0:
                        row_vals = {bo[r][k] for k in range(self._size)}
                        col_vals = {bo[k][c] for k in range(self._size)}
                        valid = [
                            x
                            for x in range(1, self._size + 1)
                            if x not in row_vals and x not in col_vals
                        ]
                        np.random.shuffle(valid)
                        for val in valid:
                            bo[r][c] = val
                            if solve(bo):
                                return True
                            bo[r][c] = 0
                        return False
            return True

        solve(empty_board)
        return empty_board

    def _generate_inequalities(self, board: list[list[int]]) -> dict:
        row_ineq = [["" for _ in range(self._size - 1)] for _ in range(self._size)]
        col_ineq = [["" for _ in range(self._size)] for _ in range(self._size - 1)]

        # Add random inequalities (density ~30-50%)
        density = 0.4

        for r in range(self._size):
            for c in range(self._size - 1):
                if np.random.random() < density:
                    if board[r][c] < board[r][c + 1]:
                        row_ineq[r][c] = "<"
                    else:
                        row_ineq[r][c] = ">"

        for r in range(self._size - 1):
            for c in range(self._size):
                if np.random.random() < density:
                    if board[r][c] < board[r + 1][c]:
                        col_ineq[r][c] = (
                            "^"  # Top is smaller (standard notation?) or just use visual logic
                        )
                        # Factory uses '^' and 'v'
                    else:
                        col_ineq[r][c] = "v"

        return {"row": row_ineq, "col": col_ineq}

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

    def _board_to_text_puzzle(self) -> str:
        lines = []
        for row in self._puzzle_board:
            lines.append(" ".join(str(x) if x > 0 else "." for x in row))
        return "\n".join(lines)

    def _text_to_board(self, text: str) -> list[list[int]]:
        lines = text.strip().split("\n")
        board = []
        for line in lines:
            line = line.strip()
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

        font = ImageFont.load_default()
        try:
            font_path = self.assets_dir / "DejaVuSans.ttf"
            if font_path.exists():
                font = ImageFont.truetype(str(font_path), size=24)
        except Exception:
            pass

        # Cells
        for r in range(self._size):
            for c in range(self._size):
                x = self._padding + c * self._cell_px
                y = self._padding + r * self._cell_px

                draw.rectangle(
                    [x, y, x + self._cell_px, y + self._cell_px],
                    outline=(0, 0, 0),
                    width=2,
                )

                val = self._puzzle_board[r][c]
                if val > 0:
                    bbox = draw.textbbox((0, 0), str(val), font=font)
                    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    draw.text(
                        (x + (self._cell_px - w) / 2, y + (self._cell_px - h) / 2),
                        str(val),
                        fill=(0, 0, 0),
                        font=font,
                    )

        # Inequalities
        sign_font = font  # Reuse font

        for r in range(self._size):
            for c in range(self._size - 1):
                sign = self._inequalities["row"][r][c]
                if sign:
                    x = self._padding + (c + 1) * self._cell_px
                    y = self._padding + r * self._cell_px + self._cell_px // 2
                    # Draw sign centered on the grid line
                    bbox = draw.textbbox((0, 0), sign, font=sign_font)
                    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    # Offset slightly to center on line
                    draw.rectangle(
                        [x - 10, y - 10, x + 10, y + 10], fill=(255, 255, 255)
                    )  # Clear background
                    draw.text(
                        (x - w / 2, y - h / 2), sign, fill=(0, 0, 0), font=sign_font
                    )

        for r in range(self._size - 1):
            for c in range(self._size):
                sign = self._inequalities["col"][r][c]
                if sign:
                    x = self._padding + c * self._cell_px + self._cell_px // 2
                    y = self._padding + (r + 1) * self._cell_px
                    bbox = draw.textbbox((0, 0), sign, font=sign_font)
                    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    draw.rectangle(
                        [x - 10, y - 10, x + 10, y + 10], fill=(255, 255, 255)
                    )
                    draw.text(
                        (x - w / 2, y - h / 2), sign, fill=(0, 0, 0), font=sign_font
                    )

        return img

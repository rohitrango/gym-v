"""Binairo single-turn environment backed by VGRP-Bench."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from gym_v import Env, Observation, get_logger
from gym_v.envs.vgrp.puzzle_generator import generate_puzzle

from .vgrp_factories import BinairoPuzzleFactory

logger = get_logger()


class VGRPBinairoEnv(Env):
    """Binairo puzzle using VGRP-Bench's Binairo puzzle generator.

    Fill the grid with white (w) and black (b) circles following the rules:
    - Each row and column must have equal white and black circles
    - No more than two consecutive circles of the same color in any row/column

    Args:
        size: Grid size (must be even, default 6)
        num_hints: Number of pre-filled cells
        cell_px: Size of each cell in pixels for rendering
        padding: Padding around the grid in pixels for rendering
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    def __init__(
        self,
        size: int = 6,
        num_hints: int = 12,
        cell_px: int = 60,
        padding: int = 24,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._size = size
        self._num_hints = num_hints
        self._cell_px = cell_px
        self._padding = padding

        self._seed: int | None = None
        self._factory = BinairoPuzzleFactory(size)
        self._puzzle_board: list[list[str]] | None = None
        self._solution_board: list[list[str]] | None = None

    @property
    def description(self) -> str:
        """Return description for Binairo puzzle."""
        return dedent(f"""
            Fill this {self._size}x{self._size} Binairo grid with white (w) and black (b) circles.

            In the image:
            - Empty cells need to be filled

            Rules:
            1. Each row and column must have exactly {self._size // 2} white and {self._size // 2} black circles
            2. No more than two consecutive circles of the same color in any row or column

            Output format: A {self._size}x{self._size} grid with 'w' for White circles or 'b' for Black circles separated by spaces within rows,
            and newlines separating rows. Example for 6x6:
            w b w b w b
            b w b w b w
            w b b w w b
            b w w b b w
            w b w w b b
            b w b b w w
        """).strip()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)
        self._seed = seed
        if seed is not None:
            np.random.seed(seed)

        # Generate a new puzzle
        result = generate_puzzle(self._factory, self._size, self._num_hints)
        if result is None:
            raise RuntimeError(
                f"Failed to generate Binairo puzzle with size {self._size} and {self._num_hints} hints"
            )

        self._puzzle_board, self._solution_board = result
        logger.info(f"Reset VGRP Binairo with {self._num_hints} hints.")

        # Convert board to text format
        puzzle_text = self._board_to_text(self._puzzle_board)

        obs = Observation(
            image=self.render(),
            text=puzzle_text,
            metadata={"size": self._size, "num_hints": self._num_hints},
        )
        info = {
            "oracle_answer": self._board_to_text(self._solution_board),
        }
        return obs, info

    def inner_step(
        self, action: str
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        # Parse answer and check correctness
        try:
            answer_board = self._text_to_board(action)
            reward = 1.0 if self._check_solution(answer_board) else 0.0
        except Exception as e:
            logger.warning(f"Failed to parse answer: {e}")
            reward = 0.0

        obs = Observation(
            image=self.render(),
            text=None,
            metadata={"size": self._size},
        )
        info = {
            "oracle_answer": self._board_to_text(self._solution_board),
        }

        return obs, reward, True, False, info

    def _check_solution(self, answer_board: list[list[str]]) -> bool:
        """Check if the answer matches the solution."""
        if len(answer_board) != self._size:
            return False
        for i in range(self._size):
            if len(answer_board[i]) != self._size:
                return False
            for j in range(self._size):
                if answer_board[i][j] != self._solution_board[i][j]:
                    return False
        return True

    def _board_to_text(self, board: list[list[str]]) -> str:
        """Convert board to text format."""
        lines = []
        for row in board:
            lines.append(" ".join(str(v) if v != 0 else "?" for v in row))
        return "\n".join(lines)

    def _text_to_board(self, text: str) -> list[list[str]]:
        """Convert text to board."""
        lines = text.strip().split("\n")
        board = []
        for line in lines:
            row = []
            for val in line.strip().split():
                if val in ["w", "b"]:
                    row.append(val)
                elif val in ["○", "o", "O", "white", "0"]:
                    row.append("w")
                elif val in ["●", "x", "X", "black", "1"]:
                    row.append("b")
                else:
                    row.append(0)
            board.append(row)
        return board

    def render(self) -> Image.Image:
        return self._render_binairo_grid(
            self._puzzle_board, cell_px=self._cell_px, padding=self._padding
        )

    def _render_binairo_grid(
        self,
        board: list[list[str]],
        cell_px: int = 60,
        padding: int = 24,
        bg: tuple[int, int, int] = (250, 250, 250),
        grid: tuple[int, int, int] = (30, 30, 30),
    ) -> Image.Image:
        n = self._size
        size = padding * 2 + cell_px * n
        img = Image.new("RGB", (size, size), bg)
        draw = ImageDraw.Draw(img)

        # Draw grid
        for i in range(n + 1):
            x = padding + i * cell_px
            y = padding + i * cell_px
            draw.line([(x, padding), (x, padding + n * cell_px)], fill=grid, width=2)
            draw.line([(padding, y), (padding + n * cell_px, y)], fill=grid, width=2)

        # Draw circles with 3D effect
        for i in range(n):
            for j in range(n):
                cell = board[i][j]
                x = padding + j * cell_px
                y = padding + i * cell_px

                if cell == "w":
                    # White circle with shadow and highlight
                    # Shadow
                    draw.ellipse(
                        [x + 13, y + 13, x + cell_px - 7, y + cell_px - 7],
                        fill=(200, 200, 200),
                    )
                    # Main circle
                    draw.ellipse(
                        [x + 10, y + 10, x + cell_px - 10, y + cell_px - 10],
                        fill=(255, 255, 255),
                        outline=(80, 80, 80),
                        width=3,
                    )
                    # Highlight for 3D effect
                    highlight_size = cell_px // 5
                    draw.ellipse(
                        [
                            x + cell_px // 4,
                            y + cell_px // 4,
                            x + cell_px // 4 + highlight_size,
                            y + cell_px // 4 + highlight_size,
                        ],
                        fill=(240, 240, 240),
                    )
                elif cell == "b":
                    # Black circle with gradient effect
                    # Shadow
                    draw.ellipse(
                        [x + 13, y + 13, x + cell_px - 7, y + cell_px - 7],
                        fill=(100, 100, 100),
                    )
                    # Main circle
                    draw.ellipse(
                        [x + 10, y + 10, x + cell_px - 10, y + cell_px - 10],
                        fill=(30, 30, 30),
                        outline=(20, 20, 20),
                        width=2,
                    )
                    # Highlight
                    highlight_size = cell_px // 6
                    draw.ellipse(
                        [
                            x + cell_px // 4,
                            y + cell_px // 4,
                            x + cell_px // 4 + highlight_size,
                            y + cell_px // 4 + highlight_size,
                        ],
                        fill=(100, 100, 100),
                    )
                # else: empty cell (0)

        return img

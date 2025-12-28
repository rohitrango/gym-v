"""Trees and Tents single-turn environment backed by VGRP-Bench."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation, get_logger

from .vgrp_factories import TreesAndTentsPuzzleFactory

logger = get_logger()


class VGRPTreesAndTentsEnv(Env):
    """Trees and Tents puzzle using VGRP-Bench's TreesAndTents puzzle generator.

    Place tents next to trees according to clues.

    Args:
        size: Grid size (default 5)
        num_hints: Not used (clues derived from solution)
        cell_px: Size of each cell in pixels for rendering
        padding: Padding around the grid in pixels for rendering
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    def __init__(
        self,
        size: int = 5,
        num_hints: int = 0,
        cell_px: int = 60,
        padding: int = 50,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._size = size
        self._cell_px = cell_px
        self._padding = padding

        self._seed: int | None = None
        self._factory = TreesAndTentsPuzzleFactory(size)
        self._puzzle_board: list[list[str]] | None = None
        self._solution_board: list[list[str]] | None = None
        self._tree_positions: list[tuple[int, int]] | None = None
        self._row_clues: list[int] | None = None
        self._col_clues: list[int] | None = None

    @property
    def description(self) -> str:
        """Return description for Trees and Tents puzzle."""
        return dedent(f"""
            Solve this {self._size}x{self._size} Trees and Tents puzzle.

            In the image:
            - Numbers on left show required tents per row
            - Numbers on top show required tents per column

            Rules:
            1. Place exactly one tent horizontally or vertically adjacent to each tree
            2. Tents cannot touch each other, even diagonally
            3. Row and column tent counts must match the given clues
            4. Each tree must have exactly one tent next to it (not diagonal)

            Output format: A {self._size}x{self._size} grid where:
            - 'tr' = tree (given, don't output)
            - 'tt' = tent (your answer)
            - 'e' = empty
            separated by spaces within rows, and newlines separating rows.
        """).strip()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed, options=options)
        self._seed = seed
        if seed is not None:
            np.random.seed(seed)

        # Generate tree positions
        self._tree_positions = self._generate_trees()

        # Generate solution directly
        self._solution_board = self._generate_tents_solution()

        # Calculate row/col clues from solution
        self._row_clues = [
            sum(1 for cell in row if cell == "tt") for row in self._solution_board
        ]
        self._col_clues = [
            sum(1 for i in range(self._size) if self._solution_board[i][j] == "tt")
            for j in range(self._size)
        ]

        # Puzzle board shows trees
        self._puzzle_board = [
            ["e" for _ in range(self._size)] for _ in range(self._size)
        ]
        for r, c in self._tree_positions:
            self._puzzle_board[r][c] = "tr"

        logger.info("Reset VGRP Trees and Tents.")

        obs = Observation(
            image=self.render(),
            text=self._board_to_text_with_clues(),
            metadata={"size": self._size},
        )
        info = {
            "oracle_answer": self._board_to_text(self._solution_board),
        }
        return obs, info

    def _generate_trees(self) -> list[tuple[int, int]]:
        """Generate random tree positions."""
        trees = []
        num_trees = np.random.randint(max(2, self._size - 2), self._size + 1)

        available_cells = [(i, j) for i in range(self._size) for j in range(self._size)]
        np.random.shuffle(available_cells)

        for cell in available_cells[:num_trees]:
            trees.append(cell)

        return trees

    def _generate_tents_solution(self) -> list[list[str]]:
        """Generate solution by placing tents next to trees."""
        solution = [["e" for _ in range(self._size)] for _ in range(self._size)]

        # Place trees
        for r, c in self._tree_positions:
            solution[r][c] = "tr"

        # Place tents next to trees (simple strategy)
        used_tents = set()
        for r, c in self._tree_positions:
            # Try to place tent adjacent
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < self._size
                    and 0 <= nc < self._size
                    and (nr, nc) not in used_tents
                    and solution[nr][nc] == "e"
                ):
                    solution[nr][nc] = "tt"
                    used_tents.add((nr, nc))
                    break

        return solution

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
            text=self._board_to_text_with_clues(),
            metadata={"size": self._size},
        )
        terminated = True
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

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

    def _board_to_text_with_clues(self) -> str:
        """Convert board to text with clues."""
        lines = []
        lines.append(
            f"Row clues (tents per row): {' '.join(map(str, self._row_clues))}"
        )
        lines.append(
            f"Col clues (tents per col): {' '.join(map(str, self._col_clues))}"
        )
        lines.append("")
        lines.append("Trees are marked in the grid below:")
        for row in self._puzzle_board:
            lines.append(" ".join(row))
        return "\n".join(lines)

    def _board_to_text(self, board: list[list[str]]) -> str:
        """Convert board to text."""
        lines = []
        for row in board:
            lines.append(" ".join(row))
        return "\n".join(lines)

    def _text_to_board(self, text: str) -> list[list[str]]:
        """Parse text to board."""
        lines = text.strip().split("\n")
        board = []
        for line in lines:
            line = line.strip()
            if not line or ":" in line or "Trees" in line:
                continue
            row = []
            for val in line.split():
                val = val.strip().lower()
                if val in ["tt", "tr", "e"]:
                    row.append(val)
                else:
                    row.append("e")
            if row:
                board.append(row)
        return board

    def render(self) -> Image.Image:
        return self._render_trees_and_tents(
            self._puzzle_board,
            self._row_clues,
            self._col_clues,
            cell_px=self._cell_px,
            padding=self._padding,
        )

    def _render_trees_and_tents(
        self,
        puzzle: list[list[str]],
        row_clues: list[int],
        col_clues: list[int],
        cell_px: int = 60,
        padding: int = 50,
        bg: tuple[int, int, int] = (220, 245, 220),  # Light green grass
        fg: tuple[int, int, int] = (20, 20, 20),
        grid: tuple[int, int, int] = (150, 180, 150),
        tree_color: tuple[int, int, int] = (34, 139, 34),
    ) -> Image.Image:
        n = self._size
        size = padding * 2 + cell_px * n
        img = Image.new("RGB", (size, size), bg)
        draw = ImageDraw.Draw(img)

        font_path = self.assets_dir / "DejaVuSans.ttf"
        if font_path.exists():
            font = ImageFont.truetype(str(font_path), int(cell_px * 0.35))
        else:
            font = ImageFont.load_default()

        # Draw grid
        for i in range(n + 1):
            x = padding + i * cell_px
            y = padding + i * cell_px
            draw.line([(padding, y), (padding + n * cell_px, y)], fill=grid, width=2)
            draw.line([(x, padding), (x, padding + n * cell_px)], fill=grid, width=2)

        # Draw trees (simple: trunk + round foliage)
        for r in range(n):
            for c in range(n):
                if puzzle[r][c] == "tr":
                    x = padding + c * cell_px
                    y = padding + r * cell_px
                    cx = x + cell_px // 2
                    cy = y + cell_px // 2

                    # Tree trunk (brown rectangle)
                    trunk_w = cell_px // 5
                    draw.rectangle(
                        [
                            cx - trunk_w // 2,
                            cy + cell_px // 6,
                            cx + trunk_w // 2,
                            cy + cell_px // 2.5,
                        ],
                        fill=(101, 67, 33),
                        outline=(70, 45, 20),
                        width=2,
                    )

                    # Tree foliage (big green circle)
                    foliage_r = cell_px // 3
                    draw.ellipse(
                        [
                            cx - foliage_r,
                            cy - foliage_r - cell_px // 6,
                            cx + foliage_r,
                            cy + foliage_r - cell_px // 6,
                        ],
                        fill=(50, 150, 50),
                        outline=(30, 100, 30),
                        width=3,
                    )

                elif puzzle[r][c] == "tt":
                    # Draw tent (triangle ⛺)
                    x = padding + c * cell_px
                    y = padding + r * cell_px
                    cx = x + cell_px // 2
                    margin = cell_px // 5

                    # Tent triangle (orange/red)
                    points = [
                        (cx, y + margin),  # top
                        (x + margin, y + cell_px - margin),  # bottom-left
                        (x + cell_px - margin, y + cell_px - margin),  # bottom-right
                    ]
                    draw.polygon(
                        points, fill=(255, 140, 0), outline=(200, 100, 0), width=3
                    )

                    # Tent entrance (dark rectangle)
                    entrance_w = cell_px // 6
                    entrance_h = cell_px // 4
                    draw.rectangle(
                        [
                            cx - entrance_w // 2,
                            y + cell_px - margin - entrance_h,
                            cx + entrance_w // 2,
                            y + cell_px - margin,
                        ],
                        fill=(80, 50, 30),
                    )

        # Draw row clues (left) with background circle
        for i in range(n):
            text = str(row_clues[i])
            y = padding + i * cell_px + cell_px // 2
            # Circle background
            circle_r = cell_px // 5
            draw.ellipse(
                [
                    padding // 3 - circle_r,
                    y - circle_r,
                    padding // 3 + circle_r,
                    y + circle_r,
                ],
                fill=(255, 200, 100),
                outline=(200, 150, 50),
                width=2,
            )
            draw.text(
                (padding // 3, y), text, fill=(40, 30, 20), font=font, anchor="mm"
            )

        # Draw col clues (top) with background circle
        for j in range(n):
            text = str(col_clues[j])
            x = padding + j * cell_px + cell_px // 2
            # Circle background
            circle_r = cell_px // 5
            draw.ellipse(
                [
                    x - circle_r,
                    padding // 3 - circle_r,
                    x + circle_r,
                    padding // 3 + circle_r,
                ],
                fill=(255, 200, 100),
                outline=(200, 150, 50),
                width=2,
            )
            draw.text(
                (x, padding // 3), text, fill=(40, 30, 20), font=font, anchor="mm"
            )

        return img

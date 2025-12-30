"""Hitori single-turn environment backed by VGRP-Bench."""

from __future__ import annotations

from importlib import resources
import random
from textwrap import dedent
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation, get_logger

from .vgrp_factories import HitoriPuzzleFactory

logger = get_logger()


class VGRPHitoriEnv(Env):
    """Hitori puzzle using VGRP-Bench's factory.

    Shade some cells so that:
    - No number appears more than once in any row or column (considering only unshaded cells).
    - Shaded cells cannot be adjacent (vertically or horizontally).
    - All unshaded cells must be connected.

    Args:
        size: Grid size (default 6)
        cell_px: Size of each cell in pixels for rendering
        padding: Padding around the grid
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    def __init__(
        self,
        size: int = 6,
        cell_px: int = 50,
        padding: int = 20,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._size = size
        self._cell_px = cell_px
        self._padding = padding

        self._seed: int | None = None
        self._factory = HitoriPuzzleFactory(size)

        self._solution_board: list[list[str]] | None = (
            None  # 'e' (empty/white), 's' (shaded)
        )
        self._number_grid: list[list[int]] | None = None  # The numbers shown to player

        # Puzzle state for player is just the numbers. The action is the shading pattern.
        # So we don't store a "puzzle_board" that changes state,
        # but in gym-v convention, obs.image usually reflects current state if interactive.
        # But this is single-turn: input is text solution.
        # The prompt asks for solution grid (e.g. 1 0 1 ... where 1=shaded?).
        # VGRP-Bench factory uses 'e' and 's'.

    @property
    def description(self) -> str:
        return dedent(f"""
            Solve this {self._size}x{self._size} Hitori puzzle.

            Rules:
            1. Shade cells so that no number appears more than once in a row or column (unshaded cells only).
            2. Shaded cells cannot be adjacent (share a side).
            3. All unshaded cells must form a single connected area.

            Output format: A {self._size}x{self._size} grid with 's' for Shaded and 'e' for Empty (Unshaded).
        """).strip()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)
        self._seed = seed
        if seed is not None:
            np.random.seed(seed)

        # 1. Generate Solution Pattern (e/s)
        self._solution_board = self._generate_hitori_pattern()

        # 2. Fill Numbers
        self._number_grid = self._generate_numbers(self._solution_board)

        logger.info("Reset VGRP Hitori.")

        obs = Observation(
            image=self.render(),
            text=self._board_to_text_numbers(),
            metadata={"size": self._size},
        )
        info = {
            "oracle_answer": self._board_to_text_solution(self._solution_board),
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
            text=self._board_to_text_numbers(),
            metadata={"size": self._size},
        )
        info = {
            "oracle_answer": self._board_to_text_solution(self._solution_board),
        }
        return obs, reward, True, False, info

    def _generate_hitori_pattern(self) -> list[list[str]]:
        # Randomly shade cells ~30%
        board = [["e" for _ in range(self._size)] for _ in range(self._size)]

        cells = [(r, c) for r in range(self._size) for c in range(self._size)]
        np.random.shuffle(cells)

        for r, c in cells:
            # Try shading
            board[r][c] = "s"

            # Check adjacency constraint
            valid_adj = True
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self._size and 0 <= nc < self._size:
                    if board[nr][nc] == "s":
                        valid_adj = False
                        break
            if not valid_adj:
                board[r][c] = "e"
                continue

            # Check connectivity
            if not self._check_connectivity(board):
                board[r][c] = "e"
                continue

        return board

    def _check_connectivity(self, board):
        # BFS
        start_node = None
        white_count = 0
        for r in range(self._size):
            for c in range(self._size):
                if board[r][c] == "e":
                    white_count += 1
                    if start_node is None:
                        start_node = (r, c)

        if white_count == 0:
            return True  # trivial
        if start_node is None:
            return False  # Should not happen if size > 0

        q = [start_node]
        visited = {start_node}
        count = 0
        while q:
            curr = q.pop(0)
            count += 1
            cr, cc = curr
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < self._size and 0 <= nc < self._size:
                    if board[nr][nc] == "e" and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        q.append((nr, nc))

        return count == white_count

    def _generate_numbers(self, pattern: list[list[str]]) -> list[list[int]]:
        # Fill unshaded cells (Partial Latin Square)
        grid = [[0 for _ in range(self._size)] for _ in range(self._size)]

        # Simple backtracking to fill unshaded
        def solve(idx):
            if idx >= self._size * self._size:
                return True
            r, c = idx // self._size, idx % self._size

            if pattern[r][c] == "s":
                return solve(idx + 1)

            # Try values 1..N
            vals = list(range(1, self._size + 1))
            np.random.shuffle(vals)

            for v in vals:
                # Check row/col for unshaded cells
                valid = True
                for k in range(self._size):
                    if grid[r][k] == v and pattern[r][k] == "e":
                        valid = False
                        break
                    if grid[k][c] == v and pattern[k][c] == "e":
                        valid = False
                        break

                if valid:
                    grid[r][c] = v
                    if solve(idx + 1):
                        return True
                    grid[r][c] = 0
            return False

        if not solve(0):
            # Fallback (should ideally retry pattern generation)
            return [[1] * self._size] * self._size

        # Fill shaded cells with conflicting numbers
        for r in range(self._size):
            for c in range(self._size):
                if pattern[r][c] == "s":
                    # Pick a number that exists in unshaded row OR col
                    candidates = []
                    # Row candidates
                    for k in range(self._size):
                        if pattern[r][k] == "e":
                            candidates.append(grid[r][k])
                    # Col candidates
                    for k in range(self._size):
                        if pattern[k][c] == "e":
                            candidates.append(grid[k][c])

                    if candidates:
                        grid[r][c] = random.choice(candidates)
                    else:
                        grid[r][c] = random.randint(1, self._size)

        return grid

    def _check_solution(self, answer_board: list[list[str]]) -> bool:
        if len(answer_board) != self._size:
            return False
        for i in range(self._size):
            if len(answer_board[i]) != self._size:
                return False
            for j in range(self._size):
                if answer_board[i][j] != self._solution_board[i][j]:
                    return False
        return True

    def _board_to_text_numbers(self) -> str:
        return "\n".join(" ".join(str(x) for x in row) for row in self._number_grid)

    def _board_to_text_solution(self, board: list[list[str]]) -> str:
        return "\n".join(" ".join(row) for row in board)

    def _text_to_board(self, text: str) -> list[list[str]]:
        lines = text.strip().split("\n")
        board = []
        for line in lines:
            line = line.strip()
            row = []
            for val in line.split():
                val = val.lower()
                if val in ["s", "e"]:
                    row.append(val)
                else:
                    row.append("e")
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
                font = ImageFont.truetype(str(font_path), size=20)
        except Exception:
            pass

        for r in range(self._size):
            for c in range(self._size):
                x = self._padding + c * self._cell_px
                y = self._padding + r * self._cell_px

                # Draw cell
                draw.rectangle(
                    [x, y, x + self._cell_px, y + self._cell_px],
                    outline=(0, 0, 0),
                    width=1,
                )

                # Draw Number
                val = self._number_grid[r][c]
                bbox = draw.textbbox((0, 0), str(val), font=font)
                w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                draw.text(
                    (x + (self._cell_px - w) / 2, y + (self._cell_px - h) / 2),
                    str(val),
                    fill=(0, 0, 0),
                    font=font,
                )

        return img

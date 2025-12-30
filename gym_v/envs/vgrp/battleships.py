"""Battleships single-turn environment backed by VGRP-Bench."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation, get_logger

from .vgrp_factories import BattleshipsPuzzleFactory

logger = get_logger()


class VGRPBattleshipsEnv(Env):
    """Battleships puzzle using VGRP-Bench's Battleships puzzle generator.

    Place ships in grid according to clues.

    Args:
        size: Grid size (default 6)
        num_hints: Not used
        cell_px: Size of each cell in pixels for rendering
        padding: Padding around the grid in pixels for rendering
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    def __init__(
        self,
        size: int = 6,
        num_hints: int = 0,
        cell_px: int = 55,
        padding: int = 50,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._size = size
        self._cell_px = cell_px
        self._padding = padding

        self._seed: int | None = None
        self._factory = BattleshipsPuzzleFactory(size)
        self._puzzle_board: list[list[str]] | None = None
        self._solution_board: list[list[str]] | None = None
        self._row_hints: list[int] | None = None
        self._col_hints: list[int] | None = None
        self._ships: dict[int, int] | None = None  # {length: count}

    @property
    def description(self) -> str:
        """Return description for Battleships puzzle."""
        ship_desc = ", ".join(
            [f"{count}x{length}-cell" for length, count in sorted(self._ships.items())]
        )
        return dedent(f"""
            Solve this {self._size}x{self._size} Battleships puzzle.

            In the image:
            - Numbers on left show ship cells per row
            - Numbers on top show ship cells per column
            - Ship fleet: {ship_desc}

            Rules:
            1. Place all ships in the grid (horizontal or vertical)
            2. Ships cannot touch each other, even diagonally
            3. Row and column ship counts must match the given clues
            4. Ships can be 1-4 cells long

            Output format: A {self._size}x{self._size} grid with 's' (ship) or 'e' (empty/water)
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

        # Define ship fleet (standard for 6x6)
        if self._size <= 6:
            self._ships = {
                4: 1,
                3: 1,
                2: 1,
                1: 1,
            }  # 1 battleship, 1 cruiser, 1 destroyer, 1 submarine
        else:
            self._ships = {4: 1, 3: 2, 2: 2, 1: 2}

        # Generate solution directly
        self._solution_board = self._generate_ships_solution()

        # Calculate row/col hints from solution
        self._row_hints = [
            sum(1 for cell in row if cell == "s") for row in self._solution_board
        ]
        self._col_hints = [
            sum(1 for i in range(self._size) if self._solution_board[i][j] == "s")
            for j in range(self._size)
        ]

        self._puzzle_board = [[0 for _ in range(self._size)] for _ in range(self._size)]

        logger.info("Reset VGRP Battleships.")

        obs = Observation(
            image=self.render(),
            text=self._board_to_text_with_clues(),
            metadata={"size": self._size},
        )
        info = {
            "oracle_answer": self._board_to_text(self._solution_board),
        }
        return obs, info

    def _generate_ships_solution(self) -> list[list[str]]:
        """Generate solution by placing ships."""
        solution = [["e" for _ in range(self._size)] for _ in range(self._size)]

        # Place ships
        for length, count in sorted(self._ships.items(), reverse=True):
            for _ in range(count):
                placed = False
                attempts = 0
                while not placed and attempts < 100:
                    # Random position and orientation
                    horizontal = np.random.random() < 0.5
                    if horizontal:
                        r = np.random.randint(0, self._size)
                        c = np.random.randint(0, self._size - length + 1)
                        # Check if can place
                        can_place = all(
                            solution[r][c + i] == "e" for i in range(length)
                        )
                        if can_place:
                            for i in range(length):
                                solution[r][c + i] = "s"
                            placed = True
                    else:
                        r = np.random.randint(0, self._size - length + 1)
                        c = np.random.randint(0, self._size)
                        # Check if can place
                        can_place = all(
                            solution[r + i][c] == "e" for i in range(length)
                        )
                        if can_place:
                            for i in range(length):
                                solution[r + i][c] = "s"
                            placed = True
                    attempts += 1

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
        info = {
            "oracle_answer": self._board_to_text(self._solution_board),
        }
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
        ship_desc = ", ".join(
            [f"{count}x{length}-cell" for length, count in sorted(self._ships.items())]
        )
        lines.append(f"Ships: {ship_desc}")
        lines.append(f"Row clues: {' '.join(map(str, self._row_hints))}")
        lines.append(f"Col clues: {' '.join(map(str, self._col_hints))}")
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
            if not line or ":" in line:
                continue
            row = []
            for val in line.split():
                val = val.strip().lower()
                if val in ["s", "e"]:
                    row.append(val)
                else:
                    row.append("e")
            if row:
                board.append(row)
        return board

    def render(self) -> Image.Image:
        return self._render_battleships(
            self._puzzle_board,
            self._row_hints,
            self._col_hints,
            self._ships,
            cell_px=self._cell_px,
            padding=self._padding,
        )

    def _render_battleships(
        self,
        puzzle: list[list[str | int]],
        row_hints: list[int],
        col_hints: list[int],
        ships: dict[int, int],
        cell_px: int = 55,
        padding: int = 50,
        bg: tuple[int, int, int] = (250, 250, 250),
        water_color: tuple[int, int, int] = (200, 220, 255),
        fg: tuple[int, int, int] = (20, 20, 20),
        grid: tuple[int, int, int] = (100, 100, 100),
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

        # Draw water background with wave pattern
        for r in range(n):
            for c in range(n):
                x = padding + c * cell_px
                y = padding + r * cell_px

                # Gradient water effect
                water_base = (180, 210, 255)
                water_light = (210, 230, 255)
                # Alternate pattern
                if (r + c) % 2 == 0:
                    draw.rectangle(
                        [x + 1, y + 1, x + cell_px - 1, y + cell_px - 1],
                        fill=water_base,
                    )
                else:
                    draw.rectangle(
                        [x + 1, y + 1, x + cell_px - 1, y + cell_px - 1],
                        fill=water_light,
                    )

                # Draw wave lines for water effect
                wave_y1 = y + cell_px // 3
                wave_y2 = y + 2 * cell_px // 3
                draw.line(
                    [(x + 5, wave_y1), (x + cell_px - 5, wave_y1)],
                    fill=(160, 190, 235),
                    width=1,
                )
                draw.line(
                    [(x + 5, wave_y2), (x + cell_px - 5, wave_y2)],
                    fill=(160, 190, 235),
                    width=1,
                )

        # Draw grid
        for i in range(n + 1):
            x = padding + i * cell_px
            y = padding + i * cell_px
            draw.line([(padding, y), (padding + n * cell_px, y)], fill=grid, width=2)
            draw.line([(x, padding), (x, padding + n * cell_px)], fill=grid, width=2)

        # Draw row hints (left)
        for i in range(n):
            text = str(row_hints[i])
            y = padding + i * cell_px + cell_px // 2
            bbox = draw.textbbox((0, 0), text, font=font)
            text_height = bbox[3] - bbox[1]
            draw.text((padding // 3, y - text_height // 2), text, fill=fg, font=font)

        # Draw col hints (top)
        for j in range(n):
            text = str(col_hints[j])
            x = padding + j * cell_px + cell_px // 2
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            draw.text((x - text_width // 2, padding // 4), text, fill=fg, font=font)

        return img

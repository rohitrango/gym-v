"""Thermometers single-turn environment backed by VGRP-Bench."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation, get_logger

from .vgrp_factories import ThermometersPuzzleFactory

logger = get_logger()


class VGRPThermometersEnv(Env):
    """Thermometers puzzle using VGRP-Bench's Thermometers puzzle generator.

    Fill thermometers from the bulb according to row/column clues.

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
        self._factory = ThermometersPuzzleFactory(size)
        self._puzzle_board: list[list[str]] | None = None
        self._solution_board: list[list[str]] | None = None
        self._thermometers: list[list[tuple[int, int]]] | None = None
        self._row_counts: list[int] | None = None
        self._col_counts: list[int] | None = None

    @property
    def description(self) -> str:
        """Return description for Thermometers puzzle."""
        return dedent(f"""
            Solve this {self._size}x{self._size} Thermometers puzzle.

            In the image:
            - Thermometer shapes are shown with bulbs (red circles) at one end
            - Numbers on left show required filled cells per row
            - Numbers on top show required filled cells per column

            Rules:
            1. Each thermometer can be filled 0 to full length (you choose how much to fill)
            2. If filling, must start from the bulb (circular end) and fill continuously
            3. You can: not fill at all, fill only the bulb, or fill from bulb continuously
            4. You cannot skip cells - must fill from bulb without gaps
            5. Row and column counts must match the given clues
            6. Only thermometer cells can be filled

            Output format: A {self._size}x{self._size} grid with 's' (filled) or 'e' (empty)
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

        # Generate thermometers configuration
        self._thermometers = self._generate_thermometers()

        # Generate solution directly (not via backtracking due to circular dependency)
        self._solution_board = self._generate_thermometer_solution()

        # Calculate row/col counts from solution
        self._row_counts = [
            sum(1 for cell in row if cell == "s") for row in self._solution_board
        ]
        self._col_counts = [
            sum(1 for i in range(self._size) if self._solution_board[i][j] == "s")
            for j in range(self._size)
        ]

        self._puzzle_board = [[0 for _ in range(self._size)] for _ in range(self._size)]

        logger.info("Reset VGRP Thermometers.")

        obs = Observation(
            image=self.render(),
            text=self._board_to_text_with_clues(),
            metadata={"size": self._size},
        )
        info = {
            "oracle_answer": self._board_to_text(self._solution_board),
        }
        return obs, info

    def _generate_thermometers(self) -> list[list[tuple[int, int]]]:
        """Generate random thermometer configurations."""
        thermometers = []
        used_cells = set()

        # Generate 3-5 thermometers
        num_thermos = np.random.randint(3, min(6, self._size + 1))

        for _ in range(num_thermos):
            # Random starting position
            attempts = 0
            while attempts < 50:
                start_r = np.random.randint(0, self._size)
                start_c = np.random.randint(0, self._size)

                if (start_r, start_c) in used_cells:
                    attempts += 1
                    continue

                # Build thermometer path
                thermo = [(start_r, start_c)]
                used_cells.add((start_r, start_c))

                # Random length 2-4
                length = np.random.randint(2, min(5, self._size))

                # Random direction
                for _step in range(1, length):
                    last_r, last_c = thermo[-1]

                    # Try directions: right, down, left, up
                    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
                    np.random.shuffle(directions)

                    added = False
                    for dr, dc in directions:
                        new_r, new_c = last_r + dr, last_c + dc
                        if (
                            0 <= new_r < self._size
                            and 0 <= new_c < self._size
                            and (new_r, new_c) not in used_cells
                        ):
                            thermo.append((new_r, new_c))
                            used_cells.add((new_r, new_c))
                            added = True
                            break

                    if not added:
                        break

                if len(thermo) >= 2:
                    thermometers.append(thermo)
                    break
                else:
                    # Remove from used if failed
                    for cell in thermo:
                        used_cells.discard(cell)

                attempts += 1

        return thermometers if thermometers else [[(0, 0), (0, 1)]]

    def _generate_thermometer_solution(self) -> list[list[str]]:
        """Generate solution by randomly filling thermometers."""
        solution = [["e" for _ in range(self._size)] for _ in range(self._size)]

        # For each thermometer, randomly decide how many cells to fill (from bulb)
        for thermo in self._thermometers:
            fill_count = np.random.randint(0, len(thermo) + 1)
            for idx in range(fill_count):
                r, c = thermo[idx]
                solution[r][c] = "s"

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
        lines.append(f"Row clues: {' '.join(map(str, self._row_counts))}")
        lines.append(f"Col clues: {' '.join(map(str, self._col_counts))}")
        lines.append("")
        lines.append(f"Thermometers count: {len(self._thermometers)}")
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
        return self._render_thermometers(
            self._puzzle_board,
            self._thermometers,
            self._row_counts,
            self._col_counts,
            cell_px=self._cell_px,
            padding=self._padding,
        )

    def _render_thermometers(
        self,
        puzzle: list[list[str | int]],
        thermometers: list[list[tuple[int, int]]],
        row_counts: list[int],
        col_counts: list[int],
        cell_px: int = 60,
        padding: int = 50,
        bg: tuple[int, int, int] = (245, 245, 250),
        fg: tuple[int, int, int] = (20, 20, 20),
        grid: tuple[int, int, int] = (180, 180, 200),
    ) -> Image.Image:
        n = self._size
        size = padding * 2 + cell_px * n
        img = Image.new("RGB", (size, size), bg)
        draw = ImageDraw.Draw(img)

        font_path = self.assets_dir / "DejaVuSans.ttf"
        if font_path.exists():
            font = ImageFont.truetype(str(font_path), int(cell_px * 0.4))
        else:
            font = ImageFont.load_default()

        # Draw thermometers with gradient effect
        for thermo_idx, thermo in enumerate(thermometers):
            # Alternate colors for different thermometers
            colors = [
                (180, 210, 255),  # Light blue
                (255, 210, 180),  # Light orange
                (210, 255, 180),  # Light green
                (255, 180, 210),  # Light pink
                (210, 180, 255),  # Light purple
            ]
            thermo_color = colors[thermo_idx % len(colors)]

            for idx, (r, c) in enumerate(thermo):
                x = padding + c * cell_px
                y = padding + r * cell_px

                # Draw thermometer tube with rounded corners and shadow
                # Shadow
                draw.rectangle(
                    [x + 3, y + 3, x + cell_px - 1, y + cell_px - 1],
                    fill=(200, 200, 210),
                    outline=None,
                )
                # Main tube
                draw.rounded_rectangle(
                    [x + 1, y + 1, x + cell_px - 3, y + cell_px - 3],
                    radius=8,
                    fill=thermo_color,
                    outline=(120, 120, 150),
                    width=2,
                )

                # Draw bulb at start (glossy circle with highlight)
                if idx == 0:
                    margin = cell_px // 5
                    # Shadow
                    draw.ellipse(
                        [
                            x + margin + 2,
                            y + margin + 2,
                            x + cell_px - margin,
                            y + cell_px - margin,
                        ],
                        fill=(180, 80, 80),
                    )
                    # Main bulb
                    draw.ellipse(
                        [
                            x + margin,
                            y + margin,
                            x + cell_px - margin - 2,
                            y + cell_px - margin - 2,
                        ],
                        fill=(255, 100, 100),
                        outline=(180, 50, 50),
                        width=3,
                    )
                    # Highlight for glossy effect
                    highlight_size = cell_px // 8
                    draw.ellipse(
                        [
                            x + margin + cell_px // 6,
                            y + margin + cell_px // 6,
                            x + margin + cell_px // 6 + highlight_size,
                            y + margin + cell_px // 6 + highlight_size,
                        ],
                        fill=(255, 200, 200),
                    )

        # Draw grid
        for i in range(n + 1):
            x = padding + i * cell_px
            y = padding + i * cell_px
            draw.line([(padding, y), (padding + n * cell_px, y)], fill=grid, width=2)
            draw.line([(x, padding), (x, padding + n * cell_px)], fill=grid, width=2)

        # Draw row clues (left)
        for i in range(n):
            text = str(row_counts[i])
            y = padding + i * cell_px + cell_px // 2
            bbox = draw.textbbox((0, 0), text, font=font)
            text_height = bbox[3] - bbox[1]
            draw.text((padding // 3, y - text_height // 2), text, fill=fg, font=font)

        # Draw col clues (top)
        for j in range(n):
            text = str(col_counts[j])
            x = padding + j * cell_px + cell_px // 2
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            draw.text((x - text_width // 2, padding // 3), text, fill=fg, font=font)

        return img

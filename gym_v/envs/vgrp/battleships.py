"""Battleships single-turn environment backed by VGRP-Bench."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation, get_logger

from .utils import generate_puzzle
from .vgrp_base import Constraint, PuzzleFactory

logger = get_logger()


class ConstraintBattleships(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_battleships"

    def check(self, game_state: dict[str, Any]) -> bool:
        board = game_state["board"]
        size = len(board)

        # Check if ships touch diagonally or orthogonally
        for i in range(size):
            for j in range(size):
                if isinstance(board[i][j], tuple):
                    # Revealed ship with direction logic (omitted for brevity if simpler logic suffices,
                    # but keeping structure)
                    _, direction = board[i][j]
                    if direction in "<>-":
                        for di in [-1, 1]:
                            if 0 <= i + di < size and board[i + di][j] == "s":
                                return False
                    elif direction in "^V|":
                        for dj in [-1, 1]:
                            if 0 <= j + dj < size and board[i][j + dj] == "s":
                                return False
                elif board[i][j] == "s":
                    # Regular ship cell checks
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if (
                                0 <= ni < size
                                and 0 <= nj < size
                                and (
                                    board[ni][nj] == "s"
                                    or (
                                        isinstance(board[ni][nj], tuple)
                                        and board[ni][nj][0] == "s"
                                    )
                                )
                                and (di != 0 and dj != 0)
                            ):  # Diagonal check
                                return False
        return True


class ConstraintBattleshipsHints(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_battleships_hints"

    def check(self, game_state: dict[str, Any]) -> bool:
        board = game_state["board"]
        hints = game_state["hints"]

        row_hints = hints["row_hints"]
        col_hints = hints["col_hints"]
        ships = hints["ships"]
        size = len(board)

        # Calculate total required ship cells
        total_ship_cells_required = sum(
            int(length) * int(count) for length, count in ships.items()
        )
        total_ship_cells_selected = sum(
            1 for i in range(size) for j in range(size) if board[i][j] == "s"
        )
        total_undefined_cells = sum(
            1 for i in range(size) for j in range(size) if board[i][j] == 0
        )

        if (
            total_ship_cells_selected + total_undefined_cells
            < total_ship_cells_required
        ):
            return False

        if total_ship_cells_selected > total_ship_cells_required:
            return False

        # Check row hints
        for i in range(size):
            row_selected = sum(1 for j in range(size) if board[i][j] == "s")
            row_undefined = sum(1 for j in range(size) if board[i][j] == 0)
            if all(cell != 0 and cell != -1 for cell in board[i]):
                if row_selected != row_hints[i]:
                    return False
            else:
                if row_selected > row_hints[i]:
                    return False
                if row_selected + row_undefined < row_hints[i]:
                    return False

        # Check col hints
        for j in range(size):
            col_selected = sum(1 for i in range(size) if board[i][j] == "s")
            col_undefined = sum(1 for i in range(size) if board[i][j] == 0)
            if all(board[i][j] != 0 and board[i][j] != -1 for i in range(size)):
                if col_selected != col_hints[j]:
                    return False
            else:
                if col_selected > col_hints[j]:
                    return False
                if col_selected + col_undefined < col_hints[j]:
                    return False

        # Check ship shapes when full
        if total_undefined_cells == 0:
            visited = [[False] * size for _ in range(size)]
            ship_lengths = []

            def get_ship_length(i: int, j: int) -> int:
                if (
                    i < 0
                    or i >= size
                    or j < 0
                    or j >= size
                    or visited[i][j]
                    or board[i][j] != "s"
                ):
                    return 0
                visited[i][j] = True
                length = 1
                if j + 1 < size and board[i][j + 1] == "s":
                    for col in range(j + 1, size):
                        if board[i][col] != "s":
                            break
                        visited[i][col] = True
                        length += 1
                elif i + 1 < size and board[i + 1][j] == "s":
                    for row in range(i + 1, size):
                        if board[row][j] != "s":
                            break
                        visited[row][j] = True
                        length += 1
                return length

            for i in range(size):
                for j in range(size):
                    if not visited[i][j] and board[i][j] == "s":
                        ship_lengths.append(get_ship_length(i, j))

            ship_counts = {}
            for length in ship_lengths:
                ship_counts[length] = ship_counts.get(length, 0) + 1

            for length, count in ships.items():
                if ship_counts.get(int(length), 0) != int(count):
                    return False

        return True


class BattleshipsPuzzleFactory(PuzzleFactory):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.game_name = "battleships"
        self.size = size
        self.constraints = [ConstraintBattleships(), ConstraintBattleshipsHints()]
        self.all_possible_values = ["e", "s"]

    def get_possible_values(
        self, game_state: dict[str, Any], row: int, col: int
    ) -> list[int]:
        board = game_state["board"]
        if board[row][col] != 0:
            return []
        possible_values = []
        original_value = board[row][col]
        for value in self.all_possible_values:
            board[row][col] = value
            if self.check(game_state):
                possible_values.append(value)
        board[row][col] = original_value
        return possible_values


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
        self._puzzle_board: list[list[str]] | None = None
        self._solution_board: list[list[str]] | None = None
        self._row_hints: list[int] | None = None
        self._col_hints: list[int] | None = None
        self._ships: dict[int, int] | None = None  # {length: count}
        self._factory = BattleshipsPuzzleFactory(size)

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
            Example for 6x6:
            e e s s e e
            s s s e e e
            e e e e e e
            s e s s s s
            e e e e e e
            e e s s s e
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
            self._ships = {4: 1, 3: 1, 2: 1, 1: 1}
        else:
            self._ships = {4: 1, 3: 2, 2: 2, 1: 2}

        # Use official generation mechanism
        # Temporarily disable hint constraints because we don't have hints yet
        original_constraints = self._factory.constraints
        self._factory.constraints = [
            c for c in original_constraints if c.name == "constraint_battleships"
        ]

        result = generate_puzzle(
            self._factory, self._size, num_hints=0, max_attempts=5000
        )

        # Restore constraints
        self._factory.constraints = original_constraints

        if result is None:
            raise RuntimeError("Failed to generate Battleships puzzle")

        _, self._solution_board = result

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
        """Check if the answer matches the solution or satisfies constraints."""
        if len(answer_board) != self._size:
            return False
        for i in range(self._size):
            if len(answer_board[i]) != self._size:
                return False

        # 1. Check if it matches ground truth (fastest)
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

        # 2. Check VGRP constraints (allows alternative valid solutions)
        game_state = {
            "board": answer_board,
            "hints": {
                "row_hints": self._row_hints,
                "col_hints": self._col_hints,
                "ships": self._ships,
            },
        }
        return self._factory.check(game_state)

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

    def render(self) -> Image.Image | list[Image.Image] | None:
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

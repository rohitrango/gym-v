"""Sudoku game based on GameRL."""

from __future__ import annotations

from importlib import resources
import random
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation, get_logger

logger = get_logger()


class GameRLSudokuEnv(Env):
    """Sudoku game environment.

    Fill the board with colors following Sudoku rules: no duplicate colors
    in rows, columns, or boxes.

    Args:
        size: Board size (4 or 9, default 9)
        difficulty: Puzzle difficulty ('easy', 'medium', 'hard', default 'medium')
        cell_size: Size of each cell in pixels for rendering (default 50)
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    # Color mappings
    COLORS_4 = {
        "red": (229, 144, 144),
        "green": (169, 216, 169),
        "blue": (169, 192, 229),
        "magenta": (229, 169, 229),
    }

    COLORS_9 = {
        "red": (229, 144, 144),
        "green": (169, 216, 169),
        "blue": (169, 192, 229),
        "magenta": (229, 169, 229),
        "yellow": (232, 232, 133),
        "aqua": (109, 237, 226),
        "gray": (105, 105, 105),
        "purple": (178, 150, 224),
        "forest green": (34, 139, 34),
    }

    # Difficulty settings (empty cell ratio)
    DIFFICULTIES = {
        "easy": 0.1,
        "medium": 0.2,
        "hard": 0.3,
    }

    def __init__(
        self,
        size: int = 9,
        difficulty: str = "medium",
        cell_size: int = 50,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if size not in [4, 9]:
            raise ValueError(f"Size must be 4 or 9, got {size}")
        if difficulty not in self.DIFFICULTIES:
            raise ValueError(
                f"Difficulty must be one of {list(self.DIFFICULTIES.keys())}"
            )

        self._size = size
        self._difficulty = difficulty
        self._cell_size = cell_size
        self._box_size = 2 if size == 4 else 3
        self._margin = 30

        # Color names for this size
        if size == 4:
            self._color_names = ["red", "green", "blue", "magenta"]
            self._colors = self.COLORS_4
        else:
            self._color_names = [
                "red",
                "green",
                "blue",
                "magenta",
                "yellow",
                "aqua",
                "gray",
                "purple",
                "forest green",
            ]
            self._colors = self.COLORS_9

        # Game state (initialized in reset)
        self._board: list[list[int]] = []
        self._solution: list[list[int]] = []
        self._initial_board: list[list[int]] = []

    @property
    def description(self) -> str:
        colors_str = ", ".join(self._color_names)
        return dedent(f"""
            This is a sudoku game in which the board is filled with a total number of colours equal to the length of the board's sides, and no rows, columns or squares are allowed to have duplicate colours.

            You should fill the empty cells on the board with following {self._size} colors: {colors_str}.

            In this Sudoku board, the row coordinates are 1-{self._size} from top to bottom, and the column coordinates are 1-{self._size} from left to right.

            Actions:
            - 'fill <row> <col> <color>' or 'f <row> <col> <color>': Fill cell at (row, col) with color
            - 'clear <row> <col>' or 'c <row> <col>': Clear cell at (row, col)

            Example: 'fill 1 1 red' or 'f 1 1 red' to fill top-left cell with red
            Example: 'clear 2 3' or 'c 2 3' to clear cell at row 2, column 3
        """).strip()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)

        # Generate puzzle
        self._solution = self._generate_solved_board()
        self._board, self._initial_board = self._generate_puzzle()

        empty_count = sum(row.count(0) for row in self._board)
        logger.info(
            f"Reset Sudoku ({self._size}x{self._size}, {self._difficulty}, {empty_count} empty cells)."
        )

        obs = Observation(image=self.render(), text=self._get_observation_text())
        return obs, {}

    def inner_step(
        self, action: str
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        info: dict[str, Any] = {}
        reward = 0.0
        terminated = False
        truncated = False

        # Parse action
        action = action.strip().lower()
        parts = action.split()

        if len(parts) < 3:
            logger.warning(f"Invalid action format: {action}")
            obs = Observation(
                image=self.render(),
                text="Invalid action. Use 'fill <row> <col> <color>' or 'clear <row> <col>'",
            )
            return obs, -0.1, False, False, info

        action_type = parts[0]

        try:
            row = int(parts[1]) - 1  # Convert to 0-indexed
            col = int(parts[2]) - 1
        except ValueError:
            logger.warning(f"Invalid coordinates: {parts[1]}, {parts[2]}")
            obs = Observation(
                image=self.render(),
                text="Invalid coordinates. Use integers for row and col.",
            )
            return obs, -0.1, False, False, info

        # Validate coordinates
        if not (0 <= row < self._size and 0 <= col < self._size):
            logger.warning(f"Coordinates out of bounds: ({row + 1}, {col + 1})")
            obs = Observation(
                image=self.render(),
                text=f"Coordinates out of bounds: ({row + 1}, {col + 1})",
            )
            return obs, -0.1, False, False, info

        # Check if cell is modifiable
        if self._initial_board[row][col] != 0:
            logger.warning(
                f"Cell ({row + 1}, {col + 1}) is a given clue, cannot modify"
            )
            obs = Observation(
                image=self.render(),
                text=f"Cell ({row + 1}, {col + 1}) is a given clue and cannot be modified.",
            )
            return obs, -0.1, False, False, info

        # Execute action
        if action_type in ["fill", "f"]:
            if len(parts) < 4:
                obs = Observation(
                    image=self.render(),
                    text="Fill action requires color. Use 'fill <row> <col> <color>'",
                )
                return obs, -0.1, False, False, info

            color_name = " ".join(
                parts[3:]
            )  # Handle multi-word colors like "forest green"
            if color_name not in self._color_names:
                obs = Observation(
                    image=self.render(),
                    text=f"Invalid color '{color_name}'. Valid colors: {', '.join(self._color_names)}",
                )
                return obs, -0.1, False, False, info

            color_value = self._color_names.index(color_name) + 1
            self._board[row][col] = color_value

            # Check if correct
            if color_value == self._solution[row][col]:
                reward = 1.0
            else:
                reward = -0.5

            # Check if puzzle is complete
            if self._is_complete():
                if self._is_correct():
                    terminated = True
                    reward = 100.0
                    logger.info("Sudoku solved correctly!")
                else:
                    obs = Observation(
                        image=self.render(),
                        text="Puzzle complete but incorrect. Keep trying!",
                    )
                    return obs, -10.0, False, False, info

        elif action_type in ["clear", "c"]:
            self._board[row][col] = 0
            reward = 0.0

        else:
            logger.warning(f"Unknown action type: {action_type}")
            obs = Observation(
                image=self.render(),
                text=f"Unknown action: {action_type}. Use 'fill' or 'clear'",
            )
            return obs, -0.1, False, False, info

        filled_count = sum(
            1
            for r in range(self._size)
            for c in range(self._size)
            if self._board[r][c] != 0
        )
        info = {"filled_count": filled_count, "total_cells": self._size * self._size}

        obs = Observation(image=self.render(), text=self._get_observation_text())
        return obs, reward, terminated, truncated, info

    def render(self) -> Image.Image:
        """Render the current board state as a PIL Image."""
        grid_size = self._cell_size * self._size
        padding = self._margin
        img_width = grid_size + 2 * padding
        img_height = grid_size + 2 * padding

        img = Image.new("RGB", (img_width, img_height), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Load font
        try:
            font = ImageFont.truetype(str(self.assets_dir / "DejaVuSans.ttf"), 20)
        except Exception:
            font = ImageFont.load_default()

        # Draw labels
        for i in range(self._size):
            text = str(i + 1)
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # Row labels
            row_x = padding // 2 - text_width // 2
            row_y = padding + i * self._cell_size + (self._cell_size - text_height) // 2
            draw.text((row_x, row_y), text, fill=(0, 0, 0), font=font)

            # Column labels
            col_x = padding + i * self._cell_size + (self._cell_size - text_width) // 2
            col_y = padding // 2 - text_height // 2
            draw.text((col_x, col_y), text, fill=(0, 0, 0), font=font)

        # Fill cells with colors
        for i in range(self._size):
            for j in range(self._size):
                if self._board[i][j] != 0:
                    color_name = self._color_names[self._board[i][j] - 1]
                    color = self._colors[color_name]
                    x = padding + j * self._cell_size
                    y = padding + i * self._cell_size
                    draw.rectangle(
                        [x, y, x + self._cell_size, y + self._cell_size],
                        fill=color,
                    )

        # Draw grid
        for i in range(self._size + 1):
            line_width = 2 if i % self._box_size == 0 else 1
            # Vertical lines
            draw.line(
                [
                    (padding + i * self._cell_size, padding),
                    (padding + i * self._cell_size, padding + grid_size),
                ],
                fill=(0, 0, 0),
                width=line_width,
            )
            # Horizontal lines
            draw.line(
                [
                    (padding, padding + i * self._cell_size),
                    (padding + grid_size, padding + i * self._cell_size),
                ],
                fill=(0, 0, 0),
                width=line_width,
            )

        return img

    def _get_observation_text(self) -> str:
        """Get text description of current state."""
        filled = sum(
            1
            for r in range(self._size)
            for c in range(self._size)
            if self._board[r][c] != 0
        )
        total = self._size * self._size
        given = sum(
            1
            for r in range(self._size)
            for c in range(self._size)
            if self._initial_board[r][c] != 0
        )
        return f"Filled: {filled}/{total} ({filled - given} by player, {given} given)"

    def _generate_solved_board(self) -> list[list[int]]:
        """Generate a complete solved Sudoku board."""
        board = [[0] * self._size for _ in range(self._size)]
        nums = list(range(1, self._size + 1))

        # Fill diagonal boxes
        for i in range(0, self._size, self._box_size):
            box_nums = nums.copy()
            random.shuffle(box_nums)
            for r in range(i, i + self._box_size):
                for c in range(i, i + self._box_size):
                    board[r][c] = box_nums.pop()

        # Solve the rest
        self._solve_board(board)
        return board

    def _solve_board(self, board: list[list[int]]) -> bool:
        """Solve the Sudoku board using backtracking."""
        empty = self._find_empty(board)
        if not empty:
            return True

        row, col = empty
        nums = list(range(1, self._size + 1))
        random.shuffle(nums)

        for num in nums:
            if self._is_valid(board, row, col, num):
                board[row][col] = num
                if self._solve_board(board):
                    return True
                board[row][col] = 0

        return False

    def _find_empty(self, board: list[list[int]]) -> tuple[int, int] | None:
        """Find an empty cell in the board."""
        for i in range(self._size):
            for j in range(self._size):
                if board[i][j] == 0:
                    return (i, j)
        return None

    def _is_valid(self, board: list[list[int]], row: int, col: int, num: int) -> bool:
        """Check if a number is valid at the given position."""
        # Check row
        if num in board[row]:
            return False

        # Check column
        if num in [board[i][col] for i in range(self._size)]:
            return False

        # Check box
        box_row = (row // self._box_size) * self._box_size
        box_col = (col // self._box_size) * self._box_size
        for i in range(box_row, box_row + self._box_size):
            for j in range(box_col, box_col + self._box_size):
                if board[i][j] == num:
                    return False

        return True

    def _generate_puzzle(self) -> tuple[list[list[int]], list[list[int]]]:
        """Generate a puzzle by removing cells from the solved board."""
        puzzle = [row[:] for row in self._solution]
        total_cells = self._size * self._size
        target_empty = int(total_cells * self.DIFFICULTIES[self._difficulty])

        # Remove cells
        cells = [(i, j) for i in range(self._size) for j in range(self._size)]
        random.shuffle(cells)

        removed = 0
        for row, col in cells:
            if removed >= target_empty:
                break
            puzzle[row][col] = 0
            removed += 1

        initial = [row[:] for row in puzzle]
        return puzzle, initial

    def _is_complete(self) -> bool:
        """Check if the board is completely filled."""
        return all(
            self._board[i][j] != 0 for i in range(self._size) for j in range(self._size)
        )

    def _is_correct(self) -> bool:
        """Check if the current board matches the solution."""
        return all(
            self._board[i][j] == self._solution[i][j]
            for i in range(self._size)
            for j in range(self._size)
        )

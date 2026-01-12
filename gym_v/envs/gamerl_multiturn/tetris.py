"""Tetris game based on GameRL."""

from __future__ import annotations

from importlib import resources
import random
from textwrap import dedent
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation, get_logger

logger = get_logger()


class GameRLTetrisEnv(Env):
    """Tetris game environment.

    Classic Tetris puzzle game where players arrange falling tetrominoes.

    Args:
        rows: Number of rows in the grid (default 12)
        cols: Number of columns in the grid (default 8)
        cell_size: Size of each cell in pixels for rendering (default 30)
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    # Define tetromino shapes (I, O, T, L, J, S, Z)
    TETROMINOES = {
        "I": np.array([[1, 1, 1, 1]]),
        "O": np.array([[1, 1], [1, 1]]),
        "T": np.array([[0, 1, 0], [1, 1, 1]]),
        "L": np.array([[1, 0], [1, 0], [1, 1]]),
        "J": np.array([[0, 1], [0, 1], [1, 1]]),
        "S": np.array([[0, 1, 1], [1, 1, 0]]),
        "Z": np.array([[1, 1, 0], [0, 1, 1]]),
    }

    # Tetromino colors
    TETROMINO_COLORS = {
        "I": (0, 255, 255),  # Cyan
        "O": (255, 255, 0),  # Yellow
        "T": (128, 0, 128),  # Purple
        "L": (255, 165, 0),  # Orange
        "J": (0, 0, 255),  # Blue
        "S": (0, 255, 0),  # Green
        "Z": (255, 0, 0),  # Red
    }

    # Colors
    COLORS = {
        "background": (255, 255, 255),  # White
        "grid_line": (0, 0, 0),  # Black
        "placed": (128, 128, 128),  # Gray
        "falling": (255, 0, 0),  # Red for active piece
        "text": (0, 0, 0),  # Black
        "game_over": (255, 0, 0),  # Red
    }

    def __init__(
        self,
        rows: int = 12,
        cols: int = 8,
        cell_size: int = 30,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._rows = rows
        self._cols = cols
        self._cell_size = cell_size
        self._padding = 35  # Padding for coordinate labels

        # Game state (initialized in reset)
        self._grid: np.ndarray = np.zeros((rows, cols), dtype=int)
        self._current_piece: np.ndarray | None = None
        self._current_piece_type: str = ""
        self._current_pos: tuple[int, int] = (0, 0)  # (row, col)
        self._score: int = 0
        self._lines_cleared: int = 0
        self._game_over: bool = False

    @property
    def description(self) -> str:
        return dedent(f"""
            Rules:
            1. The image shows a standard Tetris grid with {self._rows} rows and {self._cols} columns.
            2. The top row of the grid is labeled as Row 0 according to the coordinates.
            3. Row coordinates increment from top to bottom in the grid from 0 to {self._rows-1}.
            4. In the image, empty cells are painted white and a cell is painted grey if in previous moves a tetromino was placed here. Besides, a cell is painted red if a falling active tetromino is here.

            In Tetris, a Tetrimino is a geometric shape composed of four square blocks. There are seven types of Tetriminos represented by the letters I, O, T, L, J, S, and Z:
            - I: A straight line of 4 blocks.
            - O: A square shape composed of 4 blocks arranged in a 2x2 square.
            - T: A T-shape made up of 4 blocks, with one block in the center and the other three forming a horizontal row above it.
            - L: An L-shape made of 4 blocks, with 3 blocks forming a vertical line and 1 block extending right at the bottom, forming an 'L'.
            - J: A J-shape made of 4 blocks, similar to the L-shape but mirrored.
            - S: An S-shape made of 4 blocks arranged in two stacked rows, each row having two blocks arranged diagonally.
            - Z: A Z-shape made of 4 blocks arranged similarly to the S-shape, but mirrored.

            Game Rules:
            1. Tetrominoes can be moved left/right and rotated as they fall.
            2. When a Tetromino lands, it becomes fixed in place.
            3. When a horizontal row is completely filled, it gets cleared.
            4. The game ends when pieces stack up to the top of the grid.

            Available Actions: left, right, rotate, down, drop
        """).strip()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self._grid = np.zeros((self._rows, self._cols), dtype=int)
        self._score = 0
        self._lines_cleared = 0
        self._game_over = False

        # Spawn first piece
        self._spawn_new_piece()

        logger.info("Reset Tetris game.")

        obs = Observation(image=self.render(), text=self._get_observation_text())
        return obs, {}

    def inner_step(
        self, action: str
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        info: dict[str, Any] = {}
        reward = 0.0
        terminated = False
        truncated = False

        if self._game_over:
            obs = Observation(image=self.render(), text=self._get_observation_text())
            return obs, reward, True, truncated, info

        action_lower = action.lower().strip()

        valid_actions = ["left", "right", "rotate", "down", "drop", "a", "d", "w", "s"]
        if action_lower not in valid_actions:
            info["invalid_action"] = True
            obs = Observation(image=self.render(), text=self._get_observation_text())
            return obs, reward, terminated, truncated, info

        info["invalid_action"] = False

        # Map alternative keys
        action_map = {"a": "left", "d": "right", "w": "rotate", "s": "down"}
        action_lower = action_map.get(action_lower, action_lower)

        if action_lower == "left":
            self._move_piece(0, -1)
        elif action_lower == "right":
            self._move_piece(0, 1)
        elif action_lower == "rotate":
            self._rotate_piece()
        elif action_lower == "down":
            if not self._move_piece(1, 0):
                # Piece can't move down, lock it
                lines = self._lock_piece()
                reward = lines * 10.0
                info["lines_cleared"] = lines
                if not self._spawn_new_piece():
                    self._game_over = True
                    terminated = True
        elif action_lower == "drop":
            # Drop piece to bottom
            while self._move_piece(1, 0):
                pass
            lines = self._lock_piece()
            reward = lines * 10.0 + 1.0  # Bonus for dropping
            info["lines_cleared"] = lines
            if not self._spawn_new_piece():
                self._game_over = True
                terminated = True

        info["score"] = self._score
        info["total_lines"] = self._lines_cleared

        obs = Observation(image=self.render(), text=self._get_observation_text())
        return obs, reward, terminated, truncated, info

    def render(self) -> Image.Image | list[Image.Image] | None:
        """Render the game state as a PIL Image."""
        width = self._cols * self._cell_size + 2 * self._padding
        height = self._rows * self._cell_size + 2 * self._padding

        img = Image.new("RGB", (width, height), self.COLORS["background"])
        draw = ImageDraw.Draw(img)

        # Try to load font
        font_path = self.assets_dir / "DejaVuSans.ttf"
        if font_path.exists():
            font = ImageFont.truetype(str(font_path), 16)
        else:
            font = ImageFont.load_default()

        # Draw coordinate labels
        for row in range(self._rows):
            y = row * self._cell_size + self._padding + self._cell_size // 2
            draw.text(
                (self._padding - 20, y - 8),
                str(row),
                fill=self.COLORS["text"],
                font=font,
            )

        for col in range(self._cols):
            x = col * self._cell_size + self._padding + self._cell_size // 2
            draw.text(
                (x - 4, self._padding - 20),
                str(col),
                fill=self.COLORS["text"],
                font=font,
            )

        # Draw grid cells
        for row in range(self._rows):
            for col in range(self._cols):
                x1 = col * self._cell_size + self._padding
                y1 = row * self._cell_size + self._padding
                x2 = x1 + self._cell_size
                y2 = y1 + self._cell_size

                # Fill cell based on grid value
                if self._grid[row][col] == 1:
                    draw.rectangle([x1, y1, x2, y2], fill=self.COLORS["placed"])

                # Draw cell border
                draw.rectangle([x1, y1, x2, y2], outline=self.COLORS["grid_line"])

        # Draw current falling piece
        if self._current_piece is not None:
            piece_row, piece_col = self._current_pos
            for i in range(self._current_piece.shape[0]):
                for j in range(self._current_piece.shape[1]):
                    if self._current_piece[i][j] == 1:
                        row = piece_row + i
                        col = piece_col + j
                        if 0 <= row < self._rows and 0 <= col < self._cols:
                            x1 = col * self._cell_size + self._padding
                            y1 = row * self._cell_size + self._padding
                            x2 = x1 + self._cell_size
                            y2 = y1 + self._cell_size
                            draw.rectangle(
                                [x1, y1, x2, y2], fill=self.COLORS["falling"]
                            )
                            draw.rectangle(
                                [x1, y1, x2, y2], outline=self.COLORS["grid_line"]
                            )

        # Draw game over text
        if self._game_over:
            large_font_path = self.assets_dir / "DejaVuSans.ttf"
            if large_font_path.exists():
                large_font = ImageFont.truetype(str(large_font_path), 28)
            else:
                large_font = ImageFont.load_default()

            text = "GAME OVER"
            bbox = draw.textbbox((0, 0), text, font=large_font)
            text_width = bbox[2] - bbox[0]
            text_x = (width - text_width) // 2
            text_y = height // 2 - 20
            draw.text(
                (text_x, text_y), text, fill=self.COLORS["game_over"], font=large_font
            )

        return img

    def _spawn_new_piece(self) -> bool:
        """Spawn a new tetromino at the top. Returns False if game over."""
        piece_type = random.choice(list(self.TETROMINOES.keys()))
        self._current_piece = self.TETROMINOES[piece_type].copy()
        self._current_piece_type = piece_type

        # Start at top center
        start_col = (self._cols - self._current_piece.shape[1]) // 2
        self._current_pos = (0, start_col)

        # Check if spawn position is valid
        if not self._is_valid_position(self._current_piece, 0, start_col):
            return False
        return True

    def _is_valid_position(self, piece: np.ndarray, row: int, col: int) -> bool:
        """Check if piece can be placed at given position."""
        shape = piece.shape

        # Check bounds
        if row + shape[0] > self._rows or col + shape[1] > self._cols or col < 0:
            return False

        # Check collision with existing blocks
        for i in range(shape[0]):
            for j in range(shape[1]):
                if piece[i][j] == 1:
                    if (
                        row + i >= self._rows
                        or col + j >= self._cols
                        or col + j < 0
                        or self._grid[row + i][col + j] == 1
                    ):
                        return False
        return True

    def _move_piece(self, d_row: int, d_col: int) -> bool:
        """Move piece by delta. Returns True if successful."""
        if self._current_piece is None:
            return False

        new_row = self._current_pos[0] + d_row
        new_col = self._current_pos[1] + d_col

        if self._is_valid_position(self._current_piece, new_row, new_col):
            self._current_pos = (new_row, new_col)
            return True
        return False

    def _rotate_piece(self) -> bool:
        """Rotate piece clockwise. Returns True if successful."""
        if self._current_piece is None:
            return False

        rotated = np.rot90(self._current_piece, -1)  # Clockwise rotation
        if self._is_valid_position(rotated, self._current_pos[0], self._current_pos[1]):
            self._current_piece = rotated
            return True
        return False

    def _lock_piece(self) -> int:
        """Lock current piece into grid and clear lines. Returns lines cleared."""
        if self._current_piece is None:
            return 0

        row, col = self._current_pos
        for i in range(self._current_piece.shape[0]):
            for j in range(self._current_piece.shape[1]):
                if self._current_piece[i][j] == 1:
                    self._grid[row + i][col + j] = 1

        self._current_piece = None

        # Clear full lines
        lines_cleared = self._clear_full_rows()
        self._lines_cleared += lines_cleared
        self._score += lines_cleared * 100

        return lines_cleared

    def _clear_full_rows(self) -> int:
        """Clear full rows and drop blocks above. Returns number cleared."""
        full_rows = []
        for i in range(self._rows):
            if np.all(self._grid[i] == 1):
                full_rows.append(i)

        if full_rows:
            # Remove full rows and add empty rows at top
            self._grid = np.delete(self._grid, full_rows, axis=0)
            empty_rows = np.zeros((len(full_rows), self._cols), dtype=int)
            self._grid = np.vstack([empty_rows, self._grid])

        return len(full_rows)

    def _get_observation_text(self) -> str:
        """Get text description of current game state."""
        if self._game_over:
            return f"Game Over! Score: {self._score}, Lines: {self._lines_cleared}"

        piece_info = (
            f"Piece: {self._current_piece_type}"
            if self._current_piece_type
            else "No piece"
        )
        pos_info = f"Position: {self._current_pos}"
        return f"{piece_info}, {pos_info}, Score: {self._score}, Lines: {self._lines_cleared}"

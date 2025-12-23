"""TicTacToe game environment for gym-v."""

from __future__ import annotations

import re
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation, get_logger

logger = get_logger()


class TicTacToeEnv(Env):
    """TicTacToe (Tic-Tac-Toe) game environment.

    A two-player game where players take turns marking spaces in a 3x3 grid.
    The player who succeeds in placing three marks in a horizontal, vertical,
    or diagonal row wins the game.

    Args:
        cell_size: Size of each cell in pixels for rendering (default 100)
        player_starts: Which player starts first, 'O' or 'X' (default 'O')
    """

    # Winning lines (indices in 1D board)
    WINNING_LINES = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],  # Rows
        [0, 3, 6],
        [1, 4, 7],
        [2, 5, 8],  # Columns
        [0, 4, 8],
        [2, 4, 6],  # Diagonals
    ]

    # Colors
    COLORS = {
        "background": (255, 255, 255),
        "grid": (0, 0, 0),
        "O": (255, 0, 0),  # Red for O
        "X": (0, 0, 255),  # Blue for X
        "text": (0, 0, 0),
    }

    def __init__(
        self,
        cell_size: int = 100,
        player_starts: str = "O",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._cell_size = cell_size
        self._margin = 30  # Margin for coordinate labels
        self._player_starts = player_starts.upper()

        # Game state (initialized in reset)
        self._board: list[str] = []  # ' ', 'O', or 'X'
        self._current_player: str = self._player_starts
        self._winner: str | None = None
        self._game_over: bool = False

    @property
    def description(self) -> str:
        return dedent("""
            TicTacToe (Tic-Tac-Toe): A two-player game on a 3x3 grid.

            Players: O (Red, first) vs X (Blue)

            Actions: Specify position as "row,col" (e.g., "0,0" for top-left)
            Or use position number 0-8:
              0 | 1 | 2
              ---------
              3 | 4 | 5
              ---------
              6 | 7 | 8

            Win condition: Get three marks in a row (horizontal, vertical, or diagonal)
        """).strip()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)

        self._board = [" " for _ in range(9)]
        self._current_player = self._player_starts
        self._winner = None
        self._game_over = False

        logger.info("Reset TicTacToe game.")

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
        position = self._parse_action(action)

        if position is None or position < 0 or position > 8:
            info["invalid_action"] = True
            info["reason"] = "Invalid position format"
            obs = Observation(image=self.render(), text=self._get_observation_text())
            return obs, reward, terminated, truncated, info

        # Check if position is already occupied
        if self._board[position] != " ":
            info["invalid_action"] = True
            info["reason"] = "Position already occupied"
            obs = Observation(image=self.render(), text=self._get_observation_text())
            return obs, reward, terminated, truncated, info

        info["invalid_action"] = False

        # Make move
        self._board[position] = self._current_player
        info["player"] = self._current_player
        info["position"] = (position // 3, position % 3)

        # Check for winner
        winner = self._check_winner()
        if winner:
            self._winner = winner
            self._game_over = True
            terminated = True
            reward = 1.0 if winner == "O" else -1.0  # O wins: +1, X wins: -1
            info["winner"] = winner
        elif " " not in self._board:
            # Draw
            self._game_over = True
            terminated = True
            reward = 0.0
            info["draw"] = True
        else:
            # Switch player
            self._current_player = "X" if self._current_player == "O" else "O"

        obs = Observation(image=self.render(), text=self._get_observation_text())
        return obs, reward, terminated, truncated, info

    def render(self) -> Image.Image:
        """Render the game state as a PIL Image."""
        grid_size = 3 * self._cell_size
        img_width = grid_size + 2 * self._margin
        img_height = grid_size + 2 * self._margin

        img = Image.new("RGB", (img_width, img_height), self.COLORS["background"])
        draw = ImageDraw.Draw(img)

        # Try to load a font
        try:
            font_size = self._cell_size // 3
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size
            )
            coord_font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size // 2
            )
        except OSError:
            font = ImageFont.load_default()
            coord_font = font

        # Draw coordinate labels
        for i in range(3):
            # Column numbers (top)
            x = self._margin + i * self._cell_size + self._cell_size // 2
            draw.text(
                (x, self._margin // 2),
                str(i),
                fill=self.COLORS["text"],
                font=coord_font,
                anchor="mm",
            )
            # Row numbers (left)
            y = self._margin + i * self._cell_size + self._cell_size // 2
            draw.text(
                (self._margin // 2, y),
                str(i),
                fill=self.COLORS["text"],
                font=coord_font,
                anchor="mm",
            )

        # Draw grid lines
        line_width = 3
        for i in range(1, 3):
            # Vertical lines
            x = self._margin + i * self._cell_size
            draw.line(
                [(x, self._margin), (x, self._margin + grid_size)],
                fill=self.COLORS["grid"],
                width=line_width,
            )
            # Horizontal lines
            y = self._margin + i * self._cell_size
            draw.line(
                [(self._margin, y), (self._margin + grid_size, y)],
                fill=self.COLORS["grid"],
                width=line_width,
            )

        # Draw border
        draw.rectangle(
            [
                self._margin,
                self._margin,
                self._margin + grid_size,
                self._margin + grid_size,
            ],
            outline=self.COLORS["grid"],
            width=line_width,
        )

        # Draw pieces
        for pos in range(9):
            piece = self._board[pos]
            if piece == " ":
                continue

            row, col = pos // 3, pos % 3
            cx = self._margin + col * self._cell_size + self._cell_size // 2
            cy = self._margin + row * self._cell_size + self._cell_size // 2

            color = self.COLORS[piece]
            piece_size = self._cell_size // 3

            if piece == "O":
                # Draw circle
                draw.ellipse(
                    [
                        cx - piece_size,
                        cy - piece_size,
                        cx + piece_size,
                        cy + piece_size,
                    ],
                    outline=color,
                    width=5,
                )
            else:  # X
                # Draw X
                draw.line(
                    [
                        (cx - piece_size, cy - piece_size),
                        (cx + piece_size, cy + piece_size),
                    ],
                    fill=color,
                    width=5,
                )
                draw.line(
                    [
                        (cx + piece_size, cy - piece_size),
                        (cx - piece_size, cy + piece_size),
                    ],
                    fill=color,
                    width=5,
                )

        return img

    def _parse_action(self, action: str) -> int | None:
        """Parse action string to board position (0-8)."""
        action = action.strip()

        # Try parsing as single number (0-8)
        if action.isdigit():
            return int(action)

        # Try parsing as "row,col" or "(row,col)" or "[row,col]"
        match = re.match(r"[\[\(]?\s*(\d)\s*[,\s]\s*(\d)\s*[\]\)]?", action)
        if match:
            row, col = int(match.group(1)), int(match.group(2))
            if 0 <= row < 3 and 0 <= col < 3:
                return row * 3 + col

        return None

    def _check_winner(self) -> str | None:
        """Check if there's a winner. Returns 'O', 'X', or None."""
        for line in self.WINNING_LINES:
            if (
                self._board[line[0]]
                == self._board[line[1]]
                == self._board[line[2]]
                != " "
            ):
                return self._board[line[0]]
        return None

    def _get_observation_text(self) -> str:
        """Get text description of current game state."""
        if self._winner:
            return f"Game Over! Winner: {self._winner}"
        if self._game_over:
            return "Game Over! It's a draw!"

        o_count = self._board.count("O")
        x_count = self._board.count("X")
        return f"Current player: {self._current_player} | O: {o_count}, X: {x_count}"

"""Minesweeper game based on GameRL."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation, get_logger

logger = get_logger()


class GameRLMinesweeperEnv(Env):
    """Minesweeper game environment.

    Clear all non-mine cells without hitting a mine. Use numbers to deduce mine
    locations and flag them.

    Args:
        rows: Number of rows (default 8)
        cols: Number of columns (default 8)
        mines: Number of mines (default 10)
        cell_size: Size of each cell in pixels for rendering (default 60)
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    # Number colors
    NUMBER_COLORS = {
        1: (0, 0, 255),  # blue
        2: (0, 128, 0),  # green
        3: (255, 0, 0),  # red
        4: (128, 0, 128),  # purple
        5: (128, 0, 0),  # maroon
        6: (64, 224, 208),  # turquoise
        7: (0, 0, 0),  # black
        8: (128, 128, 128),  # gray
    }

    def __init__(
        self,
        rows: int = 8,
        cols: int = 8,
        mines: int = 10,
        cell_size: int = 60,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._rows = rows
        self._cols = cols
        self._mines = mines
        self._cell_size = cell_size
        self._margin = 40

        # Game state (initialized in reset)
        self._board: list[list[int | str]] = []
        self._revealed: list[list[bool]] = []
        self._flagged: list[list[bool]] = []
        self._first_click = True
        self._game_over = False
        self._won = False

    @property
    def description(self) -> str:
        return dedent(f"""
            This is a Minesweeper game. The size of the chessboard is {self._rows}x{self._cols}, and there are a total of {self._mines} mines hidden on the board.

            The numbers on the board indicate how many mines are adjacent to that cell, including diagonals. Cells marked with "F" (flagged) are identified as potential locations of mines. Cells with no numbers and no flags are safe and contain no adjacent mines.

            The board uses a coordinate system where the top-left cell corresponds to (0,0), and the rows and columns are numbered starting from 0.

            Actions:
            - 'reveal <row> <col>' or 'r <row> <col>': Reveal the cell at (row, col)
            - 'flag <row> <col>' or 'f <row> <col>': Toggle flag on cell at (row, col)

            Example: 'reveal 2 3' or 'r 2 3' to reveal cell at row 2, column 3
            Example: 'flag 1 4' or 'f 1 4' to flag/unflag cell at row 1, column 4
        """).strip()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)

        # Initialize board
        self._board = [[0 for _ in range(self._cols)] for _ in range(self._rows)]
        self._revealed = [[False for _ in range(self._cols)] for _ in range(self._rows)]
        self._flagged = [[False for _ in range(self._cols)] for _ in range(self._rows)]
        self._first_click = True
        self._game_over = False
        self._won = False

        logger.info(
            f"Reset Minesweeper ({self._rows}x{self._cols}, {self._mines} mines)."
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

        if self._game_over:
            obs = Observation(
                image=self.render(),
                text="Game is already over. Reset to play again.",
            )
            return obs, 0.0, True, False, info

        # Parse action
        action = action.strip().lower()
        parts = action.split()

        if len(parts) < 3:
            logger.warning(f"Invalid action format: {action}")
            obs = Observation(
                image=self.render(),
                text=f"Invalid action: {action}. Use 'reveal <row> <col>' or 'flag <row> <col>'",
            )
            return obs, -0.1, False, False, info

        action_type = parts[0]
        try:
            row = int(parts[1])
            col = int(parts[2])
        except ValueError:
            logger.warning(f"Invalid coordinates: {parts[1]}, {parts[2]}")
            obs = Observation(
                image=self.render(),
                text="Invalid coordinates. Use integers for row and col.",
            )
            return obs, -0.1, False, False, info

        # Validate coordinates
        if not (0 <= row < self._rows and 0 <= col < self._cols):
            logger.warning(f"Coordinates out of bounds: ({row}, {col})")
            obs = Observation(
                image=self.render(),
                text=f"Coordinates out of bounds: ({row}, {col})",
            )
            return obs, -0.1, False, False, info

        # Execute action
        if action_type in ["reveal", "r"]:
            success = self._reveal(row, col)
            if not success:
                # Hit a mine
                self._game_over = True
                terminated = True
                reward = -10.0
                logger.info(f"Hit mine at ({row}, {col}) - Game Over!")
            else:
                reward = 0.1
                # Check if won
                if self._is_won():
                    self._won = True
                    self._game_over = True
                    terminated = True
                    reward = 10.0
                    logger.info("All safe cells revealed - You Won!")

        elif action_type in ["flag", "f"]:
            self._toggle_flag(row, col)
            reward = 0.0

        else:
            logger.warning(f"Unknown action type: {action_type}")
            obs = Observation(
                image=self.render(),
                text=f"Unknown action: {action_type}. Use 'reveal' or 'flag'",
            )
            return obs, -0.1, False, False, info

        info = {
            "flagged_count": sum(sum(row) for row in self._flagged),
            "revealed_count": sum(sum(row) for row in self._revealed),
            "remaining_mines": self._mines - sum(sum(row) for row in self._flagged),
        }

        obs = Observation(image=self.render(), text=self._get_observation_text())
        return obs, reward, terminated, truncated, info

    def render(self) -> Image.Image | list[Image.Image] | None:
        """Render the current board state as a PIL Image."""
        # Calculate dimensions
        board_width = self._cols * self._cell_size
        board_height = self._rows * self._cell_size
        title_height = 60
        img_width = board_width + self._margin * 2
        img_height = board_height + self._margin * 2 + title_height

        # Create canvas
        img = Image.new("RGB", (img_width, img_height), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Load fonts
        try:
            title_font = ImageFont.truetype(
                str(self.assets_dir / "DejaVuSans-Bold.ttf"), 36
            )
            cell_font = ImageFont.truetype(
                str(self.assets_dir / "DejaVuSans-Bold.ttf"), int(self._cell_size * 0.5)
            )
            label_font = ImageFont.truetype(str(self.assets_dir / "DejaVuSans.ttf"), 18)
        except Exception:
            title_font = ImageFont.load_default()
            cell_font = ImageFont.load_default()
            label_font = ImageFont.load_default()

        # Draw title
        title = "Minesweeper Board"
        title_bbox = draw.textbbox((0, 0), title, font=title_font)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = (img_width - title_width) // 2
        draw.text((title_x, 15), title, fill=(0, 0, 0), font=title_font)

        offset_x = self._margin
        offset_y = title_height + self._margin

        # Draw row numbers
        for r in range(self._rows):
            text = str(r)
            bbox = draw.textbbox((0, 0), text, font=label_font)
            text_height = bbox[3] - bbox[1]
            y_pos = (
                offset_y + r * self._cell_size + self._cell_size // 2 - text_height // 2
            )
            draw.text((offset_x - 25, y_pos), text, fill=(0, 0, 0), font=label_font)

        # Draw column numbers
        for c in range(self._cols):
            text = str(c)
            bbox = draw.textbbox((0, 0), text, font=label_font)
            text_width = bbox[2] - bbox[0]
            x_pos = (
                offset_x + c * self._cell_size + self._cell_size // 2 - text_width // 2
            )
            draw.text((x_pos, offset_y - 25), text, fill=(0, 0, 0), font=label_font)

        # Draw cells
        for r in range(self._rows):
            for c in range(self._cols):
                x0 = offset_x + c * self._cell_size
                y0 = offset_y + r * self._cell_size
                x1 = x0 + self._cell_size
                y1 = y0 + self._cell_size

                if not self._revealed[r][c]:
                    # Unrevealed cell (gray)
                    color = (192, 192, 192)
                    draw.rectangle([x0, y0, x1, y1], fill=color, outline=(0, 0, 0))
                    if self._flagged[r][c]:
                        # Draw flag
                        bbox = draw.textbbox((0, 0), "F", font=cell_font)
                        text_width = bbox[2] - bbox[0]
                        text_height = bbox[3] - bbox[1]
                        text_x = x0 + self._cell_size // 2 - text_width // 2
                        text_y = y0 + self._cell_size // 2 - text_height // 2
                        draw.text(
                            (text_x, text_y), "F", fill=(255, 0, 0), font=cell_font
                        )
                else:
                    # Revealed cell (white)
                    color = (255, 255, 255)
                    draw.rectangle([x0, y0, x1, y1], fill=color, outline=(0, 0, 0))

                    # Draw number or mine
                    if self._game_over and self._board[r][c] == "M":
                        # Show mine on game over
                        bbox = draw.textbbox((0, 0), "M", font=cell_font)
                        text_width = bbox[2] - bbox[0]
                        text_height = bbox[3] - bbox[1]
                        text_x = x0 + self._cell_size // 2 - text_width // 2
                        text_y = y0 + self._cell_size // 2 - text_height // 2
                        draw.text((text_x, text_y), "M", fill=(0, 0, 0), font=cell_font)
                    elif self._board[r][c] > 0:
                        # Draw number
                        num = str(self._board[r][c])
                        num_color = self.NUMBER_COLORS.get(self._board[r][c], (0, 0, 0))
                        bbox = draw.textbbox((0, 0), num, font=cell_font)
                        text_width = bbox[2] - bbox[0]
                        text_height = bbox[3] - bbox[1]
                        text_x = x0 + self._cell_size // 2 - text_width // 2
                        text_y = y0 + self._cell_size // 2 - text_height // 2
                        draw.text((text_x, text_y), num, fill=num_color, font=cell_font)

        return img

    def _get_observation_text(self) -> str:
        """Get text description of current state."""
        flagged_count = sum(sum(row) for row in self._flagged)
        revealed_count = sum(sum(row) for row in self._revealed)
        remaining = self._mines - flagged_count

        if self._game_over:
            if self._won:
                return f"YOU WON! All {revealed_count} safe cells revealed."
            else:
                return f"GAME OVER! You hit a mine. Revealed: {revealed_count}"
        else:
            return f"Remaining mines: {remaining} | Flagged: {flagged_count} | Revealed: {revealed_count}/{self._rows * self._cols - self._mines}"

    def _place_mines(self, exclude_row: int, exclude_col: int) -> None:
        """Place mines randomly, avoiding the first click area."""
        placed = 0
        while placed < self._mines:
            row = self.np_random.integers(0, self._rows)
            col = self.np_random.integers(0, self._cols)
            # Avoid first click and its neighbors
            if (
                row >= exclude_row - 1
                and row <= exclude_row + 1
                and col >= exclude_col - 1
                and col <= exclude_col + 1
            ):
                continue
            if self._board[row][col] == "M":
                continue
            self._board[row][col] = "M"
            placed += 1

    def _calculate_adjacent_mines(self) -> None:
        """Calculate the number of adjacent mines for each cell."""
        for r in range(self._rows):
            for c in range(self._cols):
                if self._board[r][c] == "M":
                    continue
                count = 0
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        nr, nc = r + dr, c + dc
                        if (
                            0 <= nr < self._rows
                            and 0 <= nc < self._cols
                            and self._board[nr][nc] == "M"
                        ):
                            count += 1
                self._board[r][c] = count

    def _reveal(self, row: int, col: int) -> bool:
        """Reveal a cell. Returns False if hit a mine, True otherwise."""
        # Place mines on first click
        if self._first_click:
            self._place_mines(row, col)
            self._calculate_adjacent_mines()
            self._first_click = False

        # Check if hit a mine
        if self._board[row][col] == "M":
            self._revealed[row][col] = True
            return False

        # Remove flag if flagged
        if self._flagged[row][col]:
            self._flagged[row][col] = False

        # Reveal recursively
        self._reveal_recursive(row, col)
        return True

    def _reveal_recursive(self, row: int, col: int) -> None:
        """Recursively reveal cells."""
        if self._revealed[row][col] or self._board[row][col] == "M":
            return
        self._revealed[row][col] = True

        # If cell is 0, reveal all neighbors
        if self._board[row][col] == 0:
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < self._rows and 0 <= nc < self._cols:
                        self._reveal_recursive(nr, nc)

    def _toggle_flag(self, row: int, col: int) -> None:
        """Toggle flag on a cell."""
        if not self._revealed[row][col]:
            self._flagged[row][col] = not self._flagged[row][col]

    def _is_won(self) -> bool:
        """Check if the game is won (all non-mine cells revealed)."""
        for r in range(self._rows):
            for c in range(self._cols):
                if self._board[r][c] != "M" and not self._revealed[r][c]:
                    return False
        return True

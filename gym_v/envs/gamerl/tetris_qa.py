"""Tetris Q&A environment based on GameRL.

Single-turn Q&A environment where the model answers questions about a Tetris game state.
"""

from __future__ import annotations

from importlib import resources
import random
import re
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation, get_logger

logger = get_logger()


# Question types from original Game-RL
# Removed module-level QUESTION_TYPES - now defined as class variable
GAME_RULES = """Rules:
1. The image shows a standard Tetris grid with {rows} rows and {cols} columns.
2. The top row of the grid is labeled as Row 0 according to the coordinates.
3. Row coordinates increment from top to bottom in the grid from 0 to {max_row}.
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
4. The game ends when pieces stack up to the top of the grid."""

ANSWER_FORMAT_PROMPT = """
**Answer Format:**
- For numbers: Reply with only the number (e.g., 5 or -1)
- For shape identification: Reply with only the number (1-8)

Do not include any explanation or extra text.
"""


class GameRLTetrisQAEnv(Env):
    """Tetris Q&A environment.

    Single-turn Q&A environment based on the original Game-RL Tetris game.
    Given a game state image, answer questions about empty cells, tetromino shapes,
    collision timing, or row clearing optimization.

    Args:
        question_type: Question type ID (0-3). None for random selection.
        rows: Number of rows in the grid (default 12)
        cols: Number of columns in the grid (default 8)
        cell_size: Size of each cell in pixels for rendering (default 30)
    """

    # Question types
    QUESTION_TYPES = [
        {
            "id": "type_0",
            "name": "empty_cells_in_row",
            "level": "Easy",
            "answer_format": "fill_in_blank",
            "qa_type": "State Prediction",
        },
        {
            "id": "type_1",
            "name": "tetromino_shape",
            "level": "Easy",
            "answer_format": "multiple_choice",
            "qa_type": "State Prediction",
        },
        {
            "id": "type_2",
            "name": "timesteps_until_collision",
            "level": "Medium",
            "answer_format": "fill_in_blank",
            "qa_type": "State Prediction",
        },
        {
            "id": "type_3",
            "name": "max_rows_cleared",
            "level": "Hard",
            "answer_format": "fill_in_blank",
            "qa_type": "State Prediction",
        },
    ]

    assets_dir = resources.files("gym_v.envs") / "assets"

    # Tetromino shapes
    TETROMINOES = {
        "I": np.array([[1, 1, 1, 1]]),
        "O": np.array([[1, 1], [1, 1]]),
        "T": np.array([[0, 1, 0], [1, 1, 1]]),
        "L": np.array([[1, 0], [1, 0], [1, 1]]),
        "J": np.array([[0, 1], [0, 1], [1, 1]]),
        "S": np.array([[0, 1, 1], [1, 1, 0]]),
        "Z": np.array([[1, 1, 0], [0, 1, 1]]),
    }

    # Colors
    COLORS = {
        "background": (255, 255, 255),
        "grid_line": (0, 0, 0),
        "placed": (128, 128, 128),
        "falling": (255, 0, 0),
        "text": (0, 0, 0),
    }

    def __init__(
        self,
        question_type: int | None = None,
        rows: int = 12,
        cols: int = 8,
        cell_size: int = 30,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._question_type = question_type
        self._rows = rows
        self._cols = cols
        self._cell_size = cell_size
        self._padding = 35

        # Game state
        self._grid: np.ndarray = np.zeros((rows, cols), dtype=int)
        self._current_piece: np.ndarray | None = None
        self._current_piece_type: str = ""
        self._current_pos: tuple[int, int] = (0, 0)

        # Q&A state
        self._current_question_type: int = 0
        self._current_question: str = ""
        self._oracle_answer: str = ""
        self._options: list[str] | None = None
        self._target_row: int = 0
        self._action_direction: str = ""

    @property
    def description(self) -> str:
        """Return game rules + current question + answer format."""
        rules = GAME_RULES.format(
            rows=self._rows, cols=self._cols, max_row=self._rows - 1
        )
        desc = rules + "\n\n**Question:** " + self._current_question

        if self._options:
            desc += "\n\n**Options:**\n" + "\n".join(self._options)

        desc += ANSWER_FORMAT_PROMPT
        return desc.strip()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Generate game state
        self._generate_game_state()

        # Select question type
        if self._question_type is not None:
            self._current_question_type = self._question_type
        else:
            self._current_question_type = self.np_random.integers(
                0, len(self.QUESTION_TYPES)
            )

        # Generate Q&A
        self._generate_qa()

        logger.info(f"Reset Tetris QA (type={self._current_question_type}).")

        obs = Observation(
            image=self.render(),
            text=self._current_question,
            metadata={
                "question_type": self.QUESTION_TYPES[self._current_question_type][
                    "name"
                ],
                "level": self.QUESTION_TYPES[self._current_question_type]["level"],
            },
        )
        info = {
            "oracle_answer": self._oracle_answer,
            "question_type": self._current_question_type,
        }
        return obs, info

    def inner_step(
        self, action: str
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        """Evaluate the answer. Always terminates after one step."""
        answer = action.strip()
        reward = self._score_answer(answer)

        obs = Observation(
            image=self.render(),
            text=None,
            metadata={
                "question_type": self.QUESTION_TYPES[self._current_question_type][
                    "name"
                ],
            },
        )
        info = {
            "oracle_answer": self._oracle_answer,
            "user_answer": answer,
            "correct": reward == 1.0,
        }

        return obs, reward, True, False, info

    def _score_answer(self, answer: str) -> float:
        """Score the answer."""
        answer_format = self.QUESTION_TYPES[self._current_question_type][
            "answer_format"
        ]

        if answer_format == "number":
            match = re.search(r"-?\d+", answer)
            if match:
                try:
                    return (
                        1.0 if int(match.group()) == int(self._oracle_answer) else 0.0
                    )
                except ValueError:
                    return 0.0
            return 0.0
        elif answer_format == "choice":
            match = re.search(r"[1-8]", answer)
            if match:
                return 1.0 if match.group() == self._oracle_answer else 0.0
            return 0.0
        return 0.0

    def _generate_game_state(self) -> None:
        """Generate a random game state with placed blocks and a falling piece."""
        # Initialize empty grid
        self._grid = np.zeros((self._rows, self._cols), dtype=int)

        # Add some placed blocks (simulate a mid-game state)
        num_placed = self.np_random.integers(5, 20)
        for _ in range(num_placed):
            row = self.np_random.integers(self._rows // 2, self._rows)
            col = self.np_random.integers(0, self._cols)
            self._grid[row, col] = 1

        # Spawn a falling piece
        piece_type = list(self.TETROMINOES.keys())[self.np_random.integers(0, 7)]
        self._current_piece_type = piece_type
        self._current_piece = self.TETROMINOES[piece_type].copy()

        # Random rotation
        num_rotations = self.np_random.integers(0, 4)
        for _ in range(num_rotations):
            self._current_piece = np.rot90(self._current_piece, -1)

        # Position at top
        start_col = (self._cols - self._current_piece.shape[1]) // 2
        start_row = self.np_random.integers(0, 3)
        self._current_pos = (start_row, start_col)

    def _generate_qa(self) -> None:
        """Generate question and oracle answer."""
        q_type = self._current_question_type
        self._options = None

        if q_type == 0:
            self._generate_q0_empty_cells()
        elif q_type == 1:
            self._generate_q1_tetromino_shape()
        elif q_type == 2:
            self._generate_q2_timesteps_collision()
        elif q_type == 3:
            self._generate_q3_max_rows_cleared()

    def _generate_q0_empty_cells(self) -> None:
        """Q0: Count empty cells in a specific row."""
        self._target_row = self.np_random.integers(0, self._rows)
        self._current_question = (
            f"How many empty cells are there in Row {self._target_row} of the grid?"
        )

        # Count empty cells (not placed blocks, not falling piece)
        empty_count = 0
        for col in range(self._cols):
            if self._grid[self._target_row, col] == 0:
                # Check if falling piece occupies this cell
                if self._current_piece is not None:
                    piece_row, piece_col = self._current_pos
                    is_piece = False
                    for i in range(self._current_piece.shape[0]):
                        for j in range(self._current_piece.shape[1]):
                            if self._current_piece[i, j] == 1:
                                if (
                                    piece_row + i == self._target_row
                                    and piece_col + j == col
                                ):
                                    is_piece = True
                                    break
                        if is_piece:
                            break
                    if not is_piece:
                        empty_count += 1
                else:
                    empty_count += 1

        self._oracle_answer = str(empty_count)
        self._options = None

    def _generate_q1_tetromino_shape(self) -> None:
        """Q1: Identify the falling tetromino shape."""
        self._current_question = (
            "What is the shape of the active Tetrimino at the top of the screen?"
        )

        self._options = [
            "1: I",
            "2: O",
            "3: T",
            "4: L",
            "5: J",
            "6: S",
            "7: Z",
            "8: no falling Tetrimino",
        ]

        shape_to_num = {
            "I": "1",
            "O": "2",
            "T": "3",
            "L": "4",
            "J": "5",
            "S": "6",
            "Z": "7",
        }

        if self._current_piece is None:
            self._oracle_answer = "8"
        else:
            self._oracle_answer = shape_to_num.get(self._current_piece_type, "8")

    def _generate_q2_timesteps_collision(self) -> None:
        """Q2: Count timesteps until collision after moving left/right."""
        direction = "left" if self.np_random.random() < 0.5 else "right"
        self._action_direction = direction
        self._current_question = f"If the current active Tetrimino is only moved one step (After moving {direction} for one column), how many timesteps will it take to collide with another block or the grid boundary?"

        if self._current_piece is None:
            self._oracle_answer = "-1"
            return

        # Get active cells
        active_cells = []
        piece_row, piece_col = self._current_pos
        for i in range(self._current_piece.shape[0]):
            for j in range(self._current_piece.shape[1]):
                if self._current_piece[i, j] == 1:
                    active_cells.append((piece_row + i, piece_col + j))

        # Apply direction
        dc = -1 if direction == "left" else 1
        new_cells = [(r, c + dc) for r, c in active_cells]

        # Check if move is valid
        for r, c in new_cells:
            if c < 0 or c >= self._cols:
                self._oracle_answer = "0"
                return
            if r >= 0 and r < self._rows and self._grid[r, c] == 1:
                self._oracle_answer = "0"
                return

        # Count falling steps
        steps = 0
        while True:
            falling_cells = [(r + steps, c) for r, c in new_cells]

            # Check for collision
            can_fall = True
            for r, c in falling_cells:
                if r >= self._rows - 1:
                    can_fall = False
                    break
                if r + 1 >= 0 and r + 1 < self._rows and self._grid[r + 1, c] == 1:
                    can_fall = False
                    break

            if not can_fall:
                break
            steps += 1

            if steps > 100:
                break

        self._oracle_answer = str(steps)

    def _generate_q3_max_rows_cleared(self) -> None:
        """Q3: Maximum rows that can be cleared with optimal placement."""
        self._current_question = "What's the maximum number of rows can be cleared after positioning the active Tetrimino in this turn?"

        if self._current_piece is None:
            self._oracle_answer = "0"
            return

        # Get normalized cells
        active_cells = []
        for i in range(self._current_piece.shape[0]):
            for j in range(self._current_piece.shape[1]):
                if self._current_piece[i, j] == 1:
                    active_cells.append((i, j))

        # Try all rotations and positions
        max_cleared = 0

        for rotation in range(4):
            rotated = np.rot90(self._current_piece, -rotation)
            shape = rotated.shape

            # Get cells for this rotation
            cells = []
            for i in range(shape[0]):
                for j in range(shape[1]):
                    if rotated[i, j] == 1:
                        cells.append((i, j))

            # Try all columns
            for col_offset in range(self._cols - shape[1] + 1):
                # Find lowest valid row
                for row_offset in range(self._rows - shape[0], -1, -1):
                    # Check if position is valid
                    valid = True
                    position_cells = []

                    for cr, cc in cells:
                        nr, nc = row_offset + cr, col_offset + cc
                        if nr < 0 or nr >= self._rows or nc < 0 or nc >= self._cols:
                            valid = False
                            break
                        if self._grid[nr, nc] == 1:
                            valid = False
                            break
                        position_cells.append((nr, nc))

                    if not valid:
                        continue

                    # Check if this is the lowest position
                    can_go_lower = True
                    for cr, cc in cells:
                        nr, nc = row_offset + cr + 1, col_offset + cc
                        if nr >= self._rows or self._grid[nr, nc] == 1:
                            can_go_lower = False
                            break

                    if not can_go_lower:
                        # Count cleared rows
                        grid_copy = self._grid.copy()
                        for pr, pc in position_cells:
                            grid_copy[pr, pc] = 1

                        cleared = 0
                        for row in range(self._rows):
                            if np.all(grid_copy[row] == 1):
                                cleared += 1

                        max_cleared = max(max_cleared, cleared)

        self._oracle_answer = str(max_cleared)

    def render(self) -> Image.Image:
        """Render the game state."""
        width = self._cols * self._cell_size + 2 * self._padding
        height = self._rows * self._cell_size + 2 * self._padding

        img = Image.new("RGB", (width, height), self.COLORS["background"])
        draw = ImageDraw.Draw(img)

        font_path = self.assets_dir / "DejaVuSans.ttf"
        if font_path.exists():
            font = ImageFont.truetype(str(font_path), 16)
        else:
            font = ImageFont.load_default()

        # Coordinate labels
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

        # Grid cells
        for row in range(self._rows):
            for col in range(self._cols):
                x1 = col * self._cell_size + self._padding
                y1 = row * self._cell_size + self._padding
                x2 = x1 + self._cell_size
                y2 = y1 + self._cell_size

                if self._grid[row, col] == 1:
                    draw.rectangle([x1, y1, x2, y2], fill=self.COLORS["placed"])

                draw.rectangle([x1, y1, x2, y2], outline=self.COLORS["grid_line"])

        # Falling piece
        if self._current_piece is not None:
            piece_row, piece_col = self._current_pos
            for i in range(self._current_piece.shape[0]):
                for j in range(self._current_piece.shape[1]):
                    if self._current_piece[i, j] == 1:
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

        return img

"""Minesweeper Q&A environment based on GameRL.

Single-turn Q&A environment where the model answers questions about Minesweeper game states.
"""

from __future__ import annotations

from importlib import resources
import random
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation, get_logger

logger = get_logger()


# Question types from original Game-RL
# Removed module-level QUESTION_TYPES - now defined as class variable
GAME_RULES_TEMPLATE = """This is a Minesweeper game. The size of the chessboard is {rows}x{cols}, and there are a total of {mines} mines hidden on the board.

The numbers on the board indicate how many mines are adjacent to that cell, including diagonals. Cells marked with "F" (flagged) are identified as potential locations of mines based on logical deduction or prior knowledge. These flagged cells play a critical role in guiding your reasoning for answering the questions. Cells with no numbers and no flags are safe and contain no adjacent mines.

The board uses a coordinate system where the top-left cell corresponds to (0,0), and the rows and columns are numbered starting from 0.

Please use the provided board configuration and logical reasoning to deduce the correct answers to the following questions:"""

ANSWER_FORMAT_PROMPT = """
**Answer Format:**
- For numbers: Reply with only the number, e.g., 7
- For multiple choice: Reply with only the letter (A, B, C, D, E, or F)

Do not include any explanation or extra text."""


class GameRLMinesweeperQAEnv(Env):
    """Minesweeper Q&A environment.

    Single-turn Q&A environment based on the original Game-RL Minesweeper game.
    Given a game state image, answer questions about mine counts, cell states,
    or optimal moves.

    Args:
        question_type: Question type ID (0-5). None for random selection.
        difficulty: Game difficulty ('Easy', 'Medium', 'Hard'). None for random.
        cell_size: Size of each cell in pixels for rendering (default 60)
    """

    # Question types
    QUESTION_TYPES = [
        {
            "id": "type_0",
            "name": "flagged_count",
            "level": "Easy",
            "answer_format": "fill_in_blank",
            "qa_type": "StateInfo",
        },
        {
            "id": "type_1",
            "name": "remaining_mines",
            "level": "Easy",
            "answer_format": "fill_in_blank",
            "qa_type": "StateInfo",
        },
        {
            "id": "type_2",
            "name": "revealed_count",
            "level": "Easy",
            "answer_format": "fill_in_blank",
            "qa_type": "StateInfo",
        },
        {
            "id": "type_3",
            "name": "cell_state",
            "level": "Easy",
            "answer_format": "multiple_choice",
            "qa_type": "StateInfo",
        },
        {
            "id": "type_4",
            "name": "reveal_outcome",
            "level": "Hard",
            "answer_format": "multiple_choice",
            "qa_type": "StrategyOptimization",
        },
        {
            "id": "type_5",
            "name": "best_move",
            "level": "Hard",
            "answer_format": "multiple_choice",
            "qa_type": "StrategyOptimization",
        },
    ]

    assets_dir = resources.files("gym_v.envs") / "assets"

    # Difficulty configurations
    DIFFICULTIES = {
        "Easy": {"rows": 4, "cols": 4, "mines": 3},
        "Medium": {"rows": 5, "cols": 5, "mines": 5},
        "Hard": {"rows": 6, "cols": 6, "mines": 8},
    }

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
        question_type: int | None = None,
        difficulty: str | None = None,
        cell_size: int = 60,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._question_type = question_type
        self._difficulty_override = difficulty
        self._cell_size = cell_size
        self._margin = 40

        # Game state (initialized in reset)
        self._rows = 4
        self._cols = 4
        self._mines = 3
        self._difficulty = "Easy"
        self._board: list[list[int | str]] = []
        self._revealed: list[list[bool]] = []
        self._flagged: list[list[bool]] = []

        # Q&A state (initialized in reset)
        self._current_question_type: int = 0
        self._current_question: str = ""
        self._oracle_answer: str = ""
        self._answer_format: str = ""
        self._options: list[str] = []

    @property
    def description(self) -> str:
        """Return game rules + current question + answer format."""
        game_rules = GAME_RULES_TEMPLATE.format(
            rows=self._rows, cols=self._cols, mines=self._mines
        )
        desc = game_rules + "\n\n**Question:** " + self._current_question

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

        # Select difficulty
        if self._difficulty_override:
            self._difficulty = self._difficulty_override
        else:
            self._difficulty = random.choice(list(self.DIFFICULTIES.keys()))

        config = self.DIFFICULTIES[self._difficulty]
        self._rows = config["rows"]
        self._cols = config["cols"]
        self._mines = config["mines"]

        # Generate game state
        self._generate_game_state()

        # Select question type
        if self._question_type is not None:
            self._current_question_type = self._question_type
        else:
            self._current_question_type = self.np_random.integers(
                0, len(self.QUESTION_TYPES)
            )

        # Generate question and answer
        self._generate_qa()

        logger.info(
            f"Reset Minesweeper QA (type={self._current_question_type}, "
            f"difficulty={self._difficulty}, grid={self._rows}x{self._cols})."
        )

        obs = Observation(
            image=self.render(),
            text=self._current_question,
            metadata={
                "question_type": self.QUESTION_TYPES[self._current_question_type][
                    "name"
                ],
                "level": self.QUESTION_TYPES[self._current_question_type]["level"],
                "difficulty": self._difficulty,
            },
        )
        info = {"oracle_answer": self._oracle_answer}
        return obs, info

    def inner_step(
        self, action: str
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        """Validate the answer and return reward.

        The episode always terminates after one step (single-turn Q&A).
        """
        user_answer = action.strip().upper()

        # Normalize answer
        correct = self._check_answer(user_answer)

        reward = 1.0 if correct else 0.0

        obs = Observation(
            image=self.render(),
            text=None,
            metadata={"correct": correct, "user_answer": user_answer},
        )
        info = {
            "oracle_answer": self._oracle_answer,
            "user_answer": user_answer,
            "correct": correct,
        }
        return obs, reward, True, False, info

    def render(self) -> Image.Image:
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

                    if self._board[r][c] > 0:
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

    def _generate_game_state(self) -> None:
        """Generate a random minesweeper game state."""
        # Initialize board
        self._board = [[0 for _ in range(self._cols)] for _ in range(self._rows)]
        self._revealed = [[False for _ in range(self._cols)] for _ in range(self._rows)]
        self._flagged = [[False for _ in range(self._cols)] for _ in range(self._rows)]

        # Place mines
        placed = 0
        while placed < self._mines:
            row = self.np_random.integers(0, self._rows)
            col = self.np_random.integers(0, self._cols)
            if self._board[row][col] == "M":
                continue
            self._board[row][col] = "M"
            placed += 1

        # Calculate adjacent mines
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

        # Simulate a first click (avoid mines)
        safe_cells = [
            (r, c)
            for r in range(self._rows)
            for c in range(self._cols)
            if self._board[r][c] != "M"
        ]
        if safe_cells:
            first_row, first_col = random.choice(safe_cells)
            self._reveal_recursive(first_row, first_col)

        # Auto-flag 2-4 mines
        max_flags = random.randint(2, min(4, self._mines))
        boundary_mines = self._get_boundary_mines()
        flags_to_place = min(max_flags, len(boundary_mines))
        if boundary_mines:
            for r, c in random.sample(boundary_mines, flags_to_place):
                self._flagged[r][c] = True

    def _reveal_recursive(self, row: int, col: int) -> None:
        """Recursively reveal cells."""
        if self._revealed[row][col] or self._board[row][col] == "M":
            return
        self._revealed[row][col] = True

        if self._board[row][col] == 0:
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < self._rows and 0 <= nc < self._cols:
                        self._reveal_recursive(nr, nc)

    def _get_boundary_mines(self) -> list[tuple[int, int]]:
        """Get mines that are adjacent to revealed cells."""
        boundary = []
        for r in range(self._rows):
            for c in range(self._cols):
                if self._board[r][c] != "M" or self._revealed[r][c]:
                    continue
                # Check if adjacent to revealed cell
                has_revealed_neighbor = False
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        nr, nc = r + dr, c + dc
                        if (
                            0 <= nr < self._rows
                            and 0 <= nc < self._cols
                            and self._revealed[nr][nc]
                        ):
                            has_revealed_neighbor = True
                            break
                    if has_revealed_neighbor:
                        break
                if has_revealed_neighbor:
                    boundary.append((r, c))
        return boundary

    def _generate_qa(self) -> None:
        """Generate question and answer based on question type."""
        qtype = self._current_question_type

        if qtype == 0:
            self._generate_flagged_count_qa()
        elif qtype == 1:
            self._generate_remaining_mines_qa()
        elif qtype == 2:
            self._generate_revealed_count_qa()
        elif qtype == 3:
            self._generate_cell_state_qa()
        elif qtype == 4:
            self._generate_reveal_outcome_qa()
        elif qtype == 5:
            self._generate_best_move_qa()

    def _generate_flagged_count_qa(self) -> None:
        """Q: How many mines are currently flagged?"""
        answer = sum(sum(row) for row in self._flagged)
        self._current_question = "How many mines are currently flagged?"
        self._oracle_answer = str(answer)
        self._options = []

    def _generate_remaining_mines_qa(self) -> None:
        """Q: How many mines are left to be found?"""
        flagged_count = sum(sum(row) for row in self._flagged)
        answer = self._mines - flagged_count
        self._current_question = "How many mines are left to be found?"
        self._oracle_answer = str(answer)
        self._options = []

    def _generate_revealed_count_qa(self) -> None:
        """Q: How many cells have been revealed?"""
        answer = sum(sum(row) for row in self._revealed)
        self._current_question = "How many cells have been revealed?"
        self._oracle_answer = str(answer)
        self._options = []

    def _generate_cell_state_qa(self) -> None:
        """Q: What is the state of the cell at (row, col)?"""
        row = self.np_random.integers(0, self._rows)
        col = self.np_random.integers(0, self._cols)

        options = [
            "A. It is revealed and shows a number.",
            "B. It is flagged as mine.",
            "C. It is still hidden.",
            "D. It is revealed and shows no more information.",
        ]

        if self._revealed[row][col]:
            if self._board[row][col] == 0:
                answer = "D"
            else:
                answer = "A"
        elif self._flagged[row][col]:
            answer = "B"
        else:
            answer = "C"

        self._current_question = f"What is the state of the cell at ({row},{col})?"
        self._options = options
        self._oracle_answer = answer

    def _generate_reveal_outcome_qa(self) -> None:
        """Q: What will happen if the player reveals the cell at (row, col)?"""
        # Select boundary cell
        boundary_cells = []
        for r in range(self._rows):
            for c in range(self._cols):
                if self._revealed[r][c]:
                    continue
                has_revealed_neighbor = False
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        nr, nc = r + dr, c + dc
                        if (
                            0 <= nr < self._rows
                            and 0 <= nc < self._cols
                            and self._revealed[nr][nc]
                        ):
                            has_revealed_neighbor = True
                            break
                    if has_revealed_neighbor:
                        break
                if has_revealed_neighbor:
                    boundary_cells.append((r, c))

        if boundary_cells:
            row, col = random.choice(boundary_cells)
        else:
            row = self.np_random.integers(0, self._rows)
            col = self.np_random.integers(0, self._cols)

        # Generate options based on cell value
        if self._board[row][col] == "M":
            value = self.np_random.integers(1, 9)
            value2 = self.np_random.integers(1, 9)
            while value2 == value:
                value2 = self.np_random.integers(1, 9)
            answer = "A"
        else:
            actual_value = self._board[row][col]
            if actual_value == 0:
                value = self.np_random.integers(1, 9)
                value2 = self.np_random.integers(1, 9)
                while value2 == value:
                    value2 = self.np_random.integers(1, 9)
                answer = "B"
            else:
                if random.choice([True, False]):
                    value = actual_value
                    value2 = self.np_random.integers(0, 9)
                    while value2 == value:
                        value2 = self.np_random.integers(0, 9)
                    answer = "C"
                else:
                    value2 = actual_value
                    value = self.np_random.integers(0, 9)
                    while value == value2:
                        value = self.np_random.integers(0, 9)
                    answer = "D"

        options = [
            "A: The game will end because the cell contains a mine.",
            "B: The cell will reveal an empty area, and adjacent cells will also be revealed.",
            f"C: The cell will reveal the number {value}.",
            f"D: The cell will reveal the number {value2}.",
        ]

        self._current_question = (
            f"What will happen if the player reveals the cell at ({row},{col})?"
        )
        self._options = options
        self._oracle_answer = answer

    def _generate_best_move_qa(self) -> None:
        """Q: What is the best next move at (row, col)?"""
        row = self.np_random.integers(0, self._rows)
        col = self.np_random.integers(0, self._cols)

        options = [
            "A. Flag this cell as a mine.",
            "B. Reveal this cell.",
            "C. Analyze adjacent cells for potential mines according to the number on it.",
            "D. Skip this move and wait for more information.",
            "E. This cell has already been revealed, and no further action is required.",
            "F. This cell has already been flagged as a mine, and no further action is needed.",
        ]

        # Check if on boundary
        is_boundary = False
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                nr, nc = row + dr, col + dc
                if (
                    0 <= nr < self._rows
                    and 0 <= nc < self._cols
                    and self._revealed[nr][nc]
                ):
                    is_boundary = True
                    break
            if is_boundary:
                break

        # Determine answer
        if self._board[row][col] == "M" and not self._flagged[row][col]:
            answer = "A"
        elif not self._revealed[row][col] and not self._flagged[row][col]:
            if is_boundary:
                answer = "B"
            else:
                answer = "D"
        elif self._revealed[row][col] and self._board[row][col] != 0:
            answer = "C"
        elif self._revealed[row][col]:
            answer = "E"
        elif self._flagged[row][col]:
            answer = "F"
        else:
            answer = "D"

        self._current_question = f"What is the best next move at ({row},{col})?"
        self._options = options
        self._oracle_answer = answer

    def _check_answer(self, user_answer: str) -> bool:
        """Check if user's answer is correct."""
        user_answer = user_answer.strip().upper()
        oracle_answer = self._oracle_answer.strip().upper()
        return user_answer == oracle_answer

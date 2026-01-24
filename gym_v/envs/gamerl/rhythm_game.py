"""Rhythm Game QA environment based on GameRL."""

from __future__ import annotations

from importlib import resources
import random
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation, get_logger

logger = get_logger()


class GameRLRhythmGameQAEnv(Env):
    """Rhythm Game QA environment.

    A rhythm game where blocks fall from top to bottom at 1 cell/second.
    Players select a column and click blocks that reach the first row to score points.

    Block types:
    - Click (yellow): 10 points
    - Reverse (green): 15 points, reverses the game left-right
    - Snake (pink head, blue body, grey tail): score = length * (2*length + 7)

    Args:
        grid_size: Grid dimensions (rows, cols), default based on difficulty
        difficulty: Puzzle difficulty ('Easy', 'Medium', 'Hard')
        cell_size: Size of each cell in pixels (default 40)
        question_type: Type of question to ask
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    QUESTION_TYPES = [
        {
            "id": "block_type",
            "name": "Block Type at Position",
            "level": "Easy",
            "answer_format": "multiple_choice",
            "qa_type": "Target Perception",
        },
        {
            "id": "snake_length",
            "name": "Snake Length After Time",
            "level": "Medium",
            "answer_format": "multiple_choice",
            "qa_type": "State Prediction",
        },
        {
            "id": "column_score",
            "name": "Score for Column",
            "level": "Medium",
            "answer_format": "fill_in_blank",
            "qa_type": "State Prediction",
        },
    ]

    GRID_SIZES = {
        "Easy": (15, 4),
        "Medium": (15, 6),
        "Hard": (20, 6),
    }

    # Colors
    YELLOW = (255, 255, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    PINK = (255, 105, 180)
    GREY = (128, 128, 128)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)

    BLOCK_TYPES = {
        "yellow": ("Click", 10),
        "green": ("Reverse", 15),
        "blue": ("Snake Body", 0),
        "pink": ("Snake Head", 0),
        "grey": ("Snake Tail", 0),
    }

    GAME_RULES = dedent("""
        Rhythm Game Rules:
        - Blocks fall at 1 cell/second from top to bottom
        - Select a column and click blocks that reach row 1 to score
        - Click blocks (yellow): 10 points each
        - Reverse blocks (green): 15 points, then grid reverses left-right
        - Snake blocks: pink head + blue body + grey tail
          Score = length * (2*length + 7)
          Must click ALL cells of the snake to score
        - Coordinates: (row, col) with row 1 at bottom, col 1 at left
    """).strip()

    def __init__(
        self,
        grid_size: tuple[int, int] | None = None,
        difficulty: str | None = None,
        cell_size: int = 40,
        question_type: int | None = None,
        num_players: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if difficulty is None:
            difficulty = random.choice(["Easy", "Medium", "Hard"])
        self._difficulty = difficulty
        self._grid_size = (
            grid_size if grid_size is not None else self.GRID_SIZES[difficulty]
        )
        self._cell_size = cell_size
        self._question_type_param = question_type
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        # Game state
        self._grid: list[list[str]] = []
        self._blocks: list[dict[str, Any]] = []

        # Standard QA variables
        self._question_type_idx: int = 0
        self._question: str = ""
        self._options: list[str] | None = None
        self._oracle_answer: str = ""

    ANSWER_FORMAT_PROMPT = dedent("""
        **Answer Format:**
        Reply with only the answer (number or option number).
        For multiple choice: 1, 2, 3, etc. For numbers: 42, 100, etc.
    """).strip()

    @property
    def description(self) -> str:
        """Return game rules + current question + answer format."""
        desc = self.GAME_RULES + "\n\n**Question:** " + self._question
        if self._options:
            desc += "\n\n**Options:**\n"
            for i, opt in enumerate(self._options):
                desc += f"{i+1}. {opt}\n"
        desc += "\n\n" + self.ANSWER_FORMAT_PROMPT
        return desc.strip()

    def _get_state_text(self) -> str:
        """Generate text description of current rhythm game state."""
        text = "Rhythm Game State\n\n"
        text += f"Grid Size: {self._grid_size[0]} rows x {self._grid_size[1]} columns\n"
        text += f"Difficulty: {self._difficulty}\n"
        text += f"Total Blocks: {len(self._blocks)}\n\n"

        # Count block types
        block_counts = {}
        for block in self._blocks:
            block_type = block["type"]
            block_counts[block_type] = block_counts.get(block_type, 0) + 1

        text += "Block Distribution:\n"
        for color, (name, _points) in self.BLOCK_TYPES.items():
            count = block_counts.get(color, 0)
            if count > 0:
                text += f"  {name}: {count} blocks\n"

        return text.strip()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        super().reset(seed=seed)

        # Generate puzzle
        self._generate_puzzle()

        # Select question type
        if self._question_type_param is None:
            self._question_type_idx = random.randint(0, len(self.QUESTION_TYPES) - 1)
        else:
            self._question_type_idx = self._question_type_param

        # Validate question type index
        if not (0 <= self._question_type_idx < len(self.QUESTION_TYPES)):
            raise ValueError(f"Invalid question type index: {self._question_type_idx}")

        q_type = self.QUESTION_TYPES[self._question_type_idx]

        # Generate question
        if q_type["id"] == "block_type":
            result = self._generate_block_type_question()
        elif q_type["id"] == "snake_length":
            result = self._generate_snake_length_question()
        elif q_type["id"] == "column_score":
            result = self._generate_column_score_question()
        else:
            raise ValueError(f"Unknown question type: {q_type['id']}")

        # Extract to instance variables
        self._question = result["question"]
        self._options = result.get("options")
        self._oracle_answer = result["answer"]

        logger.info(
            f"Reset Rhythm Game QA ({self._grid_size[0]}x{self._grid_size[1]}, "
            f"question: {q_type['id']})."
        )

        # Generate text state
        text_state = self._get_state_text()

        obs = Observation(
            image=self.render(),
            text=text_state,
            metadata={
                "text_state": text_state,
                "question": self._question,
                "options": self._options,
                "question_type": q_type["name"],
                "level": q_type["level"],
            },
        )

        info = {
            "seed": seed,
            "oracle_answer": self._oracle_answer,
            "question_type": q_type["id"],
        }

        return {agent_id: obs for agent_id in self._agent_ids}, {
            agent_id: info for agent_id in self._agent_ids
        }

    def inner_step(
        self, action: dict[str, str]
    ) -> tuple[
        dict[str, Observation],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, Any],
    ]:
        agent_id = next(iter(self._agent_ids))
        action_str = action[agent_id]

        info: dict[str, Any] = {}
        reward = 0.0
        terminated = True
        truncated = False

        # Check answer
        correct = action_str.strip().lower() == self._oracle_answer.strip().lower()

        if correct:
            reward = 1.0
            response = "Correct!"
        else:
            reward = 0.0
            response = f"Incorrect. The correct answer is: {self._oracle_answer}"

        info = {
            "correct": correct,
            "user_answer": action_str.strip(),
            "oracle_answer": self._oracle_answer,
        }

        obs = Observation(image=self.render(), text=response)

        return (
            {agent_id: obs for agent_id in self._agent_ids},
            {agent_id: reward for agent_id in self._agent_ids},
            {
                **{agent_id: terminated for agent_id in self._agent_ids},
                "__all__": terminated,
            },
            {
                **{agent_id: truncated for agent_id in self._agent_ids},
                "__all__": truncated,
            },
            {agent_id: info for agent_id in self._agent_ids},
        )

    def render(self) -> Image.Image | list[Image.Image] | None:
        """Render the rhythm game grid (matching original pygame style)."""
        rows, cols = self._grid_size

        # Constants matching original pygame implementation
        MARGIN_X = 30
        MARGIN_Y_BOTTOM = 30
        MARGIN_Y_TOP = 10
        BLOCK_SCALE = 0.8

        # Calculate dimensions
        cell_width = 400 // cols
        cell_height = 500 // rows
        grid_width = cell_width * cols
        grid_height = cell_height * rows

        # Total image size
        img_width = grid_width + MARGIN_X
        img_height = grid_height + MARGIN_Y_BOTTOM + MARGIN_Y_TOP

        # Create image with light green background (matching original)
        img = Image.new("RGB", (img_width, img_height), (204, 242, 153))
        draw = ImageDraw.Draw(img)

        # Load font
        try:
            label_font = ImageFont.truetype(str(self.assets_dir / "DejaVuSans.ttf"), 18)
        except Exception:
            label_font = ImageFont.load_default()

        # Offset for grid
        grid_offset_y = MARGIN_Y_TOP

        # Fill grid background with white
        draw.rectangle(
            [MARGIN_X, grid_offset_y, img_width, grid_offset_y + grid_height],
            fill=self.WHITE,
        )

        # Draw horizontal grid lines
        for i in range(rows + 1):
            y = grid_offset_y + i * cell_height
            draw.line(
                [(MARGIN_X, y), (MARGIN_X + grid_width, y)],
                fill=self.BLACK,
                width=1,
            )

        # Draw vertical grid lines
        for j in range(cols + 1):
            x = MARGIN_X + j * cell_width
            draw.line(
                [(x, grid_offset_y), (x, grid_offset_y + grid_height)],
                fill=self.BLACK,
                width=1,
            )

        # Draw row labels (left side, from top to bottom: rows, rows-1, ..., 1)
        for i in range(rows):
            row_num = rows - i
            y = grid_offset_y + i * cell_height + cell_height // 2
            draw.text(
                (15, y),
                str(row_num),
                fill=self.BLACK,
                font=label_font,
                anchor="mm",
            )

        # Draw column labels (bottom, 1-based)
        for j in range(cols):
            col_num = j + 1
            x = MARGIN_X + j * cell_width + cell_width // 2
            y = grid_offset_y + grid_height + 15
            draw.text(
                (x, y),
                str(col_num),
                fill=self.BLACK,
                font=label_font,
                anchor="mm",
            )

        # Draw blocks
        block_width = int(cell_width * BLOCK_SCALE)
        block_height = int(cell_height * BLOCK_SCALE)

        for block in self._blocks:
            row, col = block["row"], block["col"]
            color = block["color"]

            # Convert to screen coordinates (row 1 is at bottom)
            screen_row = rows - row
            block_x = (
                MARGIN_X + (col - 1) * cell_width + (cell_width - block_width) // 2
            )
            block_y = (
                grid_offset_y
                + screen_row * cell_height
                + (cell_height - block_height) // 2
            )

            # Draw colored block
            draw.rectangle(
                [block_x, block_y, block_x + block_width, block_y + block_height],
                fill=color,
                outline=None,
            )

        return img

    def _generate_puzzle(self):
        """Generate random rhythm game blocks."""
        rows, cols = self._grid_size
        self._grid = [["" for _ in range(cols)] for _ in range(rows)]
        self._blocks = []

        total_blocks = (rows * cols) // 2
        click_count = int(total_blocks * 0.5)
        reverse_count = int(total_blocks * 0.3)
        snake_count = total_blocks - click_count - reverse_count

        occupied = set()

        # Add snake blocks first
        for _ in range(snake_count):
            self._add_snake_block(occupied)

        # Add click blocks
        for _ in range(click_count):
            self._add_single_block(occupied, self.YELLOW, "yellow")

        # Add reverse blocks
        for _ in range(reverse_count):
            self._add_single_block(occupied, self.GREEN, "green")

    def _add_single_block(self, occupied: set, color: tuple, color_name: str):
        """Add a single block (click or reverse)."""
        rows, cols = self._grid_size
        for _ in range(100):  # Try 100 times
            col = random.randint(1, cols)
            row = random.randint(1, rows)
            if (row, col) not in occupied:
                occupied.add((row, col))
                self._blocks.append(
                    {"row": row, "col": col, "color": color, "type": color_name}
                )
                break

    def _add_snake_block(self, occupied: set):
        """Add a snake block."""
        rows, cols = self._grid_size
        length = random.randint(2, 5)

        for _ in range(100):  # Try 100 times
            col = random.randint(1, cols)
            start_row = random.randint(1, rows - length + 1)

            # Check if all positions are free
            positions = [(start_row + i, col) for i in range(length)]
            if all(pos not in occupied for pos in positions):
                # Add snake blocks
                for i, (row, _) in enumerate(positions):
                    if i == 0:
                        # Head (at bottom of snake)
                        self._blocks.append(
                            {"row": row, "col": col, "color": self.PINK, "type": "pink"}
                        )
                    elif i == length - 1:
                        # Tail (at top of snake)
                        self._blocks.append(
                            {"row": row, "col": col, "color": self.GREY, "type": "grey"}
                        )
                    else:
                        # Body
                        self._blocks.append(
                            {"row": row, "col": col, "color": self.BLUE, "type": "blue"}
                        )
                    occupied.add((row, col))
                break

    def _generate_block_type_question(self) -> dict[str, Any]:
        """Generate question about block type at a position."""
        rows, cols = self._grid_size
        row = random.randint(1, rows)
        col = random.randint(1, cols)

        # Find block at position
        block = next(
            (b for b in self._blocks if b["row"] == row and b["col"] == col), None
        )
        if block:
            block_type, _ = self.BLOCK_TYPES[block["type"]]
        else:
            block_type = "Non-type"

        options = [
            "Non-type",
            "Click",
            "Reverse",
            "Snake Head",
            "Snake Body",
            "Snake Tail",
        ]
        correct_idx = options.index(block_type) + 1

        question = f"Which type of block is at row {row}, column {col}?"

        return {"question": question, "answer": str(correct_idx), "options": options}

    def _generate_snake_length_question(self) -> dict[str, Any]:
        """Generate question about snake length after time."""
        # Find snake heads
        snake_heads = [b for b in self._blocks if b["type"] == "pink" and b["row"] > 1]

        if not snake_heads:
            # Fallback to block_type question
            return self._generate_block_type_question()

        head_block = random.choice(snake_heads)
        row_before, col = head_block["row"], head_block["col"]

        # Calculate snake length
        length = self._find_snake_length(row_before, col)

        # Random time
        time_k = random.randint(1, row_before - 1)
        row_after = row_before - time_k

        options = ["2", "3", "4", "5"]
        correct_idx = options.index(str(length)) + 1 if str(length) in options else 1

        question = f"Without selecting any column, what is the length of the snake block headed by ({row_after}, {col}) after {time_k} second(s)?"

        return {"question": question, "answer": str(correct_idx), "options": options}

    def _generate_column_score_question(self) -> dict[str, Any]:
        """Generate question about total score for a column."""
        cols = self._grid_size[1]
        col = random.randint(1, cols)

        # Calculate score for this column
        score = self._calculate_column_score(col)

        question = f"If you select column {col} to click, what is your total score? (Enter the total score as a number)"

        return {"question": question, "answer": str(score)}

    def _find_snake_length(self, head_row: int, col: int) -> int:
        """Find the length of a snake starting at head_row."""
        length = 1  # Head
        current_row = head_row + 1

        while True:
            block = next(
                (
                    b
                    for b in self._blocks
                    if b["row"] == current_row
                    and b["col"] == col
                    and b["type"] in ["blue", "grey"]
                ),
                None,
            )
            if block:
                length += 1
                if block["type"] == "grey":  # Tail
                    break
                current_row += 1
            else:
                break

        return length

    def _calculate_column_score(self, col: int) -> int:
        """Calculate total score for selecting a column."""
        score = 0
        row = 1

        while row <= self._grid_size[0]:
            block = next(
                (b for b in self._blocks if b["row"] == row and b["col"] == col), None
            )

            if block:
                block_type = block["type"]
                if block_type == "yellow":  # Click
                    score += 10
                    row += 1
                elif block_type == "green":  # Reverse (skip for simplicity)
                    score += 15
                    break  # Stop after reverse
                elif block_type == "pink":  # Snake head
                    length = self._find_snake_length(row, col)
                    score += length * (2 * length + 7)
                    row += length
                else:
                    row += 1
            else:
                row += 1

        return score

    def _check_answer(self, action: str) -> bool:
        """Check if answer is correct."""
        return action.strip().lower() == self._oracle_answer.strip().lower()

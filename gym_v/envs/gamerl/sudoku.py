"""Sudoku Q&A environment based on GameRL.

Single-turn Q&A environment where the model answers questions about Sudoku game states.
"""

from __future__ import annotations

from importlib import resources
import random
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation, get_logger
from gym_v.envs.gamerl.utils import build_description

logger = get_logger()


# Question types from original Game-RL
# Removed module-level QUESTION_TYPES - now defined as class variable
GAME_RULES_TEMPLATE = dedent("""
    This is a sudoku game in which the board is filled with a total number of colours equal to the length of the board's sides, and no rows, columns or squares are allowed to have duplicate colours. You should fill the empty cells on the board with following {size} colors: {colors}.

    In this Sudoku board, the row coordinates are 1-{size} from top to bottom, and the column coordinates are 1-{size} from left to right. The top-left cell is (1, 1).
""").strip()


class GameRLSudokuQAEnv(Env):
    """Sudoku Q&A environment.

    Single-turn Q&A environment based on the original Game-RL Sudoku game.
    Given a game state image, answer questions about colors, positions,
    or deductive reasoning.

    Args:
        question_type: Question type ID (0-4). None for random selection.
        size: Board size (4 or 9). None for random selection.
        cell_size: Size of each cell in pixels for rendering (default 50)
    """

    # Question types
    QUESTION_TYPES = [
        {
            "id": "color_position",
            "name": "Color Position",
            "level": "Easy",
            "answer_format": "multiple_choice",
            "qa_type": "Target Perception",
        },
        {
            "id": "color_count",
            "name": "Color Count",
            "level": "Easy",
            "answer_format": "fill_in_blank",
            "qa_type": "Target Perception",
        },
        {
            "id": "possible_colors",
            "name": "Possible Colors",
            "level": "Medium",
            "answer_format": "fill_in_blank",
            "qa_type": "State Prediction",
        },
        {
            "id": "empty_count",
            "name": "Empty Count",
            "level": "Medium",
            "answer_format": "fill_in_blank",
            "qa_type": "Target Perception",
        },
        {
            "id": "deductive_reasoning",
            "name": "Deductive Reasoning",
            "level": "Hard",
            "answer_format": "multiple_choice",
            "qa_type": "Deductive Reasoning",
        },
    ]

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

    def __init__(
        self,
        question_type: int | None = None,
        size: int | None = None,
        cell_size: int = 50,
        num_players: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._question_type_param = question_type
        self._size_override = size
        self._cell_size = cell_size
        self._margin = 30
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        # Game state (initialized in reset)
        self._size = 9
        self._box_size = 3
        self._board: list[list[int]] = []
        self._color_names: list[str] = []
        self._colors: dict[str, tuple[int, int, int]] = {}

        # Q&A state (initialized in reset)
        self._question_type_idx: int = 0
        self._question: str = ""
        self._oracle_answer: str = ""
        self._answer_format: str = ""
        self._options: list[str] = []

    @property
    def description(self) -> str:
        """Return game rules + current question + answer format."""
        colors = list(self._colors.keys()) if self._colors else []
        game_rules = GAME_RULES_TEMPLATE.format(
            size=self._size, colors=", ".join(colors)
        )
        return build_description(
            game_name="Sudoku",
            rules=game_rules,
            question=self._question,
            options=self._options,
            oracle_answer=self._oracle_answer,
        )

    def _get_state_text(self) -> str:
        """Generate text description of current Sudoku game state.

        Returns a grid representation matching the rendered image.
        """
        grid = []
        for row in range(self._size):
            row_chars = []
            for col in range(self._size):
                cell_value = self._board[row][col]
                if cell_value == 0:
                    row_chars.append(".")
                else:
                    row_chars.append(str(cell_value))
            grid.append("".join(row_chars))

        grid_str = "\n".join(grid)

        # Create color legend
        color_legend = ", ".join(
            [f"{i+1}={self._color_names[i]}" for i in range(self._size)]
        )

        return f"""Grid Size: {self._size}x{self._size}
Grid ({color_legend}, .=empty):
{grid_str}"""

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        super().reset(seed=seed)

        if seed is not None:
            random.seed(seed)

        # Select size
        if self._size_override:
            self._size = self._size_override
        else:
            self._size = random.choice([4, 9])

        self._box_size = 2 if self._size == 4 else 3

        # Set colors for this size
        if self._size == 4:
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

        # Generate game state
        self._generate_game_state()

        # Select question type
        if self._question_type_param is not None:
            self._question_type_idx = self._question_type_param
        else:
            self._question_type_idx = self.np_random.integers(
                0, len(self.QUESTION_TYPES)
            )

        # Generate question and answer
        self._generate_qa()

        logger.info(
            f"Reset Sudoku QA (type={self._question_type_idx}, size={self._size}x{self._size})."
        )

        text_state = self._get_state_text()

        obs = Observation(
            image=self.render(),
            text=text_state,
            metadata={
                "text_state": text_state,
                "text_prompt": f"{text_state}\n\n{self.description}",
                "question": self._question,
                "options": self._options,
                "question_type": self.QUESTION_TYPES[self._question_type_idx]["name"],
                "level": self.QUESTION_TYPES[self._question_type_idx]["level"],
                "size": self._size,
            },
        )
        info = {
            "seed": seed,
            "oracle_answer": self._oracle_answer,
            "question_type": self.QUESTION_TYPES[self._question_type_idx]["id"],
        }
        return {agent_id: obs for agent_id in self._agent_ids}, {
            agent_id: info for agent_id in self._agent_ids
        }

    def _score_answer(self, answer: str) -> float:
        """Score the user's answer.

        Args:
            answer: User's answer string

        Returns:
            1.0 if correct, 0.0 otherwise
        """
        return 1.0 if self._check_answer(answer) else 0.0

    def inner_step(
        self, action: dict[str, str]
    ) -> tuple[
        dict[str, Observation],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, Any],
    ]:
        """Validate the answer and return reward.

        The episode always terminates after one step (single-turn Q&A).
        """
        agent_id = next(iter(self._agent_ids))
        action_str = action[agent_id]

        user_answer = action_str.strip().upper()

        # Check answer
        reward = self._score_answer(user_answer)
        correct = reward == 1.0

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

        terminated = True
        truncated = False

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

    def _generate_game_state(self) -> None:
        """Generate a random Sudoku game state."""
        # Generate solved board
        solved_board = self._generate_solved_board()

        # Create puzzle by removing some cells (30-50% empty)
        empty_ratio = random.uniform(0.3, 0.5)
        total_cells = self._size * self._size
        target_empty = int(total_cells * empty_ratio)

        self._board = [row[:] for row in solved_board]
        cells = [(i, j) for i in range(self._size) for j in range(self._size)]
        random.shuffle(cells)

        removed = 0
        for row, col in cells:
            if removed >= target_empty:
                break
            self._board[row][col] = 0
            removed += 1

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

    def _get_valid_numbers(
        self, board: list[list[int]], row: int, col: int
    ) -> list[int]:
        """Get valid numbers for a cell."""
        valid = []
        for num in range(1, self._size + 1):
            if self._is_valid(board, row, col, num):
                valid.append(num)
        return valid

    def _generate_qa(self) -> None:
        """Generate question and answer based on question type."""
        qtype = self._question_type_idx

        if qtype == 0:
            self._generate_color_position_qa()
        elif qtype == 1:
            self._generate_color_count_qa()
        elif qtype == 2:
            self._generate_possible_colors_qa()
        elif qtype == 3:
            self._generate_empty_count_qa()
        elif qtype == 4:
            self._generate_deductive_reasoning_qa()

    def _generate_color_position_qa(self) -> None:
        """Q: What color is at position (row, col)?"""
        # Find a filled cell
        filled_cells = [
            (i, j)
            for i in range(self._size)
            for j in range(self._size)
            if self._board[i][j] != 0
        ]
        if not filled_cells:
            # Fallback to other question
            self._generate_color_count_qa()
            return

        row, col = random.choice(filled_cells)
        answer_idx = self._board[row][col] - 1
        answer_letter = chr(65 + answer_idx)  # A, B, C, ...

        options = [
            f"{chr(65 + i)}.{self._color_names[i]}"
            for i in range(len(self._color_names))
        ]

        self._question = f"What color is at position ({row + 1},{col + 1}) (note that on the board the position ({row + 1},{col + 1}) has already been filled with a certain color)? Choose from following options: {', '.join(options)}"
        self._options = options
        self._oracle_answer = answer_letter

    def _generate_color_count_qa(self) -> None:
        """Q: How many times does a specific color appear on the board?"""
        color_idx = self.np_random.integers(0, len(self._color_names))
        count = sum(
            1
            for i in range(self._size)
            for j in range(self._size)
            if self._board[i][j] == color_idx + 1
        )

        self._question = (
            f"How many times does {self._color_names[color_idx]} appear on the board?"
        )
        self._oracle_answer = str(count)
        self._options = []

    def _generate_possible_colors_qa(self) -> None:
        """Q: How many colors can be filled in position (row, col)?"""
        empty_cells = [
            (i, j)
            for i in range(self._size)
            for j in range(self._size)
            if self._board[i][j] == 0
        ]
        if not empty_cells:
            # Fallback
            self._generate_color_count_qa()
            return

        row, col = random.choice(empty_cells)
        valid_nums = self._get_valid_numbers(self._board, row, col)

        self._question = f"How many colors can be filled in position ({row + 1},{col + 1})? Inference based on the current situation focusing only on the colour of the position."
        self._oracle_answer = str(len(valid_nums))
        self._options = []

    def _generate_empty_count_qa(self) -> None:
        """Q: How many rows/cols have more than N empty cells?"""
        target_type = random.choice(["row", "col"])
        n = random.randint(1, 2)
        count = 0

        if target_type == "row":
            for i in range(self._size):
                empty = sum(1 for j in range(self._size) if self._board[i][j] == 0)
                if empty > n:
                    count += 1
        else:  # col
            for j in range(self._size):
                empty = sum(1 for i in range(self._size) if self._board[i][j] == 0)
                if empty > n:
                    count += 1

        self._question = f"How many {target_type}s have more than {n} empty cells?"
        self._oracle_answer = str(count)
        self._options = []

    def _generate_deductive_reasoning_qa(self) -> None:
        """Q: Deductive reasoning about cell colors."""
        # Find cells with only one valid option
        candidates = []
        for i in range(self._size):
            for j in range(self._size):
                if self._board[i][j] == 0:
                    valid = self._get_valid_numbers(self._board, i, j)
                    if len(valid) == 1:
                        candidates.append((i, j, valid[0]))

        if len(candidates) < 2:
            # Fallback
            self._generate_possible_colors_qa()
            return

        # Use first two candidates
        step1_row, step1_col, step1_num = candidates[0]
        step1_color = self._color_names[step1_num - 1]

        # Simulate filling first cell
        temp_board = [row[:] for row in self._board]
        temp_board[step1_row][step1_col] = step1_num

        # Find another unique cell after first step
        step2_candidates = []
        for i in range(self._size):
            for j in range(self._size):
                if temp_board[i][j] == 0:
                    valid = self._get_valid_numbers(temp_board, i, j)
                    if len(valid) == 1:
                        step2_candidates.append((i, j, valid[0]))

        if not step2_candidates:
            # Fallback
            self._generate_possible_colors_qa()
            return

        step2_row, step2_col, step2_num = step2_candidates[0]

        # Now find final answer
        temp_board[step2_row][step2_col] = step2_num
        final_candidates = []
        for i in range(self._size):
            for j in range(self._size):
                if temp_board[i][j] == 0:
                    valid = self._get_valid_numbers(temp_board, i, j)
                    if len(valid) == 1:
                        final_candidates.append((i, j, valid[0]))

        if not final_candidates:
            # Fallback
            self._generate_possible_colors_qa()
            return

        final_row, final_col, final_num = final_candidates[0]
        answer_letter = chr(65 + final_num - 1)

        options = [
            f"{chr(65 + i)}.{self._color_names[i]}"
            for i in range(len(self._color_names))
        ]

        self._question = f"Based on the current board state, if position ({step1_row + 1},{step1_col + 1}) must be filled with {step1_color}, what color should position ({final_row + 1},{final_col + 1}) be filled with? Choose from following options: {', '.join(options)}"
        self._options = options
        self._oracle_answer = answer_letter

    def _check_answer(self, user_answer: str) -> bool:
        """Check if user's answer is correct."""
        user_answer = user_answer.strip().upper()
        oracle_answer = self._oracle_answer.strip().upper()
        return user_answer == oracle_answer

"""Word Search QA environment based on Game-RL."""

from __future__ import annotations

from importlib import resources
import random
import string
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation, get_logger
from gym_v.utils.gamerl_utils import build_description

logger = get_logger()

# Question types based on Game-RL word_search main.py
# Removed module-level QUESTION_TYPES - now defined as class variable
GAME_RULES = dedent("""
    This is a Word Search puzzle game.

    Rules:
    1. The grid contains uppercase letters arranged in rows and columns
    2. Words can be placed in 8 directions: right, down, diagonal-right-down, diagonal-right-up, diagonal-left-down, diagonal-left-up, up, or left
    3. Row and column indexes begin from 1 at the top-left corner
    4. Words read from start to end in the specified direction
""").strip()


class WordSearchQAEnv(Env):
    # Meta: source=GameRL, category=puzzles, turn=single
    # Overrides: interaction_mode=single_turn, action_format=open_ended
    """Word Search QA environment (single-turn question answering).

    Generates questions about word search puzzles.

    Args:
        question_type: Specific question type ID (0-3), or None for random
        grid_size: Grid size (5-8, default None for random)
        cell_size: Cell size in pixels for rendering (default 50)
    """

    # Question types
    QUESTION_TYPES = [
        {
            "id": "cell_letter",
            "name": "Cell Letter",
            "level": "Easy",
            "answer_format": "multiple_choice",
            "qa_type": "Target Perception",
        },
        {
            "id": "letter_count",
            "name": "Letter Count",
            "level": "Medium",
            "answer_format": "multiple_choice",
            "qa_type": "Target Perception",
        },
        {
            "id": "word_direction",
            "name": "Word Direction",
            "level": "Medium",
            "answer_format": "multiple_choice",
            "qa_type": "Target Perception",
        },
        {
            "id": "find_word_location",
            "name": "Find Word Location",
            "level": "Hard",
            "answer_format": "multiple_choice",
            "qa_type": "Target Perception",
        },
    ]

    gamerl_assets_dir = resources.files("gym_v.envs") / "assets" / "gamerl"
    assets_dir = resources.files("gym_v.envs") / "assets"

    # 8 directions for word placement
    DIRECTIONS = {
        "right": (0, 1),
        "down": (1, 0),
        "diagonal-right-down": (1, 1),
        "diagonal-right-up": (-1, 1),
        "diagonal-left-down": (1, -1),
        "diagonal-left-up": (-1, -1),
        "up": (-1, 0),
        "left": (0, -1),
    }

    def __init__(
        self,
        question_type: int | None = None,
        grid_size: int | None = None,
        cell_size: int = 50,
        num_players: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._question_type_param = question_type
        self._grid_size_param = grid_size
        self._cell_size = cell_size
        self._margin = 0
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        # Load word list
        try:
            with open(self.gamerl_assets_dir / "words.txt") as f:
                self._words = [word.strip().upper() for word in f if word.strip()]
        except FileNotFoundError:
            # Fallback to a small word list
            self._words = [
                "PYTHON",
                "CODE",
                "DATA",
                "WORD",
                "SEARCH",
                "GRID",
                "LETTER",
                "PUZZLE",
                "GAME",
                "FIND",
                "DIRECTION",
                "RANDOM",
                "MATRIX",
            ]

        # Game state (initialized in reset)
        self._grid: list[list[str]] = []
        self._grid_size: int = 5

        # Question state
        self._question_type_idx: int = 0
        self._question: str = ""
        self._options: list[str] | None = None
        self._oracle_answer: str = ""
        self._current_q_type: dict[str, Any] = self.QUESTION_TYPES[0]

    @property
    def description(self) -> str:
        """Return game rules + current question + answer format."""
        return build_description(
            game_name="Word Search",
            rules=GAME_RULES,
            question=self._question,
            options=self._options,
            oracle_answer=self._oracle_answer,
        )

    def _get_state_text(self) -> str:
        """Generate text description of current Word Search grid.

        Returns a grid representation matching the rendered image.
        """
        grid_str = "\n".join(["".join(row) for row in self._grid])
        return f"""Grid Size: {self._grid_size}x{self._grid_size}
Grid (uppercase letters):
{grid_str}"""

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        super().reset(seed=seed)

        # Select question type
        if self._question_type_param is not None:
            self._question_type_idx = self._question_type_param
            q_type = self.QUESTION_TYPES[self._question_type_idx]
        else:
            self._question_type_idx = int(self.np_random.integers(0, len(self.QUESTION_TYPES)))
            q_type = self.QUESTION_TYPES[self._question_type_idx]

        self._current_q_type = q_type

        # Set grid size (5-8)
        if self._grid_size_param is not None:
            self._grid_size = self._grid_size_param
        else:
            self._grid_size = random.randint(5, 8)

        # Generate grid
        self._generate_grid()

        # Generate question
        if q_type["id"] == "cell_letter":
            self._generate_cell_letter_question()
        elif q_type["id"] == "letter_count":
            self._generate_letter_count_question()
        elif q_type["id"] == "word_direction":
            self._generate_word_direction_question()
        elif q_type["id"] == "find_word_location":
            self._generate_find_word_location_question()

        logger.info(
            f"Reset Word Search QA ({self._grid_size}x{self._grid_size}, question: {q_type['name']})."
        )

        text_state = self._get_state_text()
        obs = Observation(
            image=self.render(),
            text=self.description,
            metadata={
                "state_text": text_state,
                "text_prompt": f"{self.description}",
                # "text_prompt": f"{text_state}\n\n{self.description}",
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
        """Single-turn environment: check answer and terminate."""
        agent_id = next(iter(self._agent_ids))
        action_str = action[agent_id]

        answer = action_str.strip()

        # Check answer
        reward = self._score_answer(answer)
        correct = reward == 1.0

        obs = Observation(image=self.render(), text=None)
        info = {
            "oracle_answer": self._oracle_answer,
            "user_answer": action_str,
            "correct": correct,
            "question_type": self._current_q_type,
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
        """Render the word search grid as a PIL Image."""
        grid_width = (self._grid_size + 1) * self._cell_size
        grid_height = (self._grid_size + 1) * self._cell_size
        img_width = grid_width + self._margin
        img_height = grid_height + self._margin

        img = Image.new("RGB", (img_width, img_height), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Load font
        try:
            font = ImageFont.truetype(str(self.assets_dir / "DejaVuSans.ttf"), 32)
        except Exception:
            font = ImageFont.load_default()

        # Draw grid lines
        for i in range(self._grid_size + 1):
            # Vertical lines
            draw.line(
                [(i * self._cell_size, 0), (i * self._cell_size, grid_height)],
                fill=(0, 0, 0),
                width=1,
            )
            # Horizontal lines
            draw.line(
                [(0, i * self._cell_size), (grid_width, i * self._cell_size)],
                fill=(0, 0, 0),
                width=1,
            )

        # Draw row numbers (1-indexed)
        for i in range(self._grid_size):
            text = str(i + 1)
            draw.text(
                (5, (i + 1) * self._cell_size + 5),
                text,
                fill=(0, 0, 0),
                font=font,
            )

        # Draw column numbers (1-indexed)
        for j in range(self._grid_size):
            text = str(j + 1)
            draw.text(
                ((j + 1) * self._cell_size + 15, 5),
                text,
                fill=(0, 0, 0),
                font=font,
            )

        # Draw grid letters
        for i in range(self._grid_size):
            for j in range(self._grid_size):
                text = self._grid[i][j]
                draw.text(
                    ((j + 1) * self._cell_size + 15, (i + 1) * self._cell_size + 5),
                    text,
                    fill=(0, 0, 0),
                    font=font,
                )

        return img

    def _generate_grid(self) -> None:
        """Generate a random grid filled with random uppercase letters."""
        self._grid = [
            [random.choice(string.ascii_uppercase) for _ in range(self._grid_size)]
            for _ in range(self._grid_size)
        ]

    def _insert_word(
        self, word: str, start_row: int, start_col: int, direction: tuple[int, int]
    ) -> bool:
        """Insert a word into the grid at the specified position and direction.

        Args:
            word: The word to insert.
            start_row: Starting row index.
            start_col: Starting column index.
            direction: Direction tuple (row_delta, col_delta).

        Returns:
            True if word was successfully inserted, False otherwise.
        """
        curr_row, curr_col = start_row, start_col

        for letter in word:
            if 0 <= curr_row < self._grid_size and 0 <= curr_col < self._grid_size:
                self._grid[curr_row][curr_col] = letter
                curr_row += direction[0]
                curr_col += direction[1]
            else:
                return False

        return True

    def _can_place_word(
        self, word: str, start_row: int, start_col: int, direction: tuple[int, int]
    ) -> bool:
        """Check if a word can be placed at the specified position and direction."""
        curr_row, curr_col = start_row, start_col

        for _ in word:
            if not (
                0 <= curr_row < self._grid_size and 0 <= curr_col < self._grid_size
            ):
                return False
            curr_row += direction[0]
            curr_col += direction[1]

        return True

    def _generate_cell_letter_question(self) -> None:
        """Generate question about the letter at a specific cell (Easy, MCQ)."""
        row = random.randint(0, self._grid_size - 1)
        col = random.randint(0, self._grid_size - 1)
        correct_letter = self._grid[row][col]

        # Generate options
        options = [correct_letter]
        while len(options) < 8:
            letter = random.choice(string.ascii_uppercase)
            if letter not in options:
                options.append(letter)

        random.shuffle(options)
        correct_idx = options.index(correct_letter) + 1

        # Format question - separate question and options
        self._question = f"What letter is at row {row + 1}, column {col + 1}?"
        self._options = [f"{i + 1}: {opt}" for i, opt in enumerate(options)]
        self._oracle_answer = str(correct_idx)

    def _generate_letter_count_question(self) -> None:
        """Generate question about counting a specific letter (Medium, MCQ)."""
        letter = random.choice(string.ascii_uppercase)
        count = sum(row.count(letter) for row in self._grid)

        # Generate options
        options = [count]
        upper = 2
        while len(options) < 8:
            fake_count = random.randint(0, upper)
            upper += 1
            if fake_count not in options:
                options.append(fake_count)

        random.shuffle(options)
        correct_idx = options.index(count) + 1

        # Format question - separate question and options
        self._question = (
            f"How many times does the letter '{letter}' appear in the grid?"
        )
        self._options = [f"{i + 1}: {opt}" for i, opt in enumerate(options)]
        self._oracle_answer = str(correct_idx)

    def _generate_word_direction_question(self) -> None:
        """Generate question about word direction from a starting position (Medium, MCQ)."""
        # Select a word
        word = random.choice([w for w in self._words if 3 <= len(w) <= self._grid_size])

        # Find valid placements
        valid_placements = []
        for i in range(self._grid_size):
            for j in range(self._grid_size):
                for dir_name, direction in self.DIRECTIONS.items():
                    if self._can_place_word(word, i, j, direction):
                        valid_placements.append((i, j, dir_name, direction))

        if not valid_placements:
            # Fallback to cell_letter question
            self._generate_cell_letter_question()
            return

        # Choose random placement
        start_row, start_col, dir_name, direction = random.choice(valid_placements)
        self._insert_word(word, start_row, start_col, direction)

        # Generate options (all 8 directions)
        options = list(self.DIRECTIONS.keys())
        random.shuffle(options)
        correct_idx = options.index(dir_name) + 1

        # Format question - separate question and options
        self._question = f"Starting from position (row {start_row + 1}, column {start_col + 1}), in which direction can you find the word '{word}'?"
        self._options = [f"{i + 1}: {opt}" for i, opt in enumerate(options)]
        self._oracle_answer = str(correct_idx)

    def _generate_find_word_location_question(self) -> None:
        """Generate question about finding a word's location and direction (Hard, MCQ)."""
        # Select a word
        word = random.choice([w for w in self._words if 3 <= len(w) <= self._grid_size])

        # Find valid placements
        valid_placements = []
        for i in range(self._grid_size):
            for j in range(self._grid_size):
                for dir_name, direction in self.DIRECTIONS.items():
                    if self._can_place_word(word, i, j, direction):
                        valid_placements.append((i, j, dir_name, direction))

        if not valid_placements:
            # Fallback to cell_letter question
            self._generate_cell_letter_question()
            return

        # Choose random placement
        start_row, start_col, dir_name, direction = random.choice(valid_placements)
        self._insert_word(word, start_row, start_col, direction)

        # Generate correct option
        correct_option = (
            f"Row {start_row + 1}, Column {start_col + 1}, Direction: {dir_name}"
        )

        # Generate incorrect options
        first_letter = word[0]
        all_positions = []
        for i in range(self._grid_size):
            for j in range(self._grid_size):
                if self._grid[i][j] == first_letter:
                    for d_name in self.DIRECTIONS.keys():
                        option = f"Row {i + 1}, Column {j + 1}, Direction: {d_name}"
                        if option != correct_option:
                            all_positions.append(option)

        # Select 7 random incorrect options
        if len(all_positions) >= 7:
            incorrect_options = random.sample(all_positions, 7)
        else:
            incorrect_options = all_positions[:]
            # Add completely random options if needed
            while len(incorrect_options) < 7:
                rand_row = random.randint(1, self._grid_size)
                rand_col = random.randint(1, self._grid_size)
                rand_dir = random.choice(list(self.DIRECTIONS.keys()))
                option = f"Row {rand_row}, Column {rand_col}, Direction: {rand_dir}"
                if option not in incorrect_options and option != correct_option:
                    incorrect_options.append(option)

        # Combine and shuffle
        options = [correct_option] + incorrect_options
        random.shuffle(options)
        correct_idx = options.index(correct_option) + 1

        # Format question - separate question and options
        self._question = f"Find the word '{word}' in the grid. Where does it start and in which direction does it go?"
        self._options = [f"{i + 1}: {opt}" for i, opt in enumerate(options)]
        self._oracle_answer = str(correct_idx)

    def _check_answer(self, answer: str) -> bool:
        """Check if the provided answer is correct."""
        answer = answer.strip()
        oracle = self._oracle_answer.strip()

        # Accept just the number
        return answer == oracle

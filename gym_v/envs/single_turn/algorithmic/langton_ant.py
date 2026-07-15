"""Langton's Ant QA environment based on Game-RL."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation, get_logger
from gym_v.utils.gamerl_utils import build_description

logger = get_logger()

# Removed module-level QUESTION_TYPES - now defined as class variable

GAME_RULES = dedent("""
    This is Langton's Ant, a cellular automaton with simple rules:

    Rules:
    - At a WHITE cell: turn 90° RIGHT, flip cell to BLACK, move forward one unit
    - At a BLACK cell: turn 90° LEFT, flip cell to WHITE, move forward one unit
    - The ant wraps around the edges of the grid

    The grid uses a coordinate system where:
    - Row coordinates are 1-N from top to bottom
    - Column coordinates are 1-N from left to right
    - The top-left cell is (1, 1)
""").strip()


class LangtonAntQAEnv(Env):
    # Meta: source=GameRL, category=algorithmic, turn=single
    # Overrides: interaction_mode=single_turn, action_format=open_ended
    """Langton's Ant QA environment (single-turn question answering).

    Generates questions about Langton's Ant cellular automaton states.

    Args:
        question_type: Specific question type ID (0-2), or None for random
        grid_size: Grid size (default None for difficulty-based sizing)
        cell_size: Cell size in pixels for rendering (default 30)
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    # Directions
    DIRECTIONS = ["up", "right", "down", "left"]
    DIRECTION_VECTORS = {
        "up": (0, -1),
        "right": (1, 0),
        "down": (0, 1),
        "left": (-1, 0),
    }

    # Difficulty settings
    DIFFICULTIES = {
        "Easy": {"grid_size": 5, "init_steps": (3, 5)},
        "Medium": {"grid_size": 9, "init_steps": (5, 10)},
        "Hard": {"grid_size": 13, "init_steps": (10, 15)},
    }

    # Question types based on Game-RL langton_ant dataset_generator.py
    QUESTION_TYPES = [
        {
            "id": "current_state",
            "name": "Current State",
            "level": "Easy",
            "answer_format": "multiple_choice",
            "qa_type": "Target Perception",
        },
        {
            "id": "future_state",
            "name": "Future State",
            "level": "Medium",
            "answer_format": "multiple_choice",
            "qa_type": "State Prediction",
        },
        {
            "id": "cell_changes",
            "name": "Cell Changes",
            "level": "Hard",
            "answer_format": "fill_in_blank",
            "qa_type": "State Prediction",
        },
    ]

    def __init__(
        self,
        question_type: int | None = None,
        grid_size: int | None = None,
        cell_size: int = 30,
        num_players: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._question_type_param = question_type
        self._grid_size_override = grid_size
        self._cell_size = cell_size
        self._margin = 20
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        # Game state (initialized in reset)
        self._grid: list[list[int]] = []
        self._ant_x: int = 0
        self._ant_y: int = 0
        self._ant_direction: str = "up"
        self._difficulty: str = "Easy"
        self._grid_size: int = 5

        # Question state
        self._question: str = ""
        self._options: list[str] | None = None
        self._oracle_answer: str = ""
        self._current_q_type: dict[str, Any] = self.QUESTION_TYPES[0]

    @property
    def description(self) -> str:
        """Return game rules + current question + answer format."""
        return build_description(
            game_name="Langton's Ant",
            rules=GAME_RULES,
            question=self._question,
            options=self._options,
            oracle_answer=self._oracle_answer,
        )

    def _get_state_text(self, grid=True) -> str:
        """Generate text description of current Langton's Ant state.

        Returns a grid representation matching the rendered image.
        """
        grid = []
        for row in range(self._grid_size):
            row_chars = []
            for col in range(self._grid_size):
                if row == self._ant_y and col == self._ant_x:
                    row_chars.append("A")
                elif self._grid[row][col] == 1:
                    row_chars.append("#")
                else:
                    row_chars.append(".")
            grid.append("".join(row_chars))

        if grid:
            grid_str = "\n".join(grid)
            return f"""Grid Size: {self._grid_size}x{self._grid_size}
    Ant Position: ({self._ant_x}, {self._ant_y}) facing {self._ant_direction}
    Grid (A=ant, #=black, .=white):
    {grid_str}"""
        else:
            return f"""Grid Size: {self._grid_size}x{self._grid_size}
    Ant Position: ({self._ant_x}, {self._ant_y}) facing {self._ant_direction}"""

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        super().reset(seed=seed)

        # Select question type
        if self._question_type_param is not None:
            q_type = self.QUESTION_TYPES[self._question_type_param]
        else:
            q_type = self.py_random.choice(self.QUESTION_TYPES)

        self._current_q_type = q_type
        self._difficulty = q_type["level"]

        # Set grid size
        if self._grid_size_override is not None:
            self._grid_size = self._grid_size_override
        else:
            self._grid_size = self.DIFFICULTIES[self._difficulty]["grid_size"]

        # Generate initial state
        self._generate_initial_state()

        # Generate question
        if q_type["id"] == "current_state":
            self._generate_current_state_question()
        elif q_type["id"] == "future_state":
            self._generate_future_state_question()
        elif q_type["id"] == "cell_changes":
            self._generate_cell_changes_question()

        logger.info(
            f"Reset Langton's Ant QA ({self._difficulty}, question: {q_type['name']})."
        )

        text_state = self._get_state_text()
        obs = Observation(
            image=self.render(),
            text=None,
            metadata={
                "state_text": text_state,
                "text_prompt": f"{text_state}\n\n{self.description}",
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
        """Render the current grid state as a PIL Image."""
        grid_size = self._cell_size * self._grid_size
        padding = self._margin
        img_width = grid_size + 2 * padding
        img_height = grid_size + 2 * padding

        img = Image.new("RGB", (img_width, img_height), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Load font
        try:
            font = ImageFont.truetype(str(self.assets_dir / "DejaVuSans.ttf"), 14)
        except Exception:
            font = ImageFont.load_default()

        # Draw labels
        for i in range(self._grid_size):
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

        # Fill cells (white/black)
        for i in range(self._grid_size):
            for j in range(self._grid_size):
                x = padding + j * self._cell_size
                y = padding + i * self._cell_size
                color = (0, 0, 0) if self._grid[i][j] == 1 else (255, 255, 255)
                draw.rectangle(
                    [x, y, x + self._cell_size, y + self._cell_size],
                    fill=color,
                )

        # Draw grid lines
        for i in range(self._grid_size + 1):
            # Vertical lines
            draw.line(
                [
                    (padding + i * self._cell_size, padding),
                    (padding + i * self._cell_size, padding + grid_size),
                ],
                fill=(128, 128, 128),
                width=1,
            )
            # Horizontal lines
            draw.line(
                [
                    (padding, padding + i * self._cell_size),
                    (padding + grid_size, padding + i * self._cell_size),
                ],
                fill=(128, 128, 128),
                width=1,
            )

        # Draw ant as a red arrow
        ant_center_x = padding + self._ant_x * self._cell_size + self._cell_size // 2
        ant_center_y = padding + self._ant_y * self._cell_size + self._cell_size // 2
        arrow_size = self._cell_size // 3

        # Arrow points based on direction
        if self._ant_direction == "up":
            points = [
                (ant_center_x, ant_center_y - arrow_size),
                (ant_center_x - arrow_size // 2, ant_center_y + arrow_size // 2),
                (ant_center_x + arrow_size // 2, ant_center_y + arrow_size // 2),
            ]
        elif self._ant_direction == "down":
            points = [
                (ant_center_x, ant_center_y + arrow_size),
                (ant_center_x - arrow_size // 2, ant_center_y - arrow_size // 2),
                (ant_center_x + arrow_size // 2, ant_center_y - arrow_size // 2),
            ]
        elif self._ant_direction == "left":
            points = [
                (ant_center_x - arrow_size, ant_center_y),
                (ant_center_x + arrow_size // 2, ant_center_y - arrow_size // 2),
                (ant_center_x + arrow_size // 2, ant_center_y + arrow_size // 2),
            ]
        else:  # right
            points = [
                (ant_center_x + arrow_size, ant_center_y),
                (ant_center_x - arrow_size // 2, ant_center_y - arrow_size // 2),
                (ant_center_x - arrow_size // 2, ant_center_y + arrow_size // 2),
            ]

        draw.polygon(points, fill=(255, 0, 0))

        return img

    def _generate_initial_state(self) -> None:
        """Generate initial grid and ant state."""
        # Initialize grid (all white)
        self._grid = [[0] * self._grid_size for _ in range(self._grid_size)]

        # Place ant at center
        self._ant_x = self._grid_size // 2
        self._ant_y = self._grid_size // 2
        self._ant_direction = self.py_random.choice(self.DIRECTIONS)

        # Run some steps to create interesting patterns
        init_steps_range = self.DIFFICULTIES[self._difficulty]["init_steps"]
        num_steps = self.py_random.randint(*init_steps_range)
        for _ in range(num_steps):
            self._execute_step()

    def _execute_step(self) -> None:
        """Execute one step of Langton's Ant."""
        current_color = self._grid[self._ant_y][self._ant_x]

        # Turn and flip color
        if current_color == 0:  # white
            self._turn_right()
            self._grid[self._ant_y][self._ant_x] = 1  # flip to black
        else:  # black
            self._turn_left()
            self._grid[self._ant_y][self._ant_x] = 0  # flip to white

        # Move forward
        dx, dy = self.DIRECTION_VECTORS[self._ant_direction]
        self._ant_x = (self._ant_x + dx) % self._grid_size
        self._ant_y = (self._ant_y + dy) % self._grid_size

    def _turn_right(self) -> None:
        """Turn the ant 90 degrees right."""
        current_idx = self.DIRECTIONS.index(self._ant_direction)
        self._ant_direction = self.DIRECTIONS[(current_idx + 1) % 4]

    def _turn_left(self) -> None:
        """Turn the ant 90 degrees left."""
        current_idx = self.DIRECTIONS.index(self._ant_direction)
        self._ant_direction = self.DIRECTIONS[(current_idx - 1) % 4]

    def _generate_current_state_question(self) -> None:
        """Generate question about current ant position and direction (Easy, MCQ)."""
        # Correct answer
        correct_pos = (self._ant_y + 1, self._ant_x + 1)
        correct_dir = self._ant_direction
        correct_answer = f"({correct_pos[0]}, {correct_pos[1]}) facing {correct_dir}"

        # Generate distractors
        distractors = []
        while len(distractors) < 3:
            # Random position
            rand_y = self.py_random.randint(1, self._grid_size)
            rand_x = self.py_random.randint(1, self._grid_size)
            rand_dir = self.py_random.choice(self.DIRECTIONS)
            distractor = f"({rand_y}, {rand_x}) facing {rand_dir}"

            if distractor != correct_answer and distractor not in distractors:
                distractors.append(distractor)

        # Shuffle options
        options = [correct_answer] + distractors
        self.py_random.shuffle(options)
        correct_idx = options.index(correct_answer)
        correct_letter = chr(65 + correct_idx)  # A, B, C, D

        # Store question and options separately
        self._question = "What is the current position and direction of the ant?"
        self._options = [f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)]
        self._oracle_answer = correct_letter

    def _generate_future_state_question(self) -> None:
        """Generate question about ant state after N steps (Medium, MCQ)."""
        # Save current state
        saved_grid = [row[:] for row in self._grid]
        saved_x, saved_y = self._ant_x, self._ant_y
        saved_dir = self._ant_direction

        # Simulate N steps
        num_steps = self.py_random.randint(3, 7)
        for _ in range(num_steps):
            self._execute_step()

        # Correct answer
        correct_pos = (self._ant_y + 1, self._ant_x + 1)
        correct_dir = self._ant_direction
        correct_answer = f"({correct_pos[0]}, {correct_pos[1]}) facing {correct_dir}"

        # Restore state
        self._grid = saved_grid
        self._ant_x, self._ant_y = saved_x, saved_y
        self._ant_direction = saved_dir

        # Generate distractors
        distractors = []
        while len(distractors) < 3:
            rand_y = self.py_random.randint(1, self._grid_size)
            rand_x = self.py_random.randint(1, self._grid_size)
            rand_dir = self.py_random.choice(self.DIRECTIONS)
            distractor = f"({rand_y}, {rand_x}) facing {rand_dir}"

            if distractor != correct_answer and distractor not in distractors:
                distractors.append(distractor)

        # Shuffle options
        options = [correct_answer] + distractors
        self.py_random.shuffle(options)
        correct_idx = options.index(correct_answer)
        correct_letter = chr(65 + correct_idx)

        # Store question and options separately
        self._question = f"After {num_steps} steps from the current state, what will be the ant's position and direction?"
        self._options = [f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)]
        self._oracle_answer = correct_letter

    def _generate_cell_changes_question(self) -> None:
        """Generate question about how many times a specific cell changes color (Hard, Fill-in)."""
        # Choose a random target cell
        target_y = self.py_random.randint(0, self._grid_size - 1)
        target_x = self.py_random.randint(0, self._grid_size - 1)

        # Save current state
        saved_grid = [row[:] for row in self._grid]
        saved_x, saved_y = self._ant_x, self._ant_y
        saved_dir = self._ant_direction

        # Simulate N steps and count changes
        num_steps = self.py_random.randint(10, 20)
        change_count = 0

        for _ in range(num_steps):
            # Check if ant is at target cell before step
            if self._ant_x == target_x and self._ant_y == target_y:
                change_count += 1
            self._execute_step()

        # Restore state
        self._grid = saved_grid
        self._ant_x, self._ant_y = saved_x, saved_y
        self._ant_direction = saved_dir

        # Store question (fill-in-blank, no options)
        self._question = f"Consider cell at position ({target_y + 1}, {target_x + 1}). If the ant moves {num_steps} steps from the current state, how many times will this cell change its color?"
        self._options = None
        self._oracle_answer = str(change_count)

    def _check_answer(self, answer: str) -> bool:
        """Check if the provided answer is correct."""
        answer = answer.strip().upper()
        oracle = self._oracle_answer.strip().upper()

        # For choice questions, accept just the letter
        if self._current_q_type["answer_format"] == "choice":
            return answer == oracle or answer.startswith(oracle + ")")

        # For number questions, exact match
        return answer == oracle

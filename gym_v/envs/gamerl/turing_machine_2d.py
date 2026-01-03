"""2D Turing Machine QA game based on GameRL."""

from __future__ import annotations

from importlib import resources
import random
from textwrap import dedent
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation, get_logger

logger = get_logger()


class GameRL2dTuringMachineQAEnv(Env):
    """2D Turing Machine QA environment.

    A computational puzzle where a Turing machine head moves around a 2D grid,
    reading/writing symbols and changing states according to transition rules.

    Args:
        grid_size: Size of the grid (tuple of (rows, cols), default (5, 5))
        num_states: Number of machine states (default 2)
        num_symbols: Number of symbols (default 2)
        max_steps: Maximum simulation steps (default 8)
        cell_size: Size of each cell in pixels for rendering (default 50)
        question_type: Type of question to ask
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    # Question types
    QUESTION_TYPES = [
        {
            "id": "position",
            "name": "Future Head Position",
            "level": "Medium",
            "answer_format": "multiple_choice",
            "qa_type": "State Prediction",
        },
        {
            "id": "head_state",
            "name": "Future Head State",
            "level": "Medium",
            "answer_format": "multiple_choice",
            "qa_type": "State Prediction",
        },
        {
            "id": "symbol_at_position",
            "name": "Symbol at Position",
            "level": "Medium",
            "answer_format": "multiple_choice",
            "qa_type": "State Prediction",
        },
        {
            "id": "first_state_entry",
            "name": "First State Entry",
            "level": "Medium",
            "answer_format": "multiple_choice",
            "qa_type": "State Prediction",
        },
    ]

    # Colors for symbols
    COLORS = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 0, 255),
        (0, 255, 255),
        (255, 255, 0),
        (255, 128, 0),
        (128, 0, 255),
    ]
    COLOR_NAMES = [
        "red",
        "green",
        "blue",
        "magenta",
        "cyan",
        "yellow",
        "orange",
        "purple",
    ]

    # Brackets for states
    STATE_BRACKETS = ["()", "[]", "{}", "<>"]

    # Directions: 0=up, 1=right, 2=down, 3=left
    DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    DIRECTION_NAMES = ["up", "right", "down", "left"]

    def __init__(
        self,
        grid_size: tuple[int, int] | None = None,
        num_states: int = 2,
        num_symbols: int = 2,
        max_steps: int = 8,
        cell_size: int = 50,
        question_type: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._grid_size = grid_size if grid_size is not None else (5, 5)
        self._num_states = num_states
        self._num_symbols = num_symbols
        self._max_steps = max_steps
        self._cell_size = cell_size
        self._question_type = question_type

        # Game state (initialized in reset)
        self._grid: np.ndarray = np.zeros(self._grid_size, dtype=np.int8)
        self._head_x: int = 0
        self._head_y: int = 0
        self._current_state: int = 0
        self._rules: dict[tuple[int, int], tuple[int, int, int]] = {}
        self._current_question: dict[str, Any] = {}

    @property
    def description(self) -> str:
        base_desc = dedent(f"""
            This is a 2D Turing Machine QA environment.

            A Turing machine head moves around a 2D grid, following transition rules.
            The head reads a symbol at its current position, writes a new symbol,
            moves in a direction (up/right/down/left), and transitions to a new state.

            States are shown with brackets: {', '.join([f'{i}:{b}' for i, b in enumerate(self.STATE_BRACKETS[:self._num_states])])}
            Symbols are shown with colors: {', '.join([f'{i}:{n}' for i, n in enumerate(self.COLOR_NAMES[:self._num_symbols])])}

            Coordinates are (row, column) with (0,0) at top-left.

            Question Types:
            - Future Head Position: Where will the head be after N steps?
            - Future Head State: What state will the head be in after N steps?
            - Symbol at Position: What symbol will be at a position after N steps?
            - First State Entry: After how many steps will the head first enter a specific state?

            The system will present you with a machine configuration and ask a specific question.
        """).strip()

        # Add question and answer format if question has been generated
        if hasattr(self, "_current_question") and self._current_question:
            desc = base_desc + "\n\n" + self._current_question["question"]
            desc += """

**Answer Format:**
Reply with only the answer (number or option number).

Examples:
- For multiple choice: 1, 2, 3, etc.
- For numbers: 42, 100, etc.

Do not include any explanation or extra text.
"""
            return desc.strip()

        return base_desc

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)

        # Generate puzzle
        self._generate_puzzle()

        # Select question type
        if self._question_type is None:
            question_type = random.choice(self.QUESTION_TYPES)["id"]
        else:
            question_type = self._question_type

        # Generate question based on type
        steps = random.randint(3, self._max_steps)
        if question_type == "position":
            self._current_question = self._generate_position_question(steps)
        elif question_type == "head_state":
            self._current_question = self._generate_head_state_question(steps)
        elif question_type == "symbol_at_position":
            self._current_question = self._generate_symbol_at_position_question(steps)
        elif question_type == "first_state_entry":
            self._current_question = self._generate_first_state_entry_question(steps)
        else:
            raise ValueError(f"Unknown question type: {question_type}")

        logger.info(
            f"Reset 2D Turing Machine QA ({self._grid_size[0]}x{self._grid_size[1]}, "
            f"question: {question_type})."
        )

        obs = Observation(image=self.render(), text=self._current_question["question"])

        info = {
            "oracle_answer": self._current_question["answer"],
            "question_type": question_type,
        }

        return obs, info

    def inner_step(
        self, action: str
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        info: dict[str, Any] = {}
        reward = 0.0
        terminated = True
        truncated = False

        # Parse answer
        action = action.strip()

        # Check if answer is correct
        correct = self._check_answer(action)

        if correct:
            reward = 1.0
            response = "Correct!"
        else:
            reward = 0.0
            response = (
                f"Incorrect. The correct answer is: {self._current_question['answer']}"
            )

        info = {
            "correct": correct,
            "user_answer": action,
            "oracle_answer": self._current_question["answer"],
        }

        obs = Observation(image=self.render(), text=response)
        return obs, reward, terminated, truncated, info

    def render(self) -> Image.Image:
        """Render the current Turing machine state as a PIL Image."""
        rows, cols = self._grid_size
        margin = 40
        img_width = cols * self._cell_size + 2 * margin
        img_height = rows * self._cell_size + 2 * margin

        img = Image.new("RGB", (img_width, img_height), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Load font
        try:
            font = ImageFont.truetype(str(self.assets_dir / "DejaVuSans.ttf"), 20)
            small_font = ImageFont.truetype(str(self.assets_dir / "DejaVuSans.ttf"), 14)
        except Exception:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()

        # Draw grid with symbols
        for y in range(rows):
            for x in range(cols):
                cell_x = margin + x * self._cell_size
                cell_y = margin + y * self._cell_size

                # Draw cell border
                draw.rectangle(
                    [
                        cell_x,
                        cell_y,
                        cell_x + self._cell_size,
                        cell_y + self._cell_size,
                    ],
                    outline=(0, 0, 0),
                    width=2,
                )

                # Draw symbol color
                symbol = self._grid[y, x]
                color = self.COLORS[symbol % len(self.COLORS)]
                padding = 5
                draw.rectangle(
                    [
                        cell_x + padding,
                        cell_y + padding,
                        cell_x + self._cell_size - padding,
                        cell_y + self._cell_size - padding,
                    ],
                    fill=color,
                    outline=(0, 0, 0),
                    width=1,
                )

                # Draw head if at this position
                if x == self._head_x and y == self._head_y:
                    left, right = self.STATE_BRACKETS[self._current_state]
                    text = f"{left}{symbol}{right}"
                    draw.text(
                        (cell_x + self._cell_size // 2, cell_y + self._cell_size // 2),
                        text,
                        fill=(255, 255, 255),
                        font=font,
                        anchor="mm",
                        stroke_width=2,
                        stroke_fill=(0, 0, 0),
                    )

        # Draw row/column labels
        for y in range(rows):
            draw.text(
                (margin // 2, margin + y * self._cell_size + self._cell_size // 2),
                str(y),
                fill=(0, 0, 0),
                font=small_font,
                anchor="mm",
            )

        for x in range(cols):
            draw.text(
                (margin + x * self._cell_size + self._cell_size // 2, margin // 2),
                str(x),
                fill=(0, 0, 0),
                font=small_font,
                anchor="mm",
            )

        return img

    def _generate_puzzle(self):
        """Generate a random 2D Turing machine configuration."""
        rows, cols = self._grid_size

        # Initialize grid with random symbols
        self._grid = np.random.randint(
            0, self._num_symbols, size=self._grid_size, dtype=np.int8
        )

        # Place head at random position
        self._head_y = random.randint(0, rows - 1)
        self._head_x = random.randint(0, cols - 1)
        self._current_state = 0

        # Generate transition rules
        self._rules = {}
        for state in range(self._num_states):
            for symbol in range(self._num_symbols):
                new_symbol = random.randint(0, self._num_symbols - 1)
                direction = random.randint(0, 3)
                new_state = random.randint(0, self._num_states - 1)
                self._rules[(state, symbol)] = (new_symbol, direction, new_state)

    def _simulate_steps(self, num_steps: int) -> list[tuple[int, int, int, int]]:
        """Simulate N steps and return positions."""
        # Save state
        saved_grid = self._grid.copy()
        saved_x, saved_y, saved_state = self._head_x, self._head_y, self._current_state

        positions = [
            (
                self._head_x,
                self._head_y,
                self._grid[self._head_y, self._head_x],
                self._current_state,
            )
        ]

        try:
            for _ in range(num_steps):
                symbol = self._grid[self._head_y, self._head_x]
                new_symbol, direction, new_state = self._rules[
                    (self._current_state, symbol)
                ]

                self._grid[self._head_y, self._head_x] = new_symbol
                self._current_state = new_state

                dx, dy = self.DIRECTIONS[direction]
                self._head_x += dx
                self._head_y += dy

                # Check bounds
                if not (
                    0 <= self._head_x < self._grid_size[1]
                    and 0 <= self._head_y < self._grid_size[0]
                ):
                    break

                positions.append(
                    (
                        self._head_x,
                        self._head_y,
                        self._grid[self._head_y, self._head_x],
                        self._current_state,
                    )
                )
        finally:
            # Restore state
            self._grid = saved_grid
            self._head_x, self._head_y, self._current_state = (
                saved_x,
                saved_y,
                saved_state,
            )

        return positions

    def _generate_position_question(self, steps: int) -> dict[str, Any]:
        """Generate question about future head position."""
        positions = self._simulate_steps(steps)
        if len(positions) <= steps:
            steps = len(positions) - 1

        final_x, final_y = positions[steps][0], positions[steps][1]
        correct_answer = (final_y, final_x)

        # Generate options
        options = [correct_answer]
        rows, cols = self._grid_size
        while len(options) < 8:
            opt = (random.randint(0, rows - 1), random.randint(0, cols - 1))
            if opt not in options:
                options.append(opt)

        random.shuffle(options)
        correct_idx = options.index(correct_answer) + 1

        rules_text = self._get_rules_description()
        options_text = "\n".join(
            [f"{i+1}: ({opt[0]}, {opt[1]})" for i, opt in enumerate(options)]
        )

        question = f"""{rules_text}

Current head position: ({self._head_y}, {self._head_x})
Current state: {self._current_state}

Question: Where will the head be after {steps} steps?

Options:
{options_text}"""

        return {"question": question, "answer": str(correct_idx), "options": options}

    def _generate_head_state_question(self, steps: int) -> dict[str, Any]:
        """Generate question about future head state."""
        positions = self._simulate_steps(steps)
        if len(positions) <= steps:
            steps = len(positions) - 1

        final_state = positions[steps][3]
        correct_answer = final_state

        # Generate options
        options = list(range(self._num_states))
        random.shuffle(options)
        correct_idx = options.index(correct_answer) + 1

        rules_text = self._get_rules_description()
        options_text = "\n".join(
            [
                f"{i+1}: State {opt} {self.STATE_BRACKETS[opt]}"
                for i, opt in enumerate(options)
            ]
        )

        question = f"""{rules_text}

Current head position: ({self._head_y}, {self._head_x})
Current state: {self._current_state}

Question: What state will the head be in after {steps} steps?

Options:
{options_text}"""

        return {"question": question, "answer": str(correct_idx), "options": options}

    def _generate_symbol_at_position_question(self, steps: int) -> dict[str, Any]:
        """Generate question about symbol at a specific position."""
        positions = self._simulate_steps(steps)
        if len(positions) <= steps:
            steps = len(positions) - 1

        # Pick a random position on the grid
        rows, cols = self._grid_size
        target_y = random.randint(0, rows - 1)
        target_x = random.randint(0, cols - 1)

        # Get final grid state
        saved_grid = self._grid.copy()
        self._simulate_steps(steps)
        final_symbol = self._grid[target_y, target_x]
        self._grid = saved_grid

        correct_answer = final_symbol

        # Generate options
        options = list(range(self._num_symbols))
        random.shuffle(options)
        correct_idx = options.index(correct_answer) + 1

        rules_text = self._get_rules_description()
        options_text = "\n".join(
            [
                f"{i+1}: Symbol {opt} ({self.COLOR_NAMES[opt]})"
                for i, opt in enumerate(options)
            ]
        )

        question = f"""{rules_text}

Current head position: ({self._head_y}, {self._head_x})
Current state: {self._current_state}

Question: What symbol will be at position ({target_y}, {target_x}) after {steps} steps?

Options:
{options_text}"""

        return {"question": question, "answer": str(correct_idx), "options": options}

    def _generate_first_state_entry_question(self, max_steps: int) -> dict[str, Any]:
        """Generate question about first entry to a state."""
        positions = self._simulate_steps(max_steps)

        # Track first entry to each state
        state_entry_times = {}
        for step, (_x, _y, _symbol, state) in enumerate(positions):
            if state not in state_entry_times:
                state_entry_times[state] = step

        # Choose a target state (not the initial state if possible)
        possible_states = [
            s for s in state_entry_times.keys() if s != self._current_state
        ]
        if not possible_states:
            possible_states = list(state_entry_times.keys())

        target_state = random.choice(possible_states)
        correct_answer = state_entry_times[target_state]

        # Generate options
        options = [correct_answer]
        while len(options) < 8:
            opt = random.randint(0, max(max_steps, 8))
            if opt not in options:
                options.append(opt)

        random.shuffle(options)
        correct_idx = options.index(correct_answer) + 1

        rules_text = self._get_rules_description()
        options_text = "\n".join(
            [f"{i+1}: {opt} steps" for i, opt in enumerate(options)]
        )

        question = f"""{rules_text}

Current head position: ({self._head_y}, {self._head_x})
Current state: {self._current_state}

Question: After how many steps will the head first enter state {target_state} {self.STATE_BRACKETS[target_state]}?

Options:
{options_text}"""

        return {"question": question, "answer": str(correct_idx), "options": options}

    def _get_rules_description(self) -> str:
        """Get description of transition rules."""
        desc = "Transition Rules:\n"
        for (state, symbol), (new_symbol, direction, new_state) in self._rules.items():
            desc += (
                f"State {state}, Symbol {symbol} ({self.COLOR_NAMES[symbol]}) -> "
                f"Write {new_symbol}, Move {self.DIRECTION_NAMES[direction]}, New State {new_state}\n"
            )

        desc += "\nColor Legend:\n"
        for i in range(self._num_symbols):
            desc += f"Symbol {i}: {self.COLOR_NAMES[i]}\n"

        desc += "\nState Brackets:\n"
        for i in range(self._num_states):
            desc += f"State {i}: {self.STATE_BRACKETS[i]}\n"

        return desc.strip()

    def _check_answer(self, action: str) -> bool:
        """Check if the provided answer is correct."""
        correct_answer = self._current_question["answer"]
        return action.strip().lower() == correct_answer.strip().lower()

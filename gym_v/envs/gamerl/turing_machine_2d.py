"""2D Turing Machine QA game based on GameRL."""

from __future__ import annotations

from importlib import resources
import io
import random
from textwrap import dedent
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

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

    # Colors for symbols (matplotlib color names)
    COLORS = [
        "red",
        "green",
        "blue",
        "magenta",
        "cyan",
        "yellow",
        "orange",
        "purple",
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
        question_type: int | None = None,
        num_players: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._grid_size = grid_size if grid_size is not None else (5, 5)
        self._num_states = num_states
        self._num_symbols = num_symbols
        self._max_steps = max_steps
        self._cell_size = cell_size
        self._question_type_param = question_type
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        # Game state (initialized in reset)
        self._grid: np.ndarray = np.zeros(self._grid_size, dtype=np.int8)
        self._head_x: int = 0
        self._head_y: int = 0
        self._current_state: int = 0
        self._rules: dict[tuple[int, int], tuple[int, int, int]] = {}

        # Standard QA variables
        self._question_type_idx: int = 0
        self._question: str = ""
        self._options: list[str] | None = None
        self._oracle_answer: str = ""

    ANSWER_FORMAT_PROMPT = dedent("""
        **Answer Format:**
        Reply with only the answer (number or option number).
        For multiple choice: 1, 2, 3, etc.
    """).strip()

    @property
    def description(self) -> str:
        """Return game rules + current question + answer format."""
        rules = f"""A Turing machine head moves around a 2D grid, following transition rules.
The head reads a symbol, writes a new symbol, moves in a direction, and transitions to a new state.
States: {', '.join([f'{i}:{b}' for i, b in enumerate(self.STATE_BRACKETS[:self._num_states])])}
Symbols: {', '.join([f'{i}:{n}' for i, n in enumerate(self.COLOR_NAMES[:self._num_symbols])])}
Coordinates are (row, column) with (0,0) at top-left."""

        desc = rules + "\n\n**Question:** " + self._question
        if self._options:
            desc += "\n\n**Options:**\n"
            for i, opt in enumerate(self._options):
                desc += f"{i+1}. {opt}\n"
        desc += "\n\n" + self.ANSWER_FORMAT_PROMPT
        return desc.strip()

    def _get_state_text(self) -> str:
        """Generate text description of current Turing machine state."""
        text = "2D Turing Machine State\n"
        text += f"Grid Size: {self._grid_size[0]}x{self._grid_size[1]}\n"
        text += f"Head Position: ({self._head_y}, {self._head_x})\n"
        text += f"Current State: {self._current_state} {self.STATE_BRACKETS[self._current_state]}\n\n"

        text += "Transition Rules:\n"
        for (state, symbol), (write_symbol, move_dir, new_state) in self._rules.items():
            text += f"  State {state}, Symbol {symbol} ({self.COLOR_NAMES[symbol]}) → "
            text += f"Write {write_symbol} ({self.COLOR_NAMES[write_symbol]}), "
            text += f"Move {self.DIRECTION_NAMES[move_dir]}, "
            text += f"New State {new_state}\n"

        text += "\nGrid State:\n"
        for y in range(self._grid_size[0]):
            row = []
            for x in range(self._grid_size[1]):
                symbol = self._grid[y, x]
                if (x, y) == (self._head_x, self._head_y):
                    row.append(f"[{symbol}]")  # Mark head position
                else:
                    row.append(f" {symbol} ")
            text += "".join(row) + "\n"

        text += "\nColor Legend:\n"
        for i in range(self._num_symbols):
            text += f"  Symbol {i}: {self.COLOR_NAMES[i]}\n"

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

        # Generate question based on type
        steps = random.randint(3, self._max_steps)
        if q_type["id"] == "position":
            result = self._generate_position_question(steps)
        elif q_type["id"] == "head_state":
            result = self._generate_head_state_question(steps)
        elif q_type["id"] == "symbol_at_position":
            result = self._generate_symbol_at_position_question(steps)
        elif q_type["id"] == "first_state_entry":
            result = self._generate_first_state_entry_question(steps)
        else:
            raise ValueError(f"Unknown question type: {q_type['id']}")

        # Extract to instance variables
        self._question = result["question"]
        self._options = result.get("options")
        self._oracle_answer = result["answer"]

        logger.info(
            f"Reset 2D Turing Machine QA ({self._grid_size[0]}x{self._grid_size[1]}, "
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

        # Parse answer
        action_str = action_str.strip()

        # Check if answer is correct
        correct = action_str.strip().lower() == self._oracle_answer.strip().lower()

        if correct:
            reward = 1.0
            response = "Correct!"
        else:
            reward = 0.0
            response = f"Incorrect. The correct answer is: {self._oracle_answer}"

        info = {
            "correct": correct,
            "user_answer": action_str,
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
        """Render the current Turing machine state using matplotlib (matching original)."""
        rows, cols = self._grid_size

        fig, ax = plt.subplots(figsize=(6, 6))

        # Assign colors for each symbol
        color_map = [
            self.COLORS[i % len(self.COLORS)] for i in range(self._grid.max() + 1)
        ]

        # Display the grid with the assigned colors
        color_grid = np.empty(self._grid.shape, dtype=object)
        for y in range(self._grid.shape[0]):
            for x in range(self._grid.shape[1]):
                color_grid[y, x] = color_map[self._grid[y, x]]

        # Draw colored rectangles for each cell
        for y in range(self._grid.shape[0]):
            for x in range(self._grid.shape[1]):
                ax.add_patch(plt.Rectangle((x, y), 1, 1, color=color_grid[y, x]))

        # Mark the head position with a black circle
        ax.scatter(
            self._head_x + 0.5,
            self._head_y + 0.5,
            color="black",
            s=200,
            edgecolors="white",
            label="Head",
        )

        # Set limits, labels, and legend
        ax.set_xlim(0, self._grid.shape[1])
        ax.set_ylim(0, self._grid.shape[0])
        num_rows, num_cols = self._grid.shape
        ax.set_xticks([x + 0.5 for x in range(num_cols)])
        ax.set_yticks([y + 0.5 for y in range(num_rows)])
        ax.set_xticklabels(range(self._grid.shape[1]))
        ax.set_yticklabels(range(self._grid.shape[0]))
        ax.xaxis.tick_top()  # Move x-axis ticks to the top
        ax.set_aspect("equal")
        ax.set_title("2D Turning Machine", fontsize=14, fontweight="bold")
        ax.legend(loc="upper center", bbox_to_anchor=(1.15, 1))
        plt.gca().invert_yaxis()  # Flip the y-axis

        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="PNG", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)

        return Image.open(buf).convert("RGB")

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
        return action.strip().lower() == self._oracle_answer.strip().lower()

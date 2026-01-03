"""Conway's Game of Life Q&A environment based on GameRL.

Single-turn Q&A environment where the model answers questions about Game of Life states.
"""

from __future__ import annotations

from importlib import resources
import random
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation, get_logger

logger = get_logger()


# Removed module-level QUESTION_TYPES - now defined as class variable

GAME_RULES = """Conway's Game of Life is a cellular automaton where each cell in the grid can be either alive (black) or dead (white).

Each cell interacts with its eight neighbors, which are the cells that are horizontally, vertically, or diagonally adjacent. For a cell at position (r,c), its neighbors are:
- (r-1,c-1)  (r-1,c)  (r-1,c+1)   [above row]
- (r,c-1)     (r,c)    (r,c+1)     [same row]
- (r+1,c-1)  (r+1,c)  (r+1,c+1)   [below row]

Region boundaries wrap around to the opposite side:
- A cell at the top edge connects to cells at the bottom edge
- A cell at the left edge connects to cells at the right edge
- Corner cells connect to the diagonally opposite corner

The game evolves in discrete steps according to these rules:
1. Any live cell with fewer than two live neighbors dies (underpopulation)
2. Any live cell with two or three live neighbors lives on to the next generation
3. Any live cell with more than three live neighbors dies (overpopulation)
4. Any dead cell with exactly three live neighbors becomes alive (reproduction)

In the image, black squares represent live cells, white squares represent dead cells, and the grid lines help visualize the cell boundaries.

Coordinate System: In this grid, we use (row, col) coordinates where row increases from top to bottom (0 at top) and col increases from left to right (0 at left). For example, the top-left cell is at (0, 0), and the cell below it is at (1, 0)."""

ANSWER_FORMAT_PROMPT = """
**Answer Format:**
- For multiple choice: Reply with only the letter (A, B, C, D, E, F, G, or H)

Do not include any explanation or extra text."""


class GameRLLifegameQAEnv(Env):
    """Game of Life Q&A environment.

    Single-turn Q&A environment based on the original Game-RL lifegame.
    Given a game state image, answer questions about cell counts, evolution,
    or stability.

    Args:
        question_type: Question type ID (0-3). None for random selection.
        grid_size: Size of grid (default varies by difficulty)
        cell_size: Size of each cell in pixels for rendering (default 30)
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    # Question types from original Game-RL
    QUESTION_TYPES = [
        {
            "id": "state_info",
            "name": "State Info",
            "level": "Easy",
            "answer_format": "multiple_choice",
            "qa_type": "StateInfo",
        },
        {
            "id": "action_outcome",
            "name": "Action Outcome",
            "level": "Medium",
            "answer_format": "multiple_choice",
            "qa_type": "ActionOutcome",
        },
        {
            "id": "cell_change_count",
            "name": "Cell Change Count",
            "level": "Medium",
            "answer_format": "multiple_choice",
            "qa_type": "CellChangeCount",
        },
        {
            "id": "stability_steps",
            "name": "Stability Steps",
            "level": "Hard",
            "answer_format": "multiple_choice",
            "qa_type": "StabilitySteps",
        },
    ]

    # Grid sizes by difficulty
    GRID_SIZES = {"Easy": 3, "Medium": 4, "Hard": 5}

    # Colors
    COLORS = {
        "white": (255, 255, 255),
        "black": (0, 0, 0),
        "gray": (128, 128, 128),
    }

    def __init__(
        self,
        question_type: int | None = None,
        grid_size: int | None = None,
        cell_size: int = 30,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._question_type = question_type
        self._override_grid_size = grid_size
        self._cell_size = cell_size
        self._margin = 40

        # Game state (initialized in reset)
        self._grid: list[list[int]] = []
        self._grid_size: int = 3

        # Q&A state (initialized in reset)
        self._current_question_type: int = 0
        self._current_question: str = ""
        self._oracle_answer: str = ""
        self._answer_format: str = ""
        self._options: list[str] = []
        self._difficulty: str = "Easy"

    @property
    def description(self) -> str:
        """Return game rules + current question + answer format."""
        desc = GAME_RULES + "\n\n**Question:** " + self._current_question

        if self._options:
            desc += "\n\nOptions:\n" + "\n".join(self._options)

        desc += ANSWER_FORMAT_PROMPT
        return desc.strip()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)

        if seed is not None:
            random.seed(seed)

        # Select question type
        if self._question_type is not None:
            self._current_question_type = self._question_type
        else:
            self._current_question_type = self.np_random.integers(
                0, len(self.QUESTION_TYPES)
            )

        # Determine difficulty and grid size
        self._difficulty = self.QUESTION_TYPES[self._current_question_type]["level"]
        if self._override_grid_size is not None:
            self._grid_size = self._override_grid_size
        else:
            self._grid_size = self.GRID_SIZES[self._difficulty]

        # Generate game state
        self._generate_grid()

        # Generate question and answer
        self._generate_qa()

        logger.info(
            f"Reset Lifegame QA (type={self._current_question_type}, grid={self._grid_size}x{self._grid_size})."
        )

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
        """Render the current grid state as a PIL Image."""
        width = self._grid_size * self._cell_size + self._margin * 2
        height = self._grid_size * self._cell_size + self._margin * 2
        img = Image.new("RGB", (width, height), self.COLORS["white"])
        draw = ImageDraw.Draw(img)

        offset = (width - self._grid_size * self._cell_size) // 2

        # Draw cells
        for x in range(self._grid_size):
            for y in range(self._grid_size):
                color = (
                    self.COLORS["black"]
                    if self._grid[x][y] == 1
                    else self.COLORS["white"]
                )
                top_left = (offset + y * self._cell_size, offset + x * self._cell_size)
                bottom_right = (
                    offset + (y + 1) * self._cell_size,
                    offset + (x + 1) * self._cell_size,
                )
                draw.rectangle([top_left, bottom_right], fill=color, outline=None)

        # Draw grid lines
        for x in range(self._grid_size + 1):
            start = (offset, offset + x * self._cell_size)
            end = (
                offset + self._grid_size * self._cell_size,
                offset + x * self._cell_size,
            )
            draw.line([start, end], fill=self.COLORS["gray"], width=1)

        for y in range(self._grid_size + 1):
            start = (offset + y * self._cell_size, offset)
            end = (
                offset + y * self._cell_size,
                offset + self._grid_size * self._cell_size,
            )
            draw.line([start, end], fill=self.COLORS["gray"], width=1)

        # Draw coordinates
        try:
            font = ImageFont.truetype(str(self.assets_dir / "DejaVuSans.ttf"), 20)
        except Exception:
            font = ImageFont.load_default()

        for x in range(self._grid_size):
            text = str(x)
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            pos = (
                offset - 5 - text_width,
                offset + x * self._cell_size + self._cell_size // 2 - text_height // 2,
            )
            draw.text(pos, text, fill=self.COLORS["black"], font=font)

        for y in range(self._grid_size):
            text = str(y)
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            pos = (
                offset + y * self._cell_size + self._cell_size // 2 - text_width // 2,
                offset - 5 - text_height,
            )
            draw.text(pos, text, fill=self.COLORS["black"], font=font)

        return img

    def _generate_grid(self) -> None:
        """Generate a random grid with 20-40% live cells."""
        self._grid = [[0] * self._grid_size for _ in range(self._grid_size)]
        for x in range(self._grid_size):
            for y in range(self._grid_size):
                self._grid[x][y] = 1 if self.np_random.random() < 0.3 else 0

    def _count_neighbors(self, x: int, y: int) -> int:
        """Count live neighbors for a cell."""
        count = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx = (x + dx) % self._grid_size
                ny = (y + dy) % self._grid_size
                count += self._grid[nx][ny]
        return count

    def _update_grid(self, grid: list[list[int]]) -> list[list[int]]:
        """Update grid according to Conway's rules."""
        size = len(grid)
        new_grid = [[0] * size for _ in range(size)]

        for x in range(size):
            for y in range(size):
                neighbors = self._count_neighbors_for_grid(grid, x, y)
                if grid[x][y] == 1:
                    new_grid[x][y] = 1 if neighbors in [2, 3] else 0
                else:
                    new_grid[x][y] = 1 if neighbors == 3 else 0

        return new_grid

    def _count_neighbors_for_grid(self, grid: list[list[int]], x: int, y: int) -> int:
        """Count neighbors for any grid."""
        size = len(grid)
        count = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx = (x + dx) % size
                ny = (y + dy) % size
                count += grid[nx][ny]
        return count

    def _generate_qa(self) -> None:
        """Generate question and answer based on question type."""
        qtype = self._current_question_type

        if qtype == 0:
            self._generate_state_info_qa()
        elif qtype == 1:
            self._generate_action_outcome_qa()
        elif qtype == 2:
            self._generate_cell_change_qa()
        elif qtype == 3:
            self._generate_stability_qa()

    def _generate_state_info_qa(self) -> None:
        """Q: How many live cells are currently in the grid?"""
        correct_count = sum(sum(row) for row in self._grid)

        # Generate options
        options, correct_idx = self._generate_numeric_options(
            correct_count, min_val=0, max_val=self._grid_size * self._grid_size
        )

        self._current_question = "How many live cells are currently in the grid?"
        self._options = [f"{chr(65 + i)}: {opt}" for i, opt in enumerate(options)]
        self._oracle_answer = chr(65 + correct_idx)

    def _generate_action_outcome_qa(self) -> None:
        """Q: How many live cells after 1 iteration?"""
        next_grid = self._update_grid(self._grid)
        correct_count = sum(sum(row) for row in next_grid)

        options, correct_idx = self._generate_numeric_options(
            correct_count, min_val=0, max_val=self._grid_size * self._grid_size
        )

        self._current_question = (
            "After 1 iteration, how many live cells will remain in the grid?"
        )
        self._options = [f"{chr(65 + i)}: {opt}" for i, opt in enumerate(options)]
        self._oracle_answer = chr(65 + correct_idx)

    def _generate_cell_change_qa(self) -> None:
        """Q: Track state changes of a specific cell over iterations."""
        # Select a cell with some activity
        target_cell = self._select_active_cell()
        if target_cell is None:
            # Fallback to random cell
            target_cell = (
                self.np_random.integers(0, self._grid_size),
                self.np_random.integers(0, self._grid_size),
            )

        row, col = target_cell
        iterations = {"Easy": 4, "Medium": 3, "Hard": 2}.get(self._difficulty, 3)

        # Track state sequence
        current_grid = [r[:] for r in self._grid]
        state_sequence = [current_grid[row][col]]

        for _ in range(iterations):
            current_grid = self._update_grid(current_grid)
            state_sequence.append(current_grid[row][col])

        # Generate options
        sequence_options, correct_idx = self._generate_sequence_options(state_sequence)

        # Convert to text
        options_text = [self._sequence_to_text(seq) for seq in sequence_options]

        self._current_question = (
            f"Consider the cell at position ({row}, {col}). "
            f"How will its state change over the next {iterations} iterations?"
        )
        self._options = [f"{chr(65 + i)}: {opt}" for i, opt in enumerate(options_text)]
        self._oracle_answer = chr(65 + correct_idx)

    def _generate_stability_qa(self) -> None:
        """Q: How many iterations until a 3x3 region reaches stability?"""
        region_size = 3

        # Select a region
        if self._grid_size < 3:
            # Grid too small, use whole grid
            row_start, col_start = 0, 0
            region_size = self._grid_size
        else:
            row_start = self.np_random.integers(0, self._grid_size)
            col_start = self.np_random.integers(0, self._grid_size)

        # Calculate stability steps
        steps = self._calculate_region_stability(row_start, col_start, region_size)

        # Generate options
        options, correct_idx = self._generate_numeric_options(
            steps, min_val=0, max_val=max(steps + 5, 10)
        )

        self._current_question = (
            f"Consider the {region_size}x{region_size} region starting at cell ({row_start},{col_start}). "
            "When analyzing this region's stability, we treat it as an independent Game of Life system. "
            "The region is stable when all cells maintain their current states or form a repeating pattern. "
            "How many iterations will it take for this region to reach a stable state?"
        )
        self._options = [f"{chr(65 + i)}: {opt}" for i, opt in enumerate(options)]
        self._oracle_answer = chr(65 + correct_idx)

    def _select_active_cell(self) -> tuple[int, int] | None:
        """Select a cell that will have state changes."""
        candidates = []
        for x in range(self._grid_size):
            for y in range(self._grid_size):
                # Check if this cell will change in next few steps
                test_grid = [r[:] for r in self._grid]
                changed = False
                for _ in range(3):
                    next_grid = self._update_grid(test_grid)
                    if next_grid[x][y] != test_grid[x][y]:
                        changed = True
                        break
                    test_grid = next_grid
                if changed:
                    candidates.append((x, y))

        if not candidates:
            return None
        return random.choice(candidates)

    def _calculate_region_stability(
        self, row_start: int, col_start: int, region_size: int
    ) -> int:
        """Calculate steps until region reaches stability."""

        def extract_region(grid: list[list[int]]) -> list[list[int]]:
            region = []
            for i in range(region_size):
                row = []
                for j in range(region_size):
                    actual_row = (row_start + i) % len(grid)
                    actual_col = (col_start + j) % len(grid)
                    row.append(grid[actual_row][actual_col])
                region.append(row)
            return region

        def is_region_stable(region: list[list[int]]) -> bool:
            for i in range(len(region)):
                for j in range(len(region)):
                    neighbors = 0
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni = (i + di) % len(region)
                            nj = (j + dj) % len(region)
                            neighbors += region[ni][nj]

                    if region[i][j] == 1:
                        if neighbors < 2 or neighbors > 3:
                            return False
                    else:
                        if neighbors == 3:
                            return False
            return True

        current_grid = [r[:] for r in self._grid]
        history = []
        max_steps = 20

        for step in range(max_steps):
            current_region = extract_region(current_grid)
            region_hash = tuple(map(tuple, current_region))

            if is_region_stable(current_region):
                return step

            if region_hash in history:
                return step

            history.append(region_hash)
            current_grid = self._update_grid(current_grid)

        return max_steps - 1

    def _generate_numeric_options(
        self,
        correct_answer: int,
        num_options: int = 8,
        min_val: int = 0,
        max_val: int = 100,
    ) -> tuple[list[int], int]:
        """Generate numeric options including correct answer."""
        min_val = min(min_val, correct_answer)
        max_val = max(max_val, correct_answer)

        candidates = list(range(min_val, max_val + 1))
        if correct_answer in candidates:
            candidates.remove(correct_answer)

        # Expand range if needed
        while len(candidates) < num_options - 1:
            if min_val > 0:
                min_val -= 1
                candidates.insert(0, min_val)
            max_val += 1
            candidates.append(max_val)

        wrong_options = random.sample(candidates, min(num_options - 1, len(candidates)))
        correct_index = random.randint(0, num_options - 1)

        options = wrong_options.copy()
        options.insert(correct_index, correct_answer)

        return options, correct_index

    def _generate_sequence_options(
        self, correct_sequence: list[int], num_options: int = 8
    ) -> tuple[list[list[int]], int]:
        """Generate sequence options."""
        unique_sequences = {tuple(correct_sequence)}

        # Generate variations
        while len(unique_sequences) < num_options:
            variant = correct_sequence.copy()
            # Flip 1-2 random positions
            num_flips = random.randint(1, min(2, len(variant)))
            positions = random.sample(range(len(variant)), num_flips)
            for pos in positions:
                variant[pos] = 1 - variant[pos]
            unique_sequences.add(tuple(variant))

        sequences = [
            list(seq) for seq in unique_sequences if seq != tuple(correct_sequence)
        ]
        sequences = sequences[: num_options - 1]

        correct_index = random.randint(0, num_options - 1)
        sequences.insert(correct_index, correct_sequence)

        return sequences, correct_index

    def _sequence_to_text(self, sequence: list[int]) -> str:
        """Convert state sequence to text."""
        states = []
        for i, state in enumerate(sequence):
            if i == 0:
                states.append(f"Initially: {'alive' if state == 1 else 'dead'}")
            else:
                states.append(f"Step {i}: {'alive' if state == 1 else 'dead'}")
        return " → ".join(states)

    def _check_answer(self, user_answer: str) -> bool:
        """Check if user's answer is correct."""
        # Normalize both answers
        user_answer = user_answer.strip().upper()
        oracle_answer = self._oracle_answer.strip().upper()

        return user_answer == oracle_answer

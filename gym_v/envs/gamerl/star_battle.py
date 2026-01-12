"""Star Battle QA game based on GameRL."""

from __future__ import annotations

from collections import deque
import random
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw

from gym_v import Env, Observation, get_logger

logger = get_logger()


class GameRLStarBattleQAEnv(Env):
    """Star Battle QA environment.

    A logic puzzle where you must place stars in a grid following rules:
    - Each row must contain exactly `stars_per_region` stars
    - Each column must contain exactly `stars_per_region` stars
    - Each region must contain exactly `stars_per_region` stars
    - Stars cannot be adjacent (including diagonally)

    Args:
        grid_size: Size of the grid (5, 6, or 8)
        stars_per_region: Number of stars per row/column/region (default 1)
        cell_size: Size of each cell in pixels for rendering (default 50)
        question_type: Type of question to ask
    """

    # Color mappings for regions
    REGION_COLORS = {
        0: {"name": "light pink", "rgb": (255, 182, 193)},
        1: {"name": "powder blue", "rgb": (176, 224, 230)},
        2: {"name": "light green", "rgb": (144, 238, 144)},
        3: {"name": "peach", "rgb": (255, 218, 185)},
        4: {"name": "red", "rgb": (255, 0, 0)},
        5: {"name": "yellow", "rgb": (255, 255, 0)},
        6: {"name": "cyan", "rgb": (0, 255, 255)},
        7: {"name": "orange", "rgb": (255, 165, 0)},
        8: {"name": "purple", "rgb": (128, 0, 128)},
    }

    # Question types
    QUESTION_TYPES = [
        {
            "id": "last_star",
            "name": "Last Star Placement",
            "level": "Hard",
            "answer_format": "fill_in_blank",
            "qa_type": "TransitionPath",
        },
        {
            "id": "cells_of_region",
            "name": "Cells of Region",
            "level": "Easy",
            "answer_format": "multiple_choice",
            "qa_type": "StateInfo",
        },
        {
            "id": "star_of_region",
            "name": "Star of Region",
            "level": "Easy",
            "answer_format": "multiple_choice",
            "qa_type": "StateInfo",
        },
        {
            "id": "valid_cells",
            "name": "Valid Cell Placement",
            "level": "Medium",
            "answer_format": "multiple_choice",
            "qa_type": "StateInfo",
        },
    ]

    GAME_RULES = dedent("""
        Star Battle Rules:
        - Each row must contain exactly {stars_per_region} star(s)
        - Each column must contain exactly {stars_per_region} star(s)
        - Each region must contain exactly {stars_per_region} star(s)
        - Stars cannot be adjacent to each other, including diagonally
        - Cells are labeled with (row, column) starting from (0, 0) at top-left
    """).strip()

    ANSWER_FORMAT_PROMPT = {
        "tuple": "Provide your answer as a coordinate tuple (row, col), e.g., (3, 2).",
        "multiple_choice": "Select the option number from the given choices.",
    }

    def __init__(
        self,
        grid_size: int = 6,
        stars_per_region: int = 1,
        cell_size: int = 50,
        question_type: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if grid_size not in [5, 6, 8]:
            raise ValueError(f"Grid size must be 5, 6, or 8, got {grid_size}")

        self._grid_size = grid_size
        self._stars_per_region = stars_per_region
        self._cell_size = cell_size
        self._question_type = question_type

        # Game state (initialized in reset)
        self._grid: list[list[int]] = []
        self._regions: list[list[tuple[int, int]]] = []
        self._solution: list[list[int]] = []
        self._current_question: dict[str, Any] = {}

    @property
    def description(self) -> str:
        rules = self.GAME_RULES.format(stars_per_region=self._stars_per_region)
        base_desc = dedent(f"""
            This is a Star Battle puzzle QA environment.

            {rules}

            In this puzzle, the grid is divided into regions shown by different colors.
            A star is represented by a black circle in the grid.

            Question Types:
            - Last Star Placement: Find where to place the final star to complete the puzzle
            - Cells of Region: Identify which cell belongs to a specific region
            - Star of Region: Find which cell in a region contains a star
            - Valid Cell Placement: Determine valid cells for placing a star

            The system will present you with a puzzle state and ask a specific question.
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

    def _get_state_text(self) -> str:
        """Generate text description of current Star Battle puzzle state.

        Returns a grid representation matching the rendered image.
        """
        # Create a grid showing region numbers
        region_grid = [[0] * self._grid_size for _ in range(self._grid_size)]
        for region_idx, region_cells in enumerate(self._regions):
            for row, col in region_cells:
                region_grid[row][col] = region_idx + 1

        # Create text representation
        grid = []
        for row in range(self._grid_size):
            row_chars = []
            for col in range(self._grid_size):
                if self._grid[row][col] == 1:
                    row_chars.append("*")
                else:
                    row_chars.append(str(region_grid[row][col]))
            grid.append("".join(row_chars))

        grid_str = "\n".join(grid)
        return f"""Grid Size: {self._grid_size}x{self._grid_size}
Grid (*=star, 1-{self._grid_size}=region number):
{grid_str}"""

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)

        # Select question type
        if self._question_type is None:
            question_type_idx = random.randint(0, len(self.QUESTION_TYPES) - 1)
        else:
            question_type_idx = self._question_type

        # Validate question type index
        if not (0 <= question_type_idx < len(self.QUESTION_TYPES)):
            raise ValueError(f"Invalid question type index: {question_type_idx}")

        # Get question type ID from index
        question_type = self.QUESTION_TYPES[question_type_idx]["id"]

        # Generate puzzle
        self._generate_puzzle()

        # Generate question based on type
        if question_type == "last_star":
            self._current_question = self._generate_last_star_question()
        elif question_type == "cells_of_region":
            self._current_question = self._generate_cells_of_region_question()
        elif question_type == "star_of_region":
            self._current_question = self._generate_star_of_region_question()
        elif question_type == "valid_cells":
            self._current_question = self._generate_valid_cells_question()
        else:
            raise ValueError(f"Unknown question type: {question_type}")

        logger.info(
            f"Reset Star Battle QA ({self._grid_size}x{self._grid_size}, question: {question_type})."
        )

        obs = Observation(
            image=self.render(),
            text=self._get_state_text(),
            metadata={
                "question": self._current_question["question"],
                "options": self._current_question.get("options"),
                "question_type": question_type,
            },
        )

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

        if "explanation" in self._current_question:
            response += f"\n\nExplanation:\n{self._current_question['explanation']}"

        info = {
            "correct": correct,
            "user_answer": action,
            "oracle_answer": self._current_question["answer"],
        }

        obs = Observation(image=self.render(), text=response)
        return obs, reward, terminated, truncated, info

    def render(self) -> Image.Image | list[Image.Image] | None:
        """Render the current puzzle state as a PIL Image."""
        img_width = self._grid_size * self._cell_size
        img_height = self._grid_size * self._cell_size

        img = Image.new("RGB", (img_width, img_height), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Fill regions with colors
        for region_idx, region in enumerate(self._regions):
            color = self.REGION_COLORS[region_idx % len(self.REGION_COLORS)]["rgb"]
            for row, col in region:
                x = col * self._cell_size
                y = row * self._cell_size
                draw.rectangle(
                    [x, y, x + self._cell_size, y + self._cell_size],
                    fill=color,
                )

        # Draw grid lines
        for i in range(self._grid_size + 1):
            # Vertical lines
            draw.line(
                [
                    (i * self._cell_size, 0),
                    (i * self._cell_size, img_height),
                ],
                fill=(0, 0, 0),
                width=2,
            )
            # Horizontal lines
            draw.line(
                [
                    (0, i * self._cell_size),
                    (img_width, i * self._cell_size),
                ],
                fill=(0, 0, 0),
                width=2,
            )

        # Draw stars
        for row in range(self._grid_size):
            for col in range(self._grid_size):
                if self._grid[row][col] == 1:
                    center_x = col * self._cell_size + self._cell_size // 2
                    center_y = row * self._cell_size + self._cell_size // 2
                    radius = self._cell_size // 4
                    draw.ellipse(
                        [
                            center_x - radius,
                            center_y - radius,
                            center_x + radius,
                            center_y + radius,
                        ],
                        fill=(0, 0, 0),
                    )

        return img

    def _generate_puzzle(self):
        """Generate a solvable Star Battle puzzle."""
        # Generate regions
        self._regions = self._generate_regions()

        # Generate solution
        self._solution = [[0] * self._grid_size for _ in range(self._grid_size)]
        self._solve_puzzle(self._solution)

        # Copy solution to grid
        self._grid = [row[:] for row in self._solution]

    def _generate_regions(self) -> list[list[tuple[int, int]]]:
        """Generate n connected regions for the grid."""
        while True:
            cells = [
                (r, c) for r in range(self._grid_size) for c in range(self._grid_size)
            ]
            random.shuffle(cells)
            regions = []
            unassigned_cells = set(cells)

            def get_neighbors(cell):
                r, c = cell
                neighbors = []
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self._grid_size and 0 <= nc < self._grid_size:
                        neighbors.append((nr, nc))
                return neighbors

            # Create exactly n regions
            while len(regions) < self._grid_size and unassigned_cells:
                start_cell = random.choice(list(unassigned_cells))
                current_region = []
                queue = deque([start_cell])
                unassigned_cells.remove(start_cell)
                current_region.append(start_cell)

                # Grow the region
                target_size = max(len(cells) // self._grid_size, 1)
                while queue and len(current_region) < target_size:
                    cell = queue.popleft()
                    neighbors = get_neighbors(cell)
                    random.shuffle(neighbors)

                    for neighbor in neighbors:
                        if neighbor in unassigned_cells:
                            unassigned_cells.remove(neighbor)
                            current_region.append(neighbor)
                            queue.append(neighbor)

                if current_region:
                    regions.append(current_region)

            # If we have exactly n regions, distribute remaining cells
            if len(regions) == self._grid_size:
                while unassigned_cells:
                    cell = unassigned_cells.pop()
                    neighbors = get_neighbors(cell)
                    adjacent_regions = []
                    for i, region in enumerate(regions):
                        for neighbor in neighbors:
                            if neighbor in region:
                                adjacent_regions.append(i)
                                break

                    if adjacent_regions:
                        region_idx = random.choice(adjacent_regions)
                        regions[region_idx].append(cell)
                    else:
                        unassigned_cells.add(cell)

                return regions

    def _is_valid(self, grid: list[list[int]], row: int, col: int) -> bool:
        """Check if placing a star at (row, col) is valid."""
        # Check row and column constraints
        if (
            sum(grid[row]) >= self._stars_per_region
            or sum(grid[r][col] for r in range(self._grid_size))
            >= self._stars_per_region
        ):
            return False

        # Check adjacent cells
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                nr, nc = row + dr, col + dc
                if (
                    0 <= nr < self._grid_size
                    and 0 <= nc < self._grid_size
                    and grid[nr][nc] == 1
                ):
                    return False

        # Check region constraints
        for region in self._regions:
            if (row, col) in region:
                stars_in_region = sum(grid[r][c] for r, c in region)
                if stars_in_region >= self._stars_per_region:
                    return False

        return True

    def _solve_puzzle(self, grid: list[list[int]], cell: int = 0) -> bool:
        """Solve the Star Battle puzzle using backtracking."""
        if cell == self._grid_size * self._grid_size:
            for region in self._regions:
                stars_in_region = sum(grid[r][c] for r, c in region)
                if stars_in_region != self._stars_per_region:
                    return False
            return True

        row, col = divmod(cell, self._grid_size)
        if grid[row][col] == 0:
            if self._is_valid(grid, row, col):
                grid[row][col] = 1
                if self._solve_puzzle(grid, cell + 1):
                    return True
                grid[row][col] = 0

        return self._solve_puzzle(grid, cell + 1)

    def _generate_last_star_question(self) -> dict[str, Any]:
        """Generate a question about placing the last star."""
        # Remove one star to create the puzzle
        preplaced_stars = []
        answer = None

        # Randomly select n-1 regions to keep their stars
        number = self._grid_size - 1
        selected_regions = random.sample(range(self._grid_size), number)

        for row in range(self._grid_size):
            for col in range(self._grid_size):
                if self._solution[row][col] == 1:
                    region_idx = self._get_region_index(row, col)
                    if region_idx not in selected_regions:
                        self._grid[row][col] = 0
                        answer = (row, col)
                    else:
                        preplaced_stars.append((row, col))

        # Generate color description
        color_description = "\n".join(
            [
                f"Region {i} has the color of {self.REGION_COLORS[i]['name']}"
                for i in range(self._grid_size)
            ]
        )

        question = f"""We have a {self._grid_size}x{self._grid_size} grid divided into {self._grid_size} regions.
Cells with the same color belong to the same region.
{color_description}

The current grid already has {len(preplaced_stars)} stars placed at: {preplaced_stars}

Your task is to find the location of the final star to complete the puzzle.

Rules:
- Each row must contain exactly {self._stars_per_region} star(s)
- Each column must contain exactly {self._stars_per_region} star(s)
- Each region must contain exactly {self._stars_per_region} star(s)
- Stars cannot be adjacent to each other, including diagonally
- Cells are labeled (row, col) starting from (0, 0) at top-left

Where should the final star be placed?"""

        return {
            "question": question,
            "answer": str(answer),
            "answer_tuple": answer,
        }

    def _generate_cells_of_region_question(self) -> dict[str, Any]:
        """Generate a question about which cell belongs to a region."""
        # Randomly remove some stars
        self._remove_random_stars()

        # Select a random region
        region_idx = random.randint(0, self._grid_size - 1)
        correct_cells = self._regions[region_idx]
        correct_cell = random.choice(correct_cells)

        # Generate options (7 wrong + 1 correct)
        options = []
        while len(options) < 7:
            distractor_region = random.randint(0, self._grid_size - 1)
            if distractor_region != region_idx:
                distractor_cell = random.choice(self._regions[distractor_region])
                option_str = f"({distractor_cell[0]},{distractor_cell[1]})"
                if option_str not in options:
                    options.append(option_str)

        options.append(f"({correct_cell[0]},{correct_cell[1]})")
        random.shuffle(options)

        # Find correct answer label
        correct_answer_label = None
        for idx, option in enumerate(options, 1):
            if option == f"({correct_cell[0]},{correct_cell[1]})":
                correct_answer_label = idx
                break

        # Generate color description
        color_description = "\n".join(
            [
                f"Region {i} has the color of {self.REGION_COLORS[i]['name']}"
                for i in range(self._grid_size)
            ]
        )

        # Generate option text
        option_text = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)])

        question = f"""We have a {self._grid_size}x{self._grid_size} grid divided into {self._grid_size} regions.
Cells with the same color belong to the same region.
{color_description}

In the image, a star is represented by a black dot.

Rules:
- Each row must contain exactly {self._stars_per_region} star(s)
- Each column must contain exactly {self._stars_per_region} star(s)
- Each region must contain exactly {self._stars_per_region} star(s)
- Stars cannot be adjacent to each other, including diagonally

The region with index {region_idx} is represented by the color {self.REGION_COLORS[region_idx]['name']}.
Which cell in the following options belongs to region {region_idx}?

Options:
{option_text}"""

        return {
            "question": question,
            "answer": str(correct_answer_label),
            "options": options,
        }

    def _generate_star_of_region_question(self) -> dict[str, Any]:
        """Generate a question about which cell in a region has a star."""
        # Randomly remove some stars
        self._remove_random_stars()

        # Select a random region
        region_idx = random.randint(0, self._grid_size - 1)
        region_cells = self._regions[region_idx]

        # Find the star in this region
        correct_cell = None
        for row, col in region_cells:
            if self._grid[row][col] == 1:
                correct_cell = (row, col)
                break

        correct_answer = (
            f"({correct_cell[0]},{correct_cell[1]})" if correct_cell else "null"
        )

        # Generate options from the region
        options = [correct_answer]
        while len(options) < min(8, len(region_cells)):
            random_cell = random.choice(region_cells)
            option_str = f"({random_cell[0]},{random_cell[1]})"
            if option_str not in options:
                options.append(option_str)

        # Add more options from other regions if needed
        if len(region_cells) < 8:
            remaining_cells = []
            for i, region in enumerate(self._regions):
                if i != region_idx:
                    remaining_cells.extend(region)

            while len(options) < 8:
                random_cell = random.choice(remaining_cells)
                option_str = f"({random_cell[0]},{random_cell[1]})"
                if option_str not in options:
                    options.append(option_str)

        random.shuffle(options)

        # Find correct answer label
        correct_answer_label = None
        for idx, option in enumerate(options, 1):
            if option == correct_answer:
                correct_answer_label = idx
                break

        # Generate color description
        color_description = "\n".join(
            [
                f"Region {i} has the color of {self.REGION_COLORS[i]['name']}"
                for i in range(self._grid_size)
            ]
        )

        # Generate option text
        option_text = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)])

        question = f"""We have a {self._grid_size}x{self._grid_size} grid divided into {self._grid_size} regions.
Cells with the same color belong to the same region.
{color_description}

In the image, a star is represented by a black dot.

Rules:
- Each row must contain exactly {self._stars_per_region} star(s)
- Each column must contain exactly {self._stars_per_region} star(s)
- Each region must contain exactly {self._stars_per_region} star(s)
- Stars cannot be adjacent to each other, including diagonally

Region {region_idx} is represented by the color {self.REGION_COLORS[region_idx]['name']}.
Which of the following cells in this region contains a star?

Note: If no stars have been placed in the target region, choose the option "null".

Options:
{option_text}"""

        return {
            "question": question,
            "answer": str(correct_answer_label),
            "options": options,
        }

    def _generate_valid_cells_question(self) -> dict[str, Any]:
        """Generate a question about valid cell placements."""
        # Randomly remove some stars
        self._remove_random_stars()

        # Find valid and invalid cells
        valid_cells = []
        invalid_cells = []
        for row in range(self._grid_size):
            for col in range(self._grid_size):
                if self._grid[row][col] == 0:
                    if self._is_valid(self._grid, row, col):
                        valid_cells.append((row, col))
                    else:
                        invalid_cells.append((row, col))

        # Select one correct answer and distractors
        if valid_cells:
            correct_cell = random.choice(valid_cells)
            correct_answer = f"({correct_cell[0]},{correct_cell[1]})"
        else:
            correct_answer = "null"

        options = [correct_answer]

        # Add distractors
        distractors_size = min(7, len(invalid_cells))
        distractors = random.sample(invalid_cells, distractors_size)
        for distractor in distractors:
            options.append(f"({distractor[0]},{distractor[1]})")

        random.shuffle(options)

        # Find correct answer label
        correct_answer_label = None
        for idx, option in enumerate(options, 1):
            if option == correct_answer:
                correct_answer_label = idx
                break

        # Generate color description
        color_description = "\n".join(
            [
                f"Region {i} has the color of {self.REGION_COLORS[i]['name']}"
                for i in range(self._grid_size)
            ]
        )

        # Generate option text
        option_text = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)])

        question = f"""We have a {self._grid_size}x{self._grid_size} grid divided into {self._grid_size} regions.
Cells with the same color belong to the same region.
{color_description}

In the image, a star is represented by a black dot.
Some stars have already been placed in the grid.

Rules:
- Each row must contain exactly {self._stars_per_region} star(s)
- Each column must contain exactly {self._stars_per_region} star(s)
- Each region must contain exactly {self._stars_per_region} star(s)
- Stars cannot be adjacent to each other, including diagonally

Based on the current puzzle state, which of the following cells can a star be placed in?

Options:
{option_text}"""

        return {
            "question": question,
            "answer": str(correct_answer_label),
            "options": options,
        }

    def _remove_random_stars(self):
        """Randomly remove some stars from the grid."""
        number = random.randint(2, self._grid_size)
        selected_regions = random.sample(range(self._grid_size), number)

        for row in range(self._grid_size):
            for col in range(self._grid_size):
                if self._solution[row][col] == 1:
                    region_idx = self._get_region_index(row, col)
                    if region_idx not in selected_regions:
                        self._grid[row][col] = 0

    def _get_region_index(self, row: int, col: int) -> int:
        """Get the region index for a cell."""
        for idx, region in enumerate(self._regions):
            if (row, col) in region:
                return idx
        return -1

    def _check_answer(self, action: str) -> bool:
        """Check if the provided answer is correct."""
        correct_answer = self._current_question["answer"]

        # Normalize both answers for comparison
        action_normalized = action.strip().lower()
        correct_normalized = correct_answer.strip().lower()

        # For tuple answers, try to parse and compare
        if "answer_tuple" in self._current_question:
            try:
                # Try to parse as tuple
                action_tuple = eval(action_normalized)
                correct_tuple = self._current_question["answer_tuple"]
                return action_tuple == correct_tuple
            except (ValueError, IndexError, NameError, SyntaxError):
                pass

        # For multiple choice, compare the option number
        return action_normalized == correct_normalized

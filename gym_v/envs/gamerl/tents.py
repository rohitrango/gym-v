"""Tents puzzle QA game based on GameRL."""

from __future__ import annotations

from importlib import resources
import random
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation, get_logger

logger = get_logger()


class GameRLTentsQAEnv(Env):
    """Tents puzzle QA environment.

    A logic puzzle where you place tents adjacent to trees following rules:
    - Each cell can be empty, contain a tree, or contain a tent
    - Total number of tents must equal number of trees
    - Tents must be placed horizontally or vertically adjacent to at least one tree
    - No two tents can be adjacent (including diagonally)
    - Row and column constraints: numbers indicate how many tents should be in each row/column

    Args:
        grid_size: Size of the grid (tuple of (width, height))
        num_trees: Number of trees in the puzzle
        cell_size: Size of each cell in pixels for rendering (default 50)
        question_type: Type of question to ask
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    # Question types
    QUESTION_TYPES = [
        {
            "id": "num_tents_in_row",
            "name": "Number of Tents in Row",
            "level": "Easy",
            "answer_format": "fill_in_blank",
            "qa_type": "Target Perception",
        },
        {
            "id": "num_missing_tents_in_column",
            "name": "Missing Tents in Column",
            "level": "Easy",
            "answer_format": "fill_in_blank",
            "qa_type": "Target Perception",
        },
        {
            "id": "num_missing_tents_in_grid",
            "name": "Missing Tents in Grid",
            "level": "Medium",
            "answer_format": "fill_in_blank",
            "qa_type": "Target Perception",
        },
        {
            "id": "possible_tent_positions",
            "name": "Possible Tent Positions",
            "level": "Medium",
            "answer_format": "fill_in_blank",
            "qa_type": "State Prediction",
        },
        {
            "id": "new_tent_count",
            "name": "New Tent Count",
            "level": "Hard",
            "answer_format": "fill_in_blank",
            "qa_type": "State Prediction",
        },
        {
            "id": "tree_position",
            "name": "Tree Position",
            "level": "Easy",
            "answer_format": "multiple_choice",
            "qa_type": "Target Perception",
        },
        {
            "id": "new_tent_position",
            "name": "New Tent Position",
            "level": "Medium",
            "answer_format": "multiple_choice",
            "qa_type": "State Prediction",
        },
    ]

    GAME_RULES = dedent("""
        Tents Puzzle Rules:
        1. Each cell can only be in one of three states: empty, containing a tree, or containing a tent
        2. The total number of tents must equal the number of trees
        3. Tents can only be placed horizontally or vertically adjacent to at least one tree
        4. No two tents can be adjacent, including diagonally
        5. Row and column constraints: numbers indicate how many tents should be in each row/column

        - Tree positions are marked with green circles
        - Tent positions are marked with orange triangles
        - Blue numbers on the left show required tents per row
        - Blue numbers on top show required tents per column
        - Coordinates (x, y): x is row number, y is column number
        - Numbering starts from 0, origin (0,0) is in the upper-left corner
    """).strip()

    def __init__(
        self,
        grid_size: tuple[int, int] | None = None,
        num_trees: int | None = None,
        cell_size: int = 50,
        question_type: int | None = None,
        num_players: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Default grid size and num_trees based on difficulty
        if grid_size is None or num_trees is None:
            difficulty = random.choice(["Easy", "Medium", "Hard"])
            if difficulty == "Easy":
                self._grid_size = (7, 7)
                self._num_trees = 5
            elif difficulty == "Medium":
                self._grid_size = (10, 10)
                self._num_trees = 10
            else:  # Hard
                self._grid_size = (13, 13)
                self._num_trees = 17
        else:
            self._grid_size = grid_size
            self._num_trees = num_trees

        self._cell_size = cell_size
        self._question_type_param = question_type
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        # Game state (initialized in reset)
        self._grid: list[list[str]] = []
        self._tree_positions: set[tuple[int, int]] = set()
        self._tent_positions: set[tuple[int, int]] = set()
        self._row_tent_counts: list[int] = []
        self._col_tent_counts: list[int] = []

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
        desc = self.GAME_RULES + "\n\n**Question:** " + self._question
        if self._options:
            desc += "\n\n**Options:**\n"
            for i, opt in enumerate(self._options):
                desc += f"{i+1}. {opt}\n"
        desc += "\n\n" + self.ANSWER_FORMAT_PROMPT
        return desc.strip()

    def _get_state_text(self) -> str:
        """Generate text description of current Tents puzzle state.

        Returns a grid representation matching the rendered image.
        """
        width, height = self._grid_size
        grid = []
        for row in range(height):
            row_chars = []
            for col in range(width):
                cell = self._grid[row][col]
                if cell == "T":
                    row_chars.append("T")
                elif cell == "X":
                    row_chars.append("X")
                else:
                    row_chars.append(".")
            grid.append("".join(row_chars))

        grid_str = "\n".join(grid)
        return f"""Grid Size: {width}x{height}
Grid (T=tree, X=tent, .=empty):
{grid_str}"""

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
        if q_type["id"] == "num_tents_in_row":
            result = self._generate_num_tents_in_row_question()
        elif q_type["id"] == "num_missing_tents_in_column":
            result = self._generate_missing_tents_in_column_question()
        elif q_type["id"] == "num_missing_tents_in_grid":
            result = self._generate_missing_tents_in_grid_question()
        elif q_type["id"] == "possible_tent_positions":
            result = self._generate_possible_tent_positions_question()
        elif q_type["id"] == "new_tent_count":
            result = self._generate_new_tent_count_question()
        elif q_type["id"] == "tree_position":
            result = self._generate_tree_position_question()
        elif q_type["id"] == "new_tent_position":
            result = self._generate_new_tent_position_question()
        else:
            raise ValueError(f"Unknown question type: {q_type['id']}")

        # Extract to instance variables
        self._question = result["question"]
        self._options = result.get("options")
        self._oracle_answer = result["answer"]

        logger.info(
            f"Reset Tents QA ({self._grid_size[0]}x{self._grid_size[1]}, "
            f"{self._num_trees} trees, question: {q_type['id']})."
        )

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

        if "explanation" in self._question:
            response += f"\n\nExplanation:\n{self._question['explanation']}"

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
        """Render the current puzzle state as a PIL Image (matching original matplotlib implementation)."""
        width, height = self._grid_size
        margin = 80  # Increased margin to avoid overlap
        img_width = width * self._cell_size + 2 * margin
        img_height = height * self._cell_size + 2 * margin

        img = Image.new("RGB", (img_width, img_height), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Load font (matching original fontsize=18 and fontsize=15)
        try:
            font = ImageFont.truetype(str(self.assets_dir / "DejaVuSans.ttf"), 18)
            small_font = ImageFont.truetype(str(self.assets_dir / "DejaVuSans.ttf"), 15)
        except Exception:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()

        # Draw grid
        for i in range(height + 1):
            y = margin + i * self._cell_size
            draw.line(
                [(margin, y), (margin + width * self._cell_size, y)],
                fill=(0, 0, 0),
                width=2,
            )

        for j in range(width + 1):
            x = margin + j * self._cell_size
            draw.line(
                [(x, margin), (x, margin + height * self._cell_size)],
                fill=(0, 0, 0),
                width=2,
            )

        # Draw row numbers and tent counts (on the left)
        # Matching original: row number at -width/7-0.1, tent count closer
        for i in range(height):
            y = margin + i * self._cell_size + self._cell_size // 2
            # Row tent count (blue) - closer to grid (matching original position)
            draw.text(
                (margin - self._cell_size // 3, y),
                str(self._row_tent_counts[i]),
                fill=(0, 51, 204),
                font=font,
                anchor="mm",
            )
            # Row number (black) - further from grid
            draw.text(
                (margin - int(self._cell_size * 1.2), y),
                str(i),
                fill=(0, 0, 0),
                font=small_font,
                anchor="mm",
            )

        # Draw column numbers and tent counts (on top)
        # Matching original: column number at -width/7-0.1, tent count closer
        for j in range(width):
            x = margin + j * self._cell_size + self._cell_size // 2
            # Column tent count (blue) - closer to grid (matching original position)
            draw.text(
                (x, margin - self._cell_size // 3),
                str(self._col_tent_counts[j]),
                fill=(0, 51, 204),
                font=font,
                anchor="mm",
            )
            # Column number (black) - further from grid
            draw.text(
                (x, margin - int(self._cell_size * 1.2)),
                str(j),
                fill=(0, 0, 0),
                font=small_font,
                anchor="mm",
            )

        # Draw trees and tents using actual images (matching original matplotlib implementation)
        try:
            # Load tree and tent images
            tree_img = Image.open(self.assets_dir / "tents" / "tree.png").convert(
                "RGBA"
            )
            tent_img = Image.open(self.assets_dir / "tents" / "tent.png").convert(
                "RGBA"
            )

            # Resize images to fit in cells
            img_size = int(self._cell_size * 0.9)  # 90% of cell size
            tree_img = tree_img.resize((img_size, img_size), Image.Resampling.LANCZOS)
            tent_img = tent_img.resize((img_size, img_size), Image.Resampling.LANCZOS)

            # Draw trees
            for row, col in self._tree_positions:
                x = margin + col * self._cell_size + (self._cell_size - img_size) // 2
                y = margin + row * self._cell_size + (self._cell_size - img_size) // 2
                img.paste(tree_img, (x, y), tree_img)

            # Draw tents
            for row, col in self._tent_positions:
                x = margin + col * self._cell_size + (self._cell_size - img_size) // 2
                y = margin + row * self._cell_size + (self._cell_size - img_size) // 2
                img.paste(tent_img, (x, y), tent_img)

        except Exception as e:
            # Fallback to simple shapes if images can't be loaded
            logger.warning(f"Could not load tree/tent images: {e}, using simple shapes")

            # Draw trees (green circles with brown trunk - fallback)
            for row, col in self._tree_positions:
                x = margin + col * self._cell_size + self._cell_size // 2
                y = margin + row * self._cell_size + self._cell_size // 2
                radius = self._cell_size // 4
                draw.ellipse(
                    [x - radius, y - radius, x + radius, y + radius],
                    fill=(34, 139, 34),  # Forest green
                    outline=(0, 100, 0),
                    width=2,
                )

            # Draw tents (orange triangles - fallback)
            for row, col in self._tent_positions:
                x = margin + col * self._cell_size + self._cell_size // 2
                y = margin + row * self._cell_size + self._cell_size // 2
                size = self._cell_size // 3
                # Triangle pointing up
                points = [
                    (x, y - size),  # Top
                    (x - size, y + size // 2),  # Bottom left
                    (x + size, y + size // 2),  # Bottom right
                ]
                draw.polygon(points, fill=(255, 140, 0), outline=(200, 100, 0), width=2)

        return img

    def _generate_puzzle(self):
        """Generate a Tents puzzle."""
        width, height = self._grid_size

        while True:
            self._grid = [["" for _ in range(width)] for _ in range(height)]
            tent_available = [[False for _ in range(width)] for _ in range(height)]

            # Place trees
            self._tree_positions = set()
            while len(self._tree_positions) < self._num_trees:
                x, y = random.randint(0, width - 1), random.randint(0, height - 1)
                if self._grid[x][y] == "":
                    potential_positions = [
                        (x + 1, y),
                        (x - 1, y),
                        (x, y + 1),
                        (x, y - 1),
                    ]
                    for tx, ty in potential_positions:
                        if (
                            0 <= ty < width
                            and 0 <= tx < height
                            and self._grid[tx][ty] == ""
                        ):
                            tent_available[tx][ty] = True
                    self._tree_positions.add((x, y))
                    tent_available[x][y] = False
                    self._grid[x][y] = "T"

            # Place tents
            self._tent_positions = set()
            trying_times = 0
            while len(self._tent_positions) < self._num_trees:
                x, y = random.choice(list(self._tree_positions))
                potential_positions = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
                random.shuffle(potential_positions)
                for tx, ty in potential_positions:
                    if 0 <= ty < width and 0 <= tx < height and tent_available[tx][ty]:
                        self._grid[tx][ty] = "X"
                        tent_available[tx][ty] = False
                        self._tent_positions.add((tx, ty))
                        # Mark surrounding cells as unavailable
                        near_positions = [
                            (tx + 1, ty),
                            (tx + 1, ty + 1),
                            (tx + 1, ty - 1),
                            (tx, ty + 1),
                            (tx, ty - 1),
                            (tx - 1, ty + 1),
                            (tx - 1, ty),
                            (tx - 1, ty - 1),
                        ]
                        for nx, ny in near_positions:
                            if 0 <= ny < width and 0 <= nx < height:
                                tent_available[nx][ny] = False
                        break
                trying_times += 1
                if trying_times > 10000:
                    break
            if trying_times <= 10000:
                break

        # Calculate row and column tent counts
        self._row_tent_counts = [0] * height
        self._col_tent_counts = [0] * width
        for tx, ty in self._tent_positions:
            self._row_tent_counts[tx] += 1
            self._col_tent_counts[ty] += 1

        # Sort positions
        self._tree_positions = set(sorted(list(self._tree_positions)))
        self._tent_positions = set(sorted(list(self._tent_positions)))

        # Randomly remove some tents
        max_remove = max(2, len(self._tent_positions) // 3)
        num_to_remove = random.randint(1, max_remove)
        removed_tents = random.sample(list(self._tent_positions), num_to_remove)
        for tx, ty in removed_tents:
            self._grid[tx][ty] = ""
            self._tent_positions.remove((tx, ty))

    def _generate_num_tents_in_row_question(self) -> dict[str, Any]:
        """Generate question about number of tents in a row."""
        height = self._grid_size[1]
        row_to_ask = random.randint(0, height - 1)

        # Count tents in the row
        tents_in_row = [(tx, ty) for tx, ty in self._tent_positions if tx == row_to_ask]
        answer = len(tents_in_row)

        question = f"In the current state, only some of the correct positions of the tents are marked in the grid. Given the current state, how many tents are there in row {row_to_ask} currently?"

        return {
            "question": question,
            "answer": str(answer),
        }

    def _generate_missing_tents_in_column_question(self) -> dict[str, Any]:
        """Generate question about missing tents in a column."""
        width = self._grid_size[0]
        col_to_ask = random.randint(0, width - 1)

        # Count current tents in the column
        current_tents = len(
            [(tx, ty) for tx, ty in self._tent_positions if ty == col_to_ask]
        )
        answer = self._col_tent_counts[col_to_ask] - current_tents

        question = f"In the current state, only some of the correct positions of the tents are marked in the grid. Given the current state, how many tents are still missing in column {col_to_ask}?"

        return {
            "question": question,
            "answer": str(answer),
        }

    def _generate_missing_tents_in_grid_question(self) -> dict[str, Any]:
        """Generate question about total missing tents in grid."""
        total_needed = sum(self._col_tent_counts)
        current_placed = len(self._tent_positions)
        answer = total_needed - current_placed

        question = "In the current state, only some of the correct positions of the tents are marked in the grid. Given the current state, how many tents are still missing in the entire grid?"

        return {
            "question": question,
            "answer": str(answer),
        }

    def _generate_possible_tent_positions_question(self) -> dict[str, Any]:
        """Generate question about possible tent positions (considering only tree adjacency)."""
        width, height = self._grid_size
        possible_positions = set()

        for x, y in self._tree_positions:
            # Check adjacent cells
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if (
                    0 <= ny < width
                    and 0 <= nx < height
                    and (nx, ny) not in self._tree_positions
                ):
                    possible_positions.add((nx, ny))

        answer = len(possible_positions)

        question = "In the current state, only some of the correct positions of the tents are marked in the grid. Given the tree positions and considering only the first and the third rule, how many positions in the entire grid are available to place tents (including both positions that are currently occupied by tents and positions that are currently empty)?"

        return {
            "question": question,
            "answer": str(answer),
        }

    def _generate_new_tent_count_question(self) -> dict[str, Any]:
        """Generate question about valid positions for new tents."""
        width, height = self._grid_size
        valid_positions = []

        for x in range(height):
            for y in range(width):
                if self._is_valid_new_tent_position(x, y):
                    valid_positions.append((x, y))

        answer = len(valid_positions)

        question = "In the current state, only some of the correct positions of the tents are marked in the grid. Given the current state, how many positions in the grid are available to place a new tent without breaking the game rules immediately (it does not have to be a part of a whole solution to the puzzle)?"

        return {
            "question": question,
            "answer": str(answer),
        }

    def _generate_tree_position_question(self) -> dict[str, Any]:
        """Generate multiple choice question about tree position."""
        width, height = self._grid_size
        num_options = 8

        # Select correct answer
        correct_option = random.choice(list(self._tree_positions))

        # Generate wrong options
        options = [correct_option]
        while len(options) < num_options:
            random_position = (
                random.randint(0, height - 1),
                random.randint(0, width - 1),
            )
            if (
                random_position not in options
                and random_position not in self._tree_positions
            ):
                options.append(random_position)

        random.shuffle(options)

        # Find correct answer index
        correct_answer = options.index(correct_option) + 1

        # Generate option text
        options_str = "\n".join(
            [f"{i+1}: ({x}, {y})" for i, (x, y) in enumerate(options)]
        )

        question = "In the current state, only some of the correct positions of the tents are marked in the grid. Given the current state, which of the following positions contains a tree?"

        return {
            "question": question,
            "answer": str(correct_answer),
            "options": [f"({x}, {y})" for x, y in options],
        }

    def _generate_new_tent_position_question(self) -> dict[str, Any]:
        """Generate multiple choice question about valid position for new tent."""
        width, height = self._grid_size
        num_options = 8

        # Find all valid positions
        valid_positions = []
        for x in range(height):
            for y in range(width):
                if self._is_valid_new_tent_position(x, y):
                    valid_positions.append((x, y))

        if not valid_positions:
            # If no valid positions, create a dummy question
            correct_option = (0, 0)
        else:
            correct_option = random.choice(valid_positions)

        # Generate wrong options
        options = [correct_option]
        while len(options) < num_options:
            random_position = (
                random.randint(0, height - 1),
                random.randint(0, width - 1),
            )
            if random_position not in options and not self._is_valid_new_tent_position(
                random_position[0], random_position[1]
            ):
                options.append(random_position)

        random.shuffle(options)

        # Find correct answer index
        correct_answer = options.index(correct_option) + 1

        # Generate option text
        options_str = "\n".join(
            [f"{i+1}: ({x}, {y})" for i, (x, y) in enumerate(options)]
        )

        question = "In the current state, only some of the correct positions of the tents are marked in the grid. Given the current state, which of the following positions is allowed to place a new tent without breaking the game rules immediately (it does not have to be a part of a whole solution to the puzzle)?"

        return {
            "question": question,
            "answer": str(correct_answer),
            "options": [f"({x}, {y})" for x, y in options],
        }

    def _is_valid_new_tent_position(self, x: int, y: int) -> bool:
        """Check if a position is valid for placing a new tent."""
        width, height = self._grid_size

        # Position already has tree
        if (x, y) in self._tree_positions:
            return False

        # Position already has tent
        if (x, y) in self._tent_positions:
            return False

        # Check if adjacent to tree
        adjacent_tree_found = False
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if (
                0 <= ny < width
                and 0 <= nx < height
                and (nx, ny) in self._tree_positions
            ):
                adjacent_tree_found = True
                break

        if not adjacent_tree_found:
            return False

        # Check if adjacent to tent
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if (nx, ny) in self._tent_positions:
                    return False

        # Check row constraint
        current_row_tents = sum(1 for tx, ty in self._tent_positions if tx == x)
        if self._row_tent_counts[x] <= current_row_tents:
            return False

        # Check column constraint
        current_col_tents = sum(1 for tx, ty in self._tent_positions if ty == y)
        if self._col_tent_counts[y] <= current_col_tents:
            return False

        return True

    def _check_answer(self, action: str) -> bool:
        """Check if the provided answer is correct."""
        action_normalized = action.strip().lower()
        correct_normalized = self._oracle_answer.strip().lower()
        return action_normalized == correct_normalized

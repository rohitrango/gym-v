"""Space Invaders Q&A environment based on GameRL.

Single-turn Q&A environment where the model answers questions about a Space Invaders game state.
"""

from __future__ import annotations

from importlib import resources
import random
import re
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation, get_logger

logger = get_logger()


# Question types from original Game-RL
# Removed module-level QUESTION_TYPES - now defined as class variable
GAME_RULES = dedent("""
    The given image represents a simplified interface of the game Space Invaders. The enemy area is implicitly divided into a grid of cells, with the row and column numbers shown on the left and top sides of the grid respectively which you should strictly follow. Each cell is either empty or occupied by an incoming enemy which can be purple, blue or green. The ship is at the bottom row, aligned with one of the columns, which shoots the enemies using laser while dodging possible lasers from the enemies.

    If the ship shoots, the enemy closest to the ship (i.e. the lowermost one) on the same column as the ship, if any, will be destroyed and disappear, adding points to the player's score and exposing the enemy behind (if any). Purple enemies are worth 30 points, blue enemies are worth 20 points, and green enemies are worth 10 points.

    The enemies keep on uniformly moving in a certain direction (left or right). Carefully understand the time sequence rules below.
    - Consider the consecutive time intervals, denoted by t, t+1, t+2, ...
    - During each time interval t:
      - The ship can shoot at most once.
      - The ship can move to another column before shooting.
      - The enemies keep still.
    - At the very end of this time interval t, the enemies move one step in the direction they are moving, thus changing the columns they are on.
""").strip()

ANSWER_FORMAT_PROMPT = dedent("""
    **Answer Format:**
    - For numbers: Reply with only the number

    Do not include any explanation or extra text.
""").strip()


class Enemy:
    """Class representing an enemy in Space Invaders."""

    def __init__(self, row: int, col: int, enemy_type: int):
        self.row = row
        self.col = col
        self.type = enemy_type
        self.score = {1: 30, 2: 20, 3: 10}[enemy_type]
        self.color = {1: "purple", 2: "blue", 3: "green"}[enemy_type]


class GameRLSpaceInvadersQAEnv(Env):
    """Space Invaders Q&A environment.

    Single-turn Q&A environment based on the original Game-RL Space Invaders game.
    Given a game state image, answer questions about enemy counts, positions,
    or shooting strategies.

    Args:
        question_type: Question type ID (0-6). None for random selection.
        enemy_rows: Number of enemy rows (3-5, default 4)
        enemy_cols: Number of enemy columns (default 6)
        enemy_area_rows: Total rows in enemy area (default 8)
        cell_width: Width of each cell in pixels (default 50)
        cell_height: Height of each cell in pixels (default 40)
    """

    # Question types
    QUESTION_TYPES = [
        {
            "id": "type_0",
            "name": "enemies_on_row",
            "level": "Easy",
            "answer_format": "fill_in_blank",
            "qa_type": "State Prediction",
        },
        {
            "id": "type_1",
            "name": "enemies_on_col",
            "level": "Easy",
            "answer_format": "fill_in_blank",
            "qa_type": "State Prediction",
        },
        {
            "id": "type_2",
            "name": "total_enemies",
            "level": "Medium",
            "answer_format": "fill_in_blank",
            "qa_type": "State Prediction",
        },
        {
            "id": "type_3",
            "name": "colored_enemies",
            "level": "Medium",
            "answer_format": "fill_in_blank",
            "qa_type": "State Prediction",
        },
        {
            "id": "type_4",
            "name": "shoot_here_points",
            "level": "Easy",
            "answer_format": "fill_in_blank",
            "qa_type": "State Prediction",
        },
        {
            "id": "type_5",
            "name": "move_and_shoot_points",
            "level": "Easy",
            "answer_format": "fill_in_blank",
            "qa_type": "State Prediction",
        },
        {
            "id": "type_6",
            "name": "max_shoot_once_points",
            "level": "Medium",
            "answer_format": "fill_in_blank",
            "qa_type": "State Prediction",
        },
    ]

    gamerl_assets_dir = resources.files("gym_v.envs.gamerl") / "assets"
    assets_dir = resources.files("gym_v.envs") / "assets"

    COLORS = {
        "background": (0, 0, 0),
        "grid_line": (255, 255, 255),
        "text": (255, 255, 255),
        "ship": (255, 165, 0),
        "purple": (128, 0, 128),
        "blue": (0, 191, 255),
        "green": (0, 255, 0),
    }

    def __init__(
        self,
        question_type: int | None = None,
        enemy_rows: int = 4,
        enemy_cols: int = 6,
        enemy_area_rows: int = 8,
        cell_width: int = 50,
        cell_height: int = 40,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert enemy_rows in [3, 4, 5], "enemy_rows should be 3, 4 or 5"

        self._question_type = question_type
        self._enemy_rows = enemy_rows
        self._enemy_cols = enemy_cols
        self._enemy_area_rows = enemy_area_rows
        self._total_cols = enemy_cols + 4
        self._cell_width = cell_width
        self._cell_height = cell_height
        self._top_border = 40
        self._left_border = 30

        # Game state
        self._enemies: list[Enemy] = []
        self._ship_col: int = 1
        self._score: int = 0

        # Q&A state
        self._current_question_type: int = 0
        self._current_question: str = ""
        self._oracle_answer: str = ""
        self._target_row: int = 0
        self._target_col: int = 0
        self._target_color: str = ""

        # Load images
        self._ship_image = self._load_image(
            "ship_orange.png", cell_width - 10, cell_height - 5
        )
        self._enemy_images = {
            1: self._load_image("enemy1_1.png", cell_width - 10, cell_height - 5),
            2: self._load_image("enemy2_1.png", cell_width - 10, cell_height - 5),
            3: self._load_image("enemy3_1.png", cell_width - 10, cell_height - 5),
        }

    def _load_image(self, filename: str, width: int, height: int) -> Image.Image | None:
        """Load and scale an image from assets."""
        image_path = self.gamerl_assets_dir / filename
        try:
            if image_path.is_file():
                img = Image.open(str(image_path))
                return img.resize((width, height), Image.Resampling.LANCZOS)
        except Exception as e:
            logger.warning(f"Error loading image {filename}: {e}")
        return None

    @property
    def description(self) -> str:
        """Return game rules + current question + answer format."""
        desc = GAME_RULES + "\n\n**Question:** " + self._current_question
        desc += ANSWER_FORMAT_PROMPT
        return desc.strip()

    def _get_state_text(self) -> str:
        """Generate text description of current Space Invaders game state.

        Returns a grid representation matching the rendered image.
        """
        # Create grid for enemy area (1-indexed as per game rules)
        grid = [
            ["." for _ in range(self._total_cols)] for _ in range(self._enemy_area_rows)
        ]

        # Mark enemies
        for enemy in self._enemies:
            row_idx = enemy.row - 1  # Convert to 0-indexed
            col_idx = enemy.col - 1  # Convert to 0-indexed
            if 0 <= row_idx < self._enemy_area_rows and 0 <= col_idx < self._total_cols:
                if enemy.type == 1:
                    grid[row_idx][col_idx] = "P"  # Purple
                elif enemy.type == 2:
                    grid[row_idx][col_idx] = "B"  # Blue
                elif enemy.type == 3:
                    grid[row_idx][col_idx] = "G"  # Green

        grid_str = "\n".join(["".join(row) for row in grid])

        # Add ship position below the grid
        ship_row = ["."] * self._total_cols
        ship_row[self._ship_col - 1] = "S"
        ship_str = "".join(ship_row)

        return f"""Grid Size: {self._enemy_area_rows}x{self._total_cols}
Ship Position: Column {self._ship_col}
Grid (P=purple enemy, B=blue enemy, G=green enemy, .=empty):
{grid_str}
Ship Row: {ship_str}"""

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)

        if seed is not None:
            random.seed(seed)

        self._score = 0

        # Generate enemies
        self._generate_enemies()

        # Random ship position
        enemy_cols = list(set(e.col for e in self._enemies))
        if enemy_cols:
            self._ship_col = enemy_cols[self.np_random.integers(0, len(enemy_cols))]
        else:
            self._ship_col = self._total_cols // 2

        # Select question type
        if self._question_type is not None:
            self._current_question_type = self._question_type
        else:
            self._current_question_type = self.np_random.integers(
                0, len(self.QUESTION_TYPES)
            )

        # Generate Q&A
        self._generate_qa()

        logger.info(f"Reset Space Invaders QA (type={self._current_question_type}).")

        obs = Observation(
            image=self.render(),
            text=self._get_state_text(),
            metadata={
                "question": self._current_question,
                "options": self._options,
                "question_type": self.QUESTION_TYPES[self._current_question_type][
                    "name"
                ],
                "level": self.QUESTION_TYPES[self._current_question_type]["level"],
            },
        )
        info = {
            "oracle_answer": self._oracle_answer,
            "question_type": self._current_question_type,
        }
        return obs, info

    def inner_step(
        self, action: str
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        """Evaluate the answer. Always terminates after one step."""
        answer = action.strip()
        reward = self._score_answer(answer)

        obs = Observation(
            image=self.render(),
            text=None,
            metadata={
                "question_type": self.QUESTION_TYPES[self._current_question_type][
                    "name"
                ],
            },
        )
        info = {
            "oracle_answer": self._oracle_answer,
            "user_answer": answer,
            "correct": reward == 1.0,
        }

        return obs, reward, True, False, info

    def _score_answer(self, answer: str) -> float:
        """Score the answer."""
        match = re.search(r"-?\d+", answer)
        if match:
            try:
                return 1.0 if int(match.group()) == int(self._oracle_answer) else 0.0
            except ValueError:
                return 0.0
        return 0.0

    def _generate_enemies(self) -> None:
        """Generate enemies."""
        self._enemies = []
        start_row = self.np_random.integers(
            1, self._enemy_area_rows - self._enemy_rows + 1
        )
        start_col = self.np_random.integers(1, self._total_cols - self._enemy_cols + 1)

        for col in range(start_col, start_col + self._enemy_cols):
            height = self.np_random.integers(1, self._enemy_rows + 1)
            for row in range(start_row, start_row + height):
                if row == start_row:
                    enemy_type = 1  # Purple
                elif row <= start_row + 1:
                    enemy_type = 2  # Blue
                else:
                    enemy_type = 3  # Green
                self._enemies.append(Enemy(row, col, enemy_type))

    def _generate_qa(self) -> None:
        """Generate question and oracle answer."""
        q_type = self._current_question_type

        if q_type == 0:
            self._generate_q0_enemies_on_row()
        elif q_type == 1:
            self._generate_q1_enemies_on_col()
        elif q_type == 2:
            self._generate_q2_total_enemies()
        elif q_type == 3:
            self._generate_q3_colored_enemies()
        elif q_type == 4:
            self._generate_q4_shoot_here_points()
        elif q_type == 5:
            self._generate_q5_move_and_shoot_points()
        elif q_type == 6:
            self._generate_q6_max_shoot_once_points()

    def _get_enemies_on_row(self, row: int) -> list[Enemy]:
        """Get all enemies on a specific row."""
        return [e for e in self._enemies if e.row == row]

    def _get_enemies_on_col(self, col: int) -> list[Enemy]:
        """Get all enemies on a specific column, sorted by row (bottom first)."""
        enemies = [e for e in self._enemies if e.col == col]
        return sorted(enemies, key=lambda e: e.row, reverse=True)

    def _generate_q0_enemies_on_row(self) -> None:
        """Q0: How many enemies are on a specific row?"""
        rows_with_enemies = list(set(e.row for e in self._enemies))
        if rows_with_enemies:
            self._target_row = rows_with_enemies[
                self.np_random.integers(0, len(rows_with_enemies))
            ]
        else:
            self._target_row = self.np_random.integers(1, self._enemy_area_rows + 1)

        count = len(self._get_enemies_on_row(self._target_row))
        self._current_question = f"How many enemies are on row {self._target_row}?"
        self._oracle_answer = str(count)
        self._options = []

    def _generate_q1_enemies_on_col(self) -> None:
        """Q1: How many enemies are on a specific column?"""
        cols_with_enemies = list(set(e.col for e in self._enemies))
        if cols_with_enemies:
            self._target_col = cols_with_enemies[
                self.np_random.integers(0, len(cols_with_enemies))
            ]
        else:
            self._target_col = self.np_random.integers(1, self._total_cols + 1)

        count = len(self._get_enemies_on_col(self._target_col))
        self._current_question = f"How many enemies are on column {self._target_col}?"
        self._oracle_answer = str(count)
        self._options = []

    def _generate_q2_total_enemies(self) -> None:
        """Q2: How many enemies are there in total?"""
        self._current_question = "How many enemies are there in total?"
        self._oracle_answer = str(len(self._enemies))
        self._options = []

    def _generate_q3_colored_enemies(self) -> None:
        """Q3: How many enemies of a specific color are there?"""
        colors = ["purple", "blue", "green"]
        color_to_type = {"purple": 1, "blue": 2, "green": 3}
        self._target_color = colors[self.np_random.integers(0, 3)]

        count = len(
            [e for e in self._enemies if e.type == color_to_type[self._target_color]]
        )
        self._current_question = (
            f"How many {self._target_color} enemies are there in total?"
        )
        self._oracle_answer = str(count)
        self._options = []

    def _generate_q4_shoot_here_points(self) -> None:
        """Q4: If the ship shoots at the current position, how many points?"""
        self._current_question = "If the ship shoots at the current position, how many points will the player get?"

        enemies_on_col = self._get_enemies_on_col(self._ship_col)
        if enemies_on_col:
            self._oracle_answer = str(enemies_on_col[0].score)
        else:
            self._oracle_answer = "0"
        self._options = []

    def _generate_q5_move_and_shoot_points(self) -> None:
        """Q5: If the ship moves to a column and shoots, how many points?"""
        # Select a random column (preferring columns with enemies)
        cols_with_enemies = list(set(e.col for e in self._enemies))
        if cols_with_enemies and self.np_random.random() < 0.7:
            self._target_col = cols_with_enemies[
                self.np_random.integers(0, len(cols_with_enemies))
            ]
        else:
            self._target_col = self.np_random.integers(1, self._total_cols + 1)

        self._current_question = f"Suppose that all the enemies keep still. If the ship moves to column {self._target_col} and shoots, how many points will the player get?"

        enemies_on_col = self._get_enemies_on_col(self._target_col)
        if enemies_on_col:
            self._oracle_answer = str(enemies_on_col[0].score)
        else:
            self._oracle_answer = "0"
        self._options = []

    def _generate_q6_max_shoot_once_points(self) -> None:
        """Q6: What is the maximum points from shooting once?"""
        self._current_question = "Given that the image depicts the scene at the beginning of time interval t, during which the enemies keep still. What is the maximum number of points the player can get if he can move the ship to any column and let the ship shoot?"

        max_score = 0
        for col in range(1, self._total_cols + 1):
            enemies_on_col = self._get_enemies_on_col(col)
            if enemies_on_col:
                max_score = max(max_score, enemies_on_col[0].score)

        self._oracle_answer = str(max_score)
        self._options = []

    def render(self) -> Image.Image:
        """Render the game state."""
        width = self._left_border + self._total_cols * self._cell_width
        height = self._top_border + (self._enemy_area_rows + 1) * self._cell_height

        img = Image.new("RGB", (width, height), self.COLORS["background"])
        draw = ImageDraw.Draw(img)

        font_path = self.assets_dir / "DejaVuSans.ttf"
        if font_path.exists():
            font = ImageFont.truetype(str(font_path), 16)
        else:
            font = ImageFont.load_default()

        # Grid lines
        for col in range(1, self._total_cols + 2):
            x = self._left_border + (col - 1) * self._cell_width
            draw.line(
                [(x, self._top_border), (x, height)],
                fill=self.COLORS["grid_line"],
                width=1,
            )
        for row in range(1, self._enemy_area_rows + 3):
            y = self._top_border + (row - 1) * self._cell_height
            draw.line([(0, y), (width, y)], fill=self.COLORS["grid_line"], width=1)

        # Column numbers
        for col in range(1, self._total_cols + 1):
            x = self._left_border + (col - 1) * self._cell_width + self._cell_width // 2
            draw.text(
                (x, 10), str(col), fill=self.COLORS["text"], font=font, anchor="mt"
            )

        # Row numbers
        for row in range(1, self._enemy_area_rows + 2):
            y = (
                self._top_border
                + (row - 1) * self._cell_height
                + self._cell_height // 2
            )
            draw.text(
                (10, y), str(row), fill=self.COLORS["text"], font=font, anchor="mm"
            )

        # Enemies
        for enemy in self._enemies:
            x = self._left_border + (enemy.col - 1) * self._cell_width + 5
            y = self._top_border + (enemy.row - 1) * self._cell_height + 2

            enemy_img = self._enemy_images.get(enemy.type)
            if enemy_img:
                img.paste(enemy_img, (x, y))
            else:
                x2 = x + self._cell_width - 10
                y2 = y + self._cell_height - 5
                draw.rectangle([x, y, x2, y2], fill=self.COLORS[enemy.color])

        # Ship
        ship_x = self._left_border + (self._ship_col - 1) * self._cell_width + 5
        ship_y = self._top_border + self._enemy_area_rows * self._cell_height + 2
        if self._ship_image:
            img.paste(self._ship_image, (ship_x, ship_y))
        else:
            draw.rectangle(
                [
                    ship_x,
                    ship_y,
                    ship_x + self._cell_width - 10,
                    ship_y + self._cell_height - 5,
                ],
                fill=self.COLORS["ship"],
            )

        # Score
        draw.text(
            (width - 100, 10),
            f"Score: {self._score}",
            fill=self.COLORS["text"],
            font=font,
        )

        return img

"""Space Invaders game based on GameRL."""

from __future__ import annotations

from importlib import resources
import random
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation, get_logger

logger = get_logger()


class Enemy:
    """Class representing an enemy in Space Invaders."""

    def __init__(self, row: int, col: int, enemy_type: int):
        self.row = row
        self.col = col
        self.type = enemy_type
        self.score = {1: 30, 2: 20, 3: 10}[enemy_type]
        self.color = {1: "purple", 2: "blue", 3: "green"}[enemy_type]


class GameRLSpaceInvadersEnv(Env):
    """Space Invaders game environment.

    Control a ship to shoot down enemies while avoiding being hit.

    Args:
        enemy_rows: Number of enemy rows (3-5, default 4)
        enemy_cols: Number of enemy columns (default 6)
        enemy_area_rows: Total rows in enemy area (default 8)
        cell_width: Width of each cell in pixels (default 50)
        cell_height: Height of each cell in pixels (default 40)
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    # Colors
    COLORS = {
        "background": (0, 0, 0),  # Black
        "grid_line": (255, 255, 255),  # White
        "text": (255, 255, 255),  # White
        "ship": (255, 165, 0),  # Orange
        "purple": (128, 0, 128),  # Purple enemy
        "blue": (0, 191, 255),  # Blue enemy
        "green": (0, 255, 0),  # Green enemy
        "laser": (255, 0, 0),  # Red laser
    }

    def __init__(
        self,
        enemy_rows: int = 4,
        enemy_cols: int = 6,
        enemy_area_rows: int = 8,
        cell_width: int = 50,
        cell_height: int = 40,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert enemy_rows in [3, 4, 5], "enemy_rows should be 3, 4 or 5"

        self._enemy_rows = enemy_rows
        self._enemy_cols = enemy_cols
        self._enemy_area_rows = enemy_area_rows
        self._total_cols = enemy_cols + 4  # Add padding columns
        self._cell_width = cell_width
        self._cell_height = cell_height
        self._top_border = 40
        self._left_border = 30

        # Game state (initialized in reset)
        self._enemies: list[Enemy] = []
        self._ship_col: int = 1
        self._score: int = 0
        self._game_over: bool = False

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
        image_path = self.assets_dir / filename
        try:
            if image_path.is_file():
                img = Image.open(str(image_path))
                return img.resize((width, height), Image.Resampling.LANCZOS)
        except Exception as e:
            logger.warning(f"Error loading image {filename}: {e}")
        return None

    @property
    def description(self) -> str:
        return dedent("""
            The given image represents a simplified interface of the game Space Invaders. The enemy area is implicitly divided into a grid of cells, with the row and column numbers shown on the left and top sides of the grid respectively which you should strictly follow. Each cell is either empty or occupied by an incoming enemy which can be purple, blue or green. The ship is at the bottom row, aligned with one of the columns, which shoots the enemies using laser while dodging possible lasers from the enemies.

            If the ship shoots, the enemy closest to the ship (i.e. the lowermost one) on the same column as the ship, if any, will be destroyed and disappear, adding points to the player's score and exposing the enemy behind (if any). Purple enemies are worth 30 points, blue enemies are worth 20 points, and green enemies are worth 10 points.

            The enemies keep on uniformly moving in a certain direction (left or right). Carefully understand the time sequence rules below.
            - Consider the consecutive time intervals, denoted by t, t+1, t+2, ...
            - During each time interval t:
              - The ship can shoot at most once.
              - The ship can move to another column before shooting.
              - The enemies keep still.
            - At the very end of this time interval t, the enemies move one step in the direction they are moving, thus changing the columns they are on.

            Available Actions:
            - left: Move ship left one column
            - right: Move ship right one column
            - shoot: Shoot at the current column (destroys lowermost enemy)
            - 1-N: Move ship to column N and shoot
        """).strip()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)

        if seed is not None:
            random.seed(seed)

        self._score = 0
        self._game_over = False

        # Generate enemies
        self._enemies = []
        start_row = random.randint(1, self._enemy_area_rows - self._enemy_rows + 1)
        start_col = random.randint(1, self._total_cols - self._enemy_cols + 1)

        for col in range(start_col, start_col + self._enemy_cols):
            # Random height for this column
            height = random.randint(1, self._enemy_rows)
            for row in range(start_row, start_row + height):
                # Determine enemy type based on row
                if row == start_row:
                    enemy_type = 1  # Purple (top)
                elif row <= start_row + 1:
                    enemy_type = 2  # Blue (middle)
                else:
                    enemy_type = 3  # Green (bottom)
                self._enemies.append(Enemy(row, col, enemy_type))

        # Random ship position
        enemy_cols = list(set(e.col for e in self._enemies))
        if enemy_cols:
            self._ship_col = random.choice(enemy_cols)
        else:
            self._ship_col = self._total_cols // 2

        logger.info("Reset Space Invaders game.")

        obs = Observation(image=self.render(), text=self._get_observation_text())
        return obs, {}

    def inner_step(
        self, action: str
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        info: dict[str, Any] = {}
        reward = 0.0
        terminated = False
        truncated = False

        if self._game_over:
            obs = Observation(image=self.render(), text=self._get_observation_text())
            return obs, reward, True, truncated, info

        action_lower = action.lower().strip()

        # Check if action is a column number
        try:
            col_num = int(action_lower)
            if 1 <= col_num <= self._total_cols:
                self._ship_col = col_num
                reward = self._shoot()
                info["action"] = f"move_and_shoot_{col_num}"
            else:
                info["invalid_action"] = True
        except ValueError:
            if action_lower in ["left", "a"]:
                if self._ship_col > 1:
                    self._ship_col -= 1
                info["action"] = "left"
            elif action_lower in ["right", "d"]:
                if self._ship_col < self._total_cols:
                    self._ship_col += 1
                info["action"] = "right"
            elif action_lower in ["shoot", "s", "space"]:
                reward = self._shoot()
                info["action"] = "shoot"
            else:
                info["invalid_action"] = True

        # Check win condition
        if not self._enemies:
            self._game_over = True
            terminated = True
            info["win"] = True

        info["score"] = self._score

        obs = Observation(image=self.render(), text=self._get_observation_text())
        return obs, reward, terminated, truncated, info

    def _shoot(self) -> float:
        """Shoot at current column. Returns points scored."""
        # Find lowermost enemy in current column
        enemies_in_col = [e for e in self._enemies if e.col == self._ship_col]
        if not enemies_in_col:
            return 0.0

        # Get the lowermost enemy (highest row number)
        target = max(enemies_in_col, key=lambda e: e.row)
        self._enemies.remove(target)
        self._score += target.score
        return float(target.score)

    def render(self) -> Image.Image:
        """Render the game state as a PIL Image."""
        width = self._left_border + self._total_cols * self._cell_width
        height = self._top_border + (self._enemy_area_rows + 1) * self._cell_height

        img = Image.new("RGB", (width, height), self.COLORS["background"])
        draw = ImageDraw.Draw(img)

        # Try to load font
        font_path = self.assets_dir / "DejaVuSans.ttf"
        if font_path.exists():
            font = ImageFont.truetype(str(font_path), 16)
        else:
            font = ImageFont.load_default()

        # Draw grid lines
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

        # Draw column numbers (top)
        for col in range(1, self._total_cols + 1):
            x = self._left_border + (col - 1) * self._cell_width + self._cell_width // 2
            draw.text(
                (x, 10), str(col), fill=self.COLORS["text"], font=font, anchor="mt"
            )

        # Draw row numbers (left)
        for row in range(1, self._enemy_area_rows + 2):
            y = (
                self._top_border
                + (row - 1) * self._cell_height
                + self._cell_height // 2
            )
            draw.text(
                (10, y), str(row), fill=self.COLORS["text"], font=font, anchor="mm"
            )

        # Draw enemies
        for enemy in self._enemies:
            x = self._left_border + (enemy.col - 1) * self._cell_width + 5
            y = self._top_border + (enemy.row - 1) * self._cell_height + 2

            enemy_img = self._enemy_images.get(enemy.type)
            if enemy_img:
                img.paste(enemy_img, (x, y))
            else:
                # Fallback: draw colored rectangle
                x2 = x + self._cell_width - 10
                y2 = y + self._cell_height - 5
                draw.rectangle([x, y, x2, y2], fill=self.COLORS[enemy.color])

        # Draw ship
        ship_x = self._left_border + (self._ship_col - 1) * self._cell_width + 5
        ship_y = self._top_border + self._enemy_area_rows * self._cell_height + 2
        if self._ship_image:
            img.paste(self._ship_image, (ship_x, ship_y))
        else:
            # Fallback: draw orange rectangle
            draw.rectangle(
                [
                    ship_x,
                    ship_y,
                    ship_x + self._cell_width - 10,
                    ship_y + self._cell_height - 5,
                ],
                fill=self.COLORS["ship"],
            )

        # Draw score
        score_text = f"Score: {self._score}"
        draw.text((width - 100, 10), score_text, fill=self.COLORS["text"], font=font)

        return img

    def _get_observation_text(self) -> str:
        """Get text description of current game state."""
        if self._game_over:
            if not self._enemies:
                return f"You Win! Final Score: {self._score}"
            return f"Game Over! Final Score: {self._score}"

        enemy_count = len(self._enemies)
        return f"Ship at column {self._ship_col}, Enemies: {enemy_count}, Score: {self._score}"

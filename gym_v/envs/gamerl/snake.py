"""Snake game environment for gym-v."""

from __future__ import annotations

from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation, get_logger

logger = get_logger()


class SnakeEnv(Env):
    """Snake game environment.

    Control a snake to eat food and grow longer without hitting walls or itself.

    Args:
        width: Grid width (default 10)
        height: Grid height (default 10)
        initial_snake_length: Initial snake length (default 3)
        cell_size: Size of each cell in pixels for rendering (default 40)
    """

    # Direction mappings
    DIRECTIONS = {
        "up": (-1, 0),
        "w": (-1, 0),
        "down": (1, 0),
        "s": (1, 0),
        "left": (0, -1),
        "a": (0, -1),
        "right": (0, 1),
        "d": (0, 1),
    }

    # Colors (matching Game-RL)
    COLORS = {
        "background": (255, 255, 255),
        "grid": (200, 200, 200),
        "snake_head": (255, 255, 0),  # Yellow
        "snake_body": (0, 0, 255),  # Blue
        "food": (255, 0, 0),  # Red
        "text": (0, 0, 0),  # Black
        "line": (0, 0, 0),  # Black for connecting lines
    }

    def __init__(
        self,
        width: int = 10,
        height: int = 10,
        initial_snake_length: int = 3,
        cell_size: int = 40,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._width = width
        self._height = height
        self._initial_snake_length = min(initial_snake_length, width * height - 1)
        self._cell_size = cell_size
        self._margin = 30  # Margin for coordinate labels

        # Game state (initialized in reset)
        self._snake: list[tuple[int, int]] = []
        self._food: tuple[int, int] | None = None
        self._game_over: bool = False

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def description(self) -> str:
        return dedent("""
            Snake Game: Control the snake to eat food and grow longer.

            Available actions: up, down, left, right (or w, a, s, d)

            Rules:
            - The snake moves in the direction you choose
            - Eating food (red) makes the snake grow longer
            - Game ends if you hit a wall or the snake's own body
        """).strip()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)

        self._game_over = False
        self._generate_snake()
        self._generate_food()

        logger.info("Reset Snake game.")

        obs = Observation(image=self.render(), text=self._get_observation_text())
        return obs, {}

    def inner_step(
        self, action: str
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        info: dict[str, Any] = {}
        reward = 0.0
        terminated = False
        truncated = False

        # Normalize action
        action_lower = action.lower().strip()

        # Check for invalid action
        if action_lower not in self.DIRECTIONS:
            info["invalid_action"] = True
            obs = Observation(image=self.render(), text=self._get_observation_text())
            return obs, reward, terminated, truncated, info

        info["invalid_action"] = False

        # Get direction
        dr, dc = self.DIRECTIONS[action_lower]
        head_row, head_col = self._snake[0]
        new_head = (head_row + dr, head_col + dc)

        # Check for collision with walls
        if (
            new_head[0] < 0
            or new_head[0] >= self._height
            or new_head[1] < 0
            or new_head[1] >= self._width
        ):
            self._game_over = True
            terminated = True
            info["death_reason"] = "wall"
            obs = Observation(image=self.render(), text=self._get_observation_text())
            return obs, reward, terminated, truncated, info

        # Check for collision with self (excluding tail which will move)
        if new_head in self._snake[:-1]:
            self._game_over = True
            terminated = True
            info["death_reason"] = "self"
            obs = Observation(image=self.render(), text=self._get_observation_text())
            return obs, reward, terminated, truncated, info

        # Check if eating food
        ate_food = new_head == self._food

        # Move snake
        if ate_food:
            # Grow: keep tail, add new head
            self._snake = [new_head] + self._snake
            reward = 1.0
            info["ate_food"] = True
            # Generate new food
            self._generate_food()
        else:
            # Move: remove tail, add new head
            self._snake = [new_head] + self._snake[:-1]
            info["ate_food"] = False

        info["snake_length"] = len(self._snake)

        obs = Observation(image=self.render(), text=self._get_observation_text())
        return obs, reward, terminated, truncated, info

    def render(self) -> Image.Image:
        """Render the game state as a PIL Image."""
        img_width = self._width * self._cell_size + 2 * self._margin
        img_height = self._height * self._cell_size + 2 * self._margin

        img = Image.new("RGB", (img_width, img_height), self.COLORS["grid"])
        draw = ImageDraw.Draw(img)

        # Try to load a font, fall back to default if not available
        try:
            font_size = self._cell_size // 2
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size
            )
        except OSError:
            font = ImageFont.load_default()

        # Draw coordinate labels
        # Row numbers (left side)
        for row in range(self._height):
            text = str(row)
            y = self._margin + row * self._cell_size + self._cell_size // 2
            draw.text(
                (self._margin // 2, y),
                text,
                fill=self.COLORS["text"],
                font=font,
                anchor="mm",
            )

        # Column numbers (top)
        for col in range(self._width):
            text = str(col)
            x = self._margin + col * self._cell_size + self._cell_size // 2
            draw.text(
                (x, self._margin // 2),
                text,
                fill=self.COLORS["text"],
                font=font,
                anchor="mm",
            )

        # Draw empty cells (white background)
        for row in range(self._height):
            for col in range(self._width):
                x0 = self._margin + col * self._cell_size + 1
                y0 = self._margin + row * self._cell_size + 1
                x1 = x0 + self._cell_size - 2
                y1 = y0 + self._cell_size - 2
                draw.rectangle([x0, y0, x1, y1], fill=self.COLORS["background"])

        # Draw food
        if self._food:
            row, col = self._food
            x0 = self._margin + col * self._cell_size + 1
            y0 = self._margin + row * self._cell_size + 1
            x1 = x0 + self._cell_size - 2
            y1 = y0 + self._cell_size - 2
            draw.rectangle([x0, y0, x1, y1], fill=self.COLORS["food"])

        # Draw snake
        for i, (row, col) in enumerate(self._snake):
            x0 = self._margin + col * self._cell_size + 1
            y0 = self._margin + row * self._cell_size + 1
            x1 = x0 + self._cell_size - 2
            y1 = y0 + self._cell_size - 2
            color = self.COLORS["snake_head"] if i == 0 else self.COLORS["snake_body"]
            draw.rectangle([x0, y0, x1, y1], fill=color)

        # Draw connecting lines between snake segments
        if len(self._snake) > 1:
            points = []
            for row, col in self._snake:
                cx = self._margin + col * self._cell_size + self._cell_size // 2
                cy = self._margin + row * self._cell_size + self._cell_size // 2
                points.append((cx, cy))
            draw.line(points, fill=self.COLORS["line"], width=4)

        return img

    def _generate_snake(self) -> None:
        """Generate a random snake on the board."""
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        max_attempts = 100
        for _ in range(max_attempts):
            # Random starting position for head
            head_row = self.np_random.integers(0, self._height)
            head_col = self.np_random.integers(0, self._width)
            self._snake = [(head_row, head_col)]

            success = True
            for _ in range(self._initial_snake_length - 1):
                curr_row, curr_col = self._snake[-1]
                valid_dirs = directions.copy()
                self.np_random.shuffle(valid_dirs)

                found_valid = False
                for dr, dc in valid_dirs:
                    new_row, new_col = curr_row + dr, curr_col + dc

                    if (
                        0 <= new_row < self._height
                        and 0 <= new_col < self._width
                        and (new_row, new_col) not in self._snake
                    ):
                        self._snake.append((new_row, new_col))
                        found_valid = True
                        break

                if not found_valid:
                    success = False
                    break

            if success:
                return

        # Fallback: create a short snake
        head_row = self.np_random.integers(0, self._height)
        head_col = self.np_random.integers(0, self._width)
        self._snake = [(head_row, head_col)]

        for dr, dc in directions:
            new_row, new_col = head_row + dr, head_col + dc
            if 0 <= new_row < self._height and 0 <= new_col < self._width:
                self._snake.append((new_row, new_col))
                break

    def _generate_food(self) -> None:
        """Generate food at a random empty position."""
        empty_positions = [
            (r, c)
            for r in range(self._height)
            for c in range(self._width)
            if (r, c) not in self._snake
        ]

        if empty_positions:
            idx = self.np_random.integers(0, len(empty_positions))
            self._food = empty_positions[idx]
        else:
            self._food = None

    def _get_observation_text(self) -> str:
        """Get text description of current game state."""
        if self._game_over:
            return f"Game Over! Snake length: {len(self._snake)}"

        head = self._snake[0]
        food = self._food if self._food else "None"
        return f"Snake head: {head}, Food: {food}, Length: {len(self._snake)}"

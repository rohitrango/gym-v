"""Langton's Ant game based on Game-RL."""

from __future__ import annotations

from importlib import resources
import random
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation, get_logger

logger = get_logger()


class GameRLLangtonAntEnv(Env):
    """Langton's Ant cellular automaton environment.

    A cellular automaton where an ant moves on a grid following simple rules:
    - At a white cell: turn 90° right, flip cell to black, move forward
    - At a black cell: turn 90° left, flip cell to white, move forward

    Args:
        grid_size: Size of the grid (default 15)
        cell_size: Size of each cell in pixels for rendering (default 30)
        init_black_ratio: Initial ratio of black cells (default 0.1)
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

    def __init__(
        self,
        grid_size: int = 15,
        cell_size: int = 30,
        init_black_ratio: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._grid_size = grid_size
        self._cell_size = cell_size
        self._init_black_ratio = init_black_ratio
        self._margin = 20

        # Game state (initialized in reset)
        self._grid: list[list[int]] = []  # 0=white, 1=black
        self._ant_x: int = 0
        self._ant_y: int = 0
        self._ant_direction: str = "up"
        self._step_count: int = 0

    @property
    def description(self) -> str:
        return dedent(f"""
            This is Langton's Ant, a cellular automaton with simple rules:

            Rules:
            - At a WHITE cell: turn 90° RIGHT, flip cell to BLACK, move forward one unit
            - At a BLACK cell: turn 90° LEFT, flip cell to WHITE, move forward one unit
            - The ant wraps around the edges of the grid

            The grid uses a coordinate system where:
            - Row coordinates are 1-{self._grid_size} from top to bottom
            - Column coordinates are 1-{self._grid_size} from left to right

            Actions:
            - 'step', 'next', 'n', or 'space': Advance one step

            Example: 'step' to move the ant one step
        """).strip()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)

        # Initialize grid with random black cells
        self._grid = [[0] * self._grid_size for _ in range(self._grid_size)]
        num_black = int(self._grid_size * self._grid_size * self._init_black_ratio)
        cells = [(i, j) for i in range(self._grid_size) for j in range(self._grid_size)]
        black_cells = random.sample(cells, num_black)
        for i, j in black_cells:
            self._grid[i][j] = 1

        # Initialize ant at center
        self._ant_x = self._grid_size // 2
        self._ant_y = self._grid_size // 2
        self._ant_direction = random.choice(self.DIRECTIONS)
        self._step_count = 0

        logger.info(
            f"Reset Langton's Ant ({self._grid_size}x{self._grid_size}, "
            f"{num_black} black cells, ant at ({self._ant_x + 1}, {self._ant_y + 1}), "
            f"facing {self._ant_direction})."
        )

        obs = Observation(image=self.render(), text=self._get_observation_text())
        return obs, {}

    def inner_step(
        self, action: str
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        info: dict[str, Any] = {}
        reward = 0.0
        terminated = False
        truncated = False

        # Parse action
        action = action.strip().lower()

        if action in ["step", "next", "n", "space", " "]:
            # Execute one step
            self._execute_step()
            reward = 0.01
        else:
            logger.warning(f"Invalid action: {action}")
            obs = Observation(
                image=self.render(),
                text="Invalid action. Use 'step', 'next', 'n', or 'space' to advance.",
            )
            return obs, -0.1, False, False, info

        info = {
            "step_count": self._step_count,
            "ant_position": (self._ant_x + 1, self._ant_y + 1),
            "ant_direction": self._ant_direction,
        }

        obs = Observation(image=self.render(), text=self._get_observation_text())
        return obs, reward, terminated, truncated, info

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
            font = ImageFont.truetype(str(self.assets_dir / "DejaVuSans.ttf"), 16)
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

    def _get_observation_text(self) -> str:
        """Get text description of current state."""
        black_count = sum(row.count(1) for row in self._grid)
        white_count = self._grid_size * self._grid_size - black_count
        return (
            f"Step {self._step_count}: Ant at ({self._ant_x + 1}, {self._ant_y + 1}) "
            f"facing {self._ant_direction} | Black: {black_count}, White: {white_count}"
        )

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

        self._step_count += 1

    def _turn_right(self) -> None:
        """Turn the ant 90 degrees right."""
        current_idx = self.DIRECTIONS.index(self._ant_direction)
        self._ant_direction = self.DIRECTIONS[(current_idx + 1) % 4]

    def _turn_left(self) -> None:
        """Turn the ant 90 degrees left."""
        current_idx = self.DIRECTIONS.index(self._ant_direction)
        self._ant_direction = self.DIRECTIONS[(current_idx - 1) % 4]

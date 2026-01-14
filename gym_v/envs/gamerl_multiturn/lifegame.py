"""Conway's Game of Life based on GameRL."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation, get_logger

logger = get_logger()


class GameRLLifegameEnv(Env):
    """Conway's Game of Life environment.

    A cellular automaton where each cell can be alive or dead, evolving
    according to Conway's rules based on neighbor counts.

    Args:
        grid_size: Size of the grid (grid_size x grid_size) (default 30)
        cell_size: Size of each cell in pixels for rendering (default 20)
        random_init: Whether to randomly initialize the grid (default True)
        init_density: Probability of cell being alive initially (default 0.3)
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    # Colors (matching Game-RL)
    COLORS = {
        "white": (255, 255, 255),
        "black": (0, 0, 0),
        "gray": (128, 128, 128),
    }

    # Actions
    ACTIONS = {
        "step": "Advance one generation",
        "next": "Advance one generation",
        "n": "Advance one generation",
        "space": "Advance one generation",
    }

    def __init__(
        self,
        grid_size: int = 30,
        cell_size: int = 20,
        random_init: bool = True,
        init_density: float = 0.3,
        num_players: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._grid_size = grid_size
        self._cell_size = cell_size
        self._random_init = random_init
        self._init_density = max(0.0, min(1.0, init_density))
        self._margin = 40  # Margin for coordinate labels
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        # Game state (initialized in reset)
        self._grid: list[list[int]] = []
        self._generation: int = 0

    @property
    def description(self) -> str:
        return dedent("""
            Conway's Game of Life is a cellular automaton where each cell in the grid can be either alive (black) or dead (white).

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

            Coordinate System: In this grid, we use (row, col) coordinates where row increases from top to bottom (0 at top) and col increases from left to right (0 at left). For example, the top-left cell is at (0, 0), and the cell below it is at (1, 0).

            Actions: 'step', 'next', 'n', or 'space' to advance one generation.
        """).strip()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        super().reset(seed=seed)

        self._generation = 0
        self._init_grid()

        logger.info(f"Reset Game of Life with {self._count_live_cells()} live cells.")

        obs = Observation(image=self.render(), text=self._get_observation_text())
        info = {}
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
        terminated = False
        truncated = False

        action_str = action_str.strip().lower()

        # Validate action
        if action_str not in self.ACTIONS:
            logger.warning(
                f"Invalid action: {action_str}. Valid actions: {list(self.ACTIONS.keys())}"
            )
            obs = Observation(
                image=self.render(),
                text=f"Invalid action: {action_str}. Use 'step', 'next', 'n', or 'space'.",
            )
            return (
                {agent_id: obs for agent_id in self._agent_ids},
                {agent_id: -1.0 for agent_id in self._agent_ids},
                {
                    **{agent_id: False for agent_id in self._agent_ids},
                    "__all__": False,
                },
                {
                    **{agent_id: False for agent_id in self._agent_ids},
                    "__all__": False,
                },
                {agent_id: info for agent_id in self._agent_ids},
            )

        # Evolve the grid by one generation
        old_count = self._count_live_cells()
        self._grid = self._update_grid()
        self._generation += 1
        new_count = self._count_live_cells()

        # Calculate reward based on population change
        population_change = new_count - old_count
        reward = 0.1  # Small positive reward for each step

        # Check if the population died out
        if new_count == 0:
            terminated = True
            reward = -1.0
            logger.info("All cells died - game over.")

        info = {
            "generation": self._generation,
            "live_cells": new_count,
            "population_change": population_change,
        }

        obs = Observation(image=self.render(), text=self._get_observation_text())

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
        width = self._grid_size * self._cell_size + self._margin * 2
        height = self._grid_size * self._cell_size + self._margin * 2
        img = Image.new("RGB", (width, height), self.COLORS["white"])
        draw = ImageDraw.Draw(img)

        # Calculate offset to center the grid
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
            # Horizontal lines
            start = (offset, offset + x * self._cell_size)
            end = (
                offset + self._grid_size * self._cell_size,
                offset + x * self._cell_size,
            )
            draw.line([start, end], fill=self.COLORS["gray"], width=1)

        for y in range(self._grid_size + 1):
            # Vertical lines
            start = (offset + y * self._cell_size, offset)
            end = (
                offset + y * self._cell_size,
                offset + self._grid_size * self._cell_size,
            )
            draw.line([start, end], fill=self.COLORS["gray"], width=1)

        # Draw coordinates
        try:
            font = ImageFont.truetype(str(self.assets_dir / "DejaVuSans.ttf"), 15)
        except Exception:
            font = ImageFont.load_default()

        # Draw row numbers (left side)
        for x in range(self._grid_size):
            text = str(x)
            # Use textbbox for better positioning
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            pos = (
                offset - self._margin // 2 - text_width // 2,
                offset + x * self._cell_size + self._cell_size // 2 - text_height // 2,
            )
            draw.text(pos, text, fill=self.COLORS["black"], font=font)

        # Draw column numbers (top)
        for y in range(self._grid_size):
            text = str(y)
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            pos = (
                offset + y * self._cell_size + self._cell_size // 2 - text_width // 2,
                offset - self._margin // 2 - text_height // 2,
            )
            draw.text(pos, text, fill=self.COLORS["black"], font=font)

        return img

    def _get_observation_text(self) -> str:
        """Get text description of current state."""
        live_cells = self._count_live_cells()
        total_cells = self._grid_size * self._grid_size
        return (
            f"Generation {self._generation} | "
            f"Live cells: {live_cells}/{total_cells} "
            f"({live_cells / total_cells * 100:.1f}%)"
        )

    def _init_grid(self) -> None:
        """Initialize the grid with random or empty cells."""
        self._grid = [[0] * self._grid_size for _ in range(self._grid_size)]
        if self._random_init:
            for x in range(self._grid_size):
                for y in range(self._grid_size):
                    self._grid[x][y] = (
                        1 if self.np_random.random() < self._init_density else 0
                    )

    def _count_neighbors(self, x: int, y: int) -> int:
        """Count live neighbors for a cell, including wrapping around edges."""
        count = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                # Use modulo for wrapping around edges
                nx = (x + dx) % self._grid_size
                ny = (y + dy) % self._grid_size
                count += self._grid[nx][ny]
        return count

    def _update_grid(self) -> list[list[int]]:
        """Update the grid according to Conway's Game of Life rules."""
        new_grid = [[0] * self._grid_size for _ in range(self._grid_size)]

        for x in range(self._grid_size):
            for y in range(self._grid_size):
                neighbors = self._count_neighbors(x, y)
                # Apply Conway's Game of Life rules:
                # 1. Any live cell with 2 or 3 live neighbors survives
                # 2. Any dead cell with exactly 3 live neighbors becomes alive
                # 3. All other cells die or stay dead
                if self._grid[x][y] == 1:  # Live cell
                    new_grid[x][y] = 1 if neighbors in [2, 3] else 0
                else:  # Dead cell
                    new_grid[x][y] = 1 if neighbors == 3 else 0

        return new_grid

    def _count_live_cells(self) -> int:
        """Count total number of live cells in the grid."""
        return sum(sum(row) for row in self._grid)

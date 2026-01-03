"""Maze game based on GameRL."""

from __future__ import annotations

from collections import deque
from importlib import resources
import random
from textwrap import dedent
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation, get_logger

logger = get_logger()


class GameRLMazeEnv(Env):
    """Maze navigation game environment.

    Navigate through a maze to reach the goal while avoiding walls.

    Args:
        size: Maze size - 'small' (9x9), 'medium' (11x11), or 'large' (13x13)
        cell_size: Size of each cell in pixels for rendering (default 40)
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    # Cell types
    WALL = 1
    PATH = 0
    PLAYER = 2
    GOAL = 3

    # Colors
    COLORS = {
        "wall": (135, 206, 250),  # Light blue
        "path": (255, 255, 255),  # White
        "player": (255, 0, 0),  # Red
        "goal": (0, 255, 0),  # Green
        "grid_line": (0, 0, 0),  # Black
        "text": (0, 0, 0),  # Black
    }

    # Size configurations
    SIZE_CONFIG = {
        "small": 9,
        "medium": 11,
        "large": 13,
    }

    def __init__(
        self,
        size: str = "small",
        cell_size: int = 40,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert (
            size in self.SIZE_CONFIG
        ), f"size must be one of {list(self.SIZE_CONFIG.keys())}"

        self._size_name = size
        self._grid_size = self.SIZE_CONFIG[size]
        self._cell_size = cell_size
        self._padding = 30  # Padding for coordinate labels

        # Game state (initialized in reset)
        self._maze: np.ndarray = np.zeros((self._grid_size, self._grid_size), dtype=int)
        self._player_pos: tuple[int, int] = (0, 0)  # (row, col)
        self._goal_pos: tuple[int, int] = (0, 0)
        self._game_over: bool = False
        self._won: bool = False
        self._steps: int = 0

    @property
    def description(self) -> str:
        return dedent("""
            **Rules:**
            1. This is a maze mini-game.The player needs to navigate around obstacles to reach the destination and achieve victory.
            2. The red circle represents the player, the green block is the goal and the blue blocks are obstacles.
            3. The player can only move within the white blocks.
            4. The coordinates are given in the format (row, col), where row represents the vertical position and col represents the horizontal position.

            Available Actions: up, down, left, right
        """).strip()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self._game_over = False
        self._won = False
        self._steps = 0

        # Generate maze
        self._generate_maze()

        logger.info(f"Reset Maze game ({self._size_name}).")

        obs = Observation(image=self.render(), text=self._get_observation_text())
        return obs, {}

    def _generate_maze(self) -> None:
        """Generate a random maze using recursive backtracking algorithm."""
        size = self._grid_size

        # Initialize all cells as walls
        self._maze = np.ones((size, size), dtype=int)

        # Use recursive backtracking to create paths
        # Start from (1, 1) to leave border
        self._carve_passages(1, 1)

        # Ensure the maze has a valid structure
        # Place player at a random path cell in upper portion
        path_cells_upper = []
        for r in range(1, size // 2):
            for c in range(1, size - 1):
                if self._maze[r, c] == self.PATH:
                    path_cells_upper.append((r, c))

        if path_cells_upper:
            self._player_pos = random.choice(path_cells_upper)
        else:
            self._player_pos = (1, 1)
            self._maze[1, 1] = self.PATH

        # Place goal at a random path cell in lower portion
        path_cells_lower = []
        for r in range(size // 2 + 1, size - 1):
            for c in range(1, size - 1):
                if self._maze[r, c] == self.PATH:
                    path_cells_lower.append((r, c))

        if path_cells_lower:
            self._goal_pos = random.choice(path_cells_lower)
        else:
            self._goal_pos = (size - 2, size - 2)
            self._maze[size - 2, size - 2] = self.PATH

        # Ensure there's a path from player to goal
        if not self._path_exists(self._player_pos, self._goal_pos):
            # Create a simple path if no path exists
            self._create_simple_path()

        # Mark player and goal in maze
        self._maze[self._player_pos[0], self._player_pos[1]] = self.PLAYER
        self._maze[self._goal_pos[0], self._goal_pos[1]] = self.GOAL

    def _carve_passages(self, row: int, col: int) -> None:
        """Recursively carve passages in the maze."""
        self._maze[row, col] = self.PATH

        # Random direction order
        directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
        random.shuffle(directions)

        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc

            # Check bounds (stay within maze, leave border)
            if (
                1 <= new_row < self._grid_size - 1
                and 1 <= new_col < self._grid_size - 1
            ):
                if self._maze[new_row, new_col] == self.WALL:
                    # Carve through the wall between current and new cell
                    self._maze[row + dr // 2, col + dc // 2] = self.PATH
                    self._carve_passages(new_row, new_col)

    def _path_exists(self, start: tuple[int, int], end: tuple[int, int]) -> bool:
        """Check if a path exists between two points using BFS."""
        if start == end:
            return True

        visited = set()
        queue = deque([start])
        visited.add(start)

        while queue:
            row, col = queue.popleft()

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_row, new_col = row + dr, col + dc

                if (new_row, new_col) == end:
                    return True

                if (
                    0 <= new_row < self._grid_size
                    and 0 <= new_col < self._grid_size
                    and (new_row, new_col) not in visited
                    and self._maze[new_row, new_col] != self.WALL
                ):
                    visited.add((new_row, new_col))
                    queue.append((new_row, new_col))

        return False

    def _create_simple_path(self) -> None:
        """Create a simple path from player to goal if none exists."""
        row, col = self._player_pos
        goal_row, goal_col = self._goal_pos

        # Move vertically first, then horizontally
        while row != goal_row:
            if row < goal_row:
                row += 1
            else:
                row -= 1
            if self._maze[row, col] == self.WALL:
                self._maze[row, col] = self.PATH

        while col != goal_col:
            if col < goal_col:
                col += 1
            else:
                col -= 1
            if self._maze[row, col] == self.WALL:
                self._maze[row, col] = self.PATH

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

        # Map actions to direction
        action_map = {
            "up": (-1, 0),
            "down": (1, 0),
            "left": (0, -1),
            "right": (0, 1),
            "w": (-1, 0),
            "s": (1, 0),
            "a": (0, -1),
            "d": (0, 1),
        }

        if action_lower not in action_map:
            info["invalid_action"] = True
            obs = Observation(image=self.render(), text=self._get_observation_text())
            return obs, reward, terminated, truncated, info

        info["invalid_action"] = False
        dr, dc = action_map[action_lower]
        new_row = self._player_pos[0] + dr
        new_col = self._player_pos[1] + dc

        # Check if move is valid
        if (
            0 <= new_row < self._grid_size
            and 0 <= new_col < self._grid_size
            and self._maze[new_row, new_col] != self.WALL
        ):
            # Clear old position
            self._maze[self._player_pos[0], self._player_pos[1]] = self.PATH

            # Check if reached goal
            if (new_row, new_col) == self._goal_pos:
                self._game_over = True
                self._won = True
                terminated = True
                reward = 10.0
                info["win"] = True
            else:
                reward = -0.1  # Small penalty for each step

            # Update position
            self._player_pos = (new_row, new_col)
            self._maze[new_row, new_col] = self.PLAYER
            self._steps += 1
        else:
            # Invalid move (into wall or out of bounds)
            reward = -0.5
            info["blocked"] = True

        info["steps"] = self._steps
        info["player_pos"] = self._player_pos
        info["goal_pos"] = self._goal_pos

        obs = Observation(image=self.render(), text=self._get_observation_text())
        return obs, reward, terminated, truncated, info

    def render(self) -> Image.Image:
        """Render the game state as a PIL Image."""
        width = self._grid_size * self._cell_size + 2 * self._padding
        height = self._grid_size * self._cell_size + 2 * self._padding

        img = Image.new("RGB", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Try to load font
        font_path = self.assets_dir / "DejaVuSans.ttf"
        if font_path.exists():
            font = ImageFont.truetype(str(font_path), 14)
        else:
            font = ImageFont.load_default()

        # Draw coordinate labels
        for i in range(self._grid_size):
            # Column numbers (top)
            x = i * self._cell_size + self._padding + self._cell_size // 2
            draw.text(
                (x, self._padding - 18),
                str(i),
                fill=self.COLORS["text"],
                font=font,
                anchor="mm",
            )
            # Row numbers (left)
            y = i * self._cell_size + self._padding + self._cell_size // 2
            draw.text(
                (self._padding - 15, y),
                str(i),
                fill=self.COLORS["text"],
                font=font,
                anchor="mm",
            )

        # Draw cells
        for row in range(self._grid_size):
            for col in range(self._grid_size):
                x1 = col * self._cell_size + self._padding
                y1 = row * self._cell_size + self._padding
                x2 = x1 + self._cell_size
                y2 = y1 + self._cell_size

                cell = self._maze[row, col]

                if cell == self.WALL:
                    color = self.COLORS["wall"]
                elif cell == self.GOAL:
                    color = self.COLORS["goal"]
                elif cell == self.PLAYER:
                    # Draw path background first, then player circle
                    draw.rectangle([x1, y1, x2, y2], fill=self.COLORS["path"])
                    # Draw red circle for player
                    margin = self._cell_size // 5
                    draw.ellipse(
                        [x1 + margin, y1 + margin, x2 - margin, y2 - margin],
                        fill=self.COLORS["player"],
                    )
                    # Draw grid line
                    draw.rectangle([x1, y1, x2, y2], outline=self.COLORS["grid_line"])
                    continue
                else:
                    color = self.COLORS["path"]

                draw.rectangle([x1, y1, x2, y2], fill=color)
                draw.rectangle([x1, y1, x2, y2], outline=self.COLORS["grid_line"])

        return img

    def _get_observation_text(self) -> str:
        """Get text description of current game state."""
        if self._game_over:
            if self._won:
                return f"Victory! Reached the goal in {self._steps} steps."
            return f"Game Over! Steps: {self._steps}"

        return f"Player at {self._player_pos}, Goal at {self._goal_pos}, Steps: {self._steps}"

    def get_available_directions(self) -> list[str]:
        """Get list of available directions the player can move."""
        directions = []
        row, col = self._player_pos

        if row > 0 and self._maze[row - 1, col] != self.WALL:
            directions.append("up")
        if row < self._grid_size - 1 and self._maze[row + 1, col] != self.WALL:
            directions.append("down")
        if col > 0 and self._maze[row, col - 1] != self.WALL:
            directions.append("left")
        if col < self._grid_size - 1 and self._maze[row, col + 1] != self.WALL:
            directions.append("right")

        return directions

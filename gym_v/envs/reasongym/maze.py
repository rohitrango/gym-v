"""Maze single-turn environment backed by reasoning-gym."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from reasoning_gym.factory import create_dataset

from gym_v import Env, Observation, get_logger

logger = get_logger()


class ReasoningGymMazeEnv(Env):
    """Maze pathfinding puzzle using reasoning-gym's Maze dataset.

    The player navigates from start to goal through a maze grid.
    The objective is to find the minimum number of steps to reach the goal.

    Args:
        dataset_kwargs: Configuration parameters for the reasoning-gym dataset
        cell_px: Size of each cell in pixels for rendering
        padding: Padding around the grid in pixels for rendering
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    def __init__(
        self,
        dataset_kwargs: dict[str, Any] | None = None,
        cell_px: int = 48,
        padding: int = 24,
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._dataset_kwargs = dataset_kwargs or {}
        self._cell_px = cell_px
        self._padding = padding
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._seed: int | None = None
        self._dataset = None
        self._entry: dict[str, Any] | None = None
        self._entry_idx: int | None = None
        self._metadata: dict[str, Any] | None = None
        self._grid: list[str] | None = None
        self._oracle_answer: str | None = None

    @property
    def description(self) -> str:
        """Return description with current maze visual representation.

        Original reasoning-gym question format:
        ```
        Navigate from 'S' (start) to 'G' (goal):

        ```
        #.#S###
        #.#...#
        #.###.#
        #.....#
        ###.#.#
        G...#.#
        #######
        ```
        Legend: '#' = Wall, '.' = Passage

        What is the minimum number of steps to reach the goal?
        Give only the number of steps as your final answer, no other text or formatting.
        ```
        Note: The wall/path/start/goal characters are randomly selected for each puzzle.

        Original reasoning-gym answer format:
        - A single integer string, e.g. "12", "25", etc.
        """
        start_char = self._metadata.get("start", "S") if self._metadata else "S"
        goal_char = self._metadata.get("goal", "G") if self._metadata else "G"

        return dedent(f"""
            Navigate from the start to the goal in the maze.

            In the image:
            - Start position: Green cell labeled '{start_char}'
            - Goal position: Red cell labeled '{goal_char}'

            You can move up, down, left, or right (not diagonally) through passages.

            What is the minimum number of steps to reach the goal?
            Give only the number of steps as your final answer, no other text or formatting, e.g. "12", "25", etc.
        """).strip()

    def _make_dataset(self, *, seed: int | None):
        kwargs = self._dataset_kwargs.copy()
        if seed is not None and "seed" not in kwargs:
            kwargs["seed"] = seed
        if "size" not in kwargs:
            kwargs["size"] = 500

        return create_dataset("maze", **kwargs)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        super().reset(seed=seed)
        self._seed = seed

        self._dataset = self._make_dataset(seed=self._seed)
        self._entry_idx = int(self.np_random.integers(0, len(self._dataset)))
        self._entry = self._dataset[self._entry_idx]

        self._oracle_answer = self._entry["answer"]
        self._metadata = self._entry.get("metadata", {})
        self._grid = self._metadata.get("grid", [])

        logger.info("Reset ReasoningGym Maze.")

        # obs.text = only the maze grid (caption), not the full question
        maze_text = "\n".join(self._grid) if self._grid else ""

        obs = Observation(
            image=self.render(),
            text=maze_text,
            metadata={
                **self._metadata,
                "text_prompt": self._entry.get("question", ""),
            },
        )
        info = {
            "reasoning_gym_seed": self._seed,
            "reasoning_gym_index": self._entry_idx,
            "oracle_answer": self._oracle_answer,
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
        answer = action[agent_id]
        reward = self._dataset.score_answer(answer=answer, entry=self._entry)

        obs = Observation(
            image=self.render(),
            text=None,
            metadata={
                **self._metadata,
                "text_prompt": self._entry.get("question", ""),
            },
        )
        info = {
            "reasoning_gym_seed": self._seed,
            "reasoning_gym_index": self._entry_idx,
            "oracle_answer": self._oracle_answer,
        }

        terminated = True
        truncated = False

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
        return self._render_maze_grid(
            self._grid,
            cell_px=self._cell_px,
            padding=self._padding,
            start_char=self._metadata.get("start", "S"),
            goal_char=self._metadata.get("goal", "G"),
            wall_char=self._metadata.get("wall", "#"),
            path_char=self._metadata.get("path", "."),
        )

    def _render_maze_grid(
        self,
        grid: list[str],
        cell_px: int = 48,
        padding: int = 24,
        start_char: str = "S",
        goal_char: str = "G",
        wall_char: str = "#",
        path_char: str = ".",
        bg: tuple[int, int, int] = (250, 250, 250),
        wall_color: tuple[int, int, int] = (40, 40, 40),
        path_color: tuple[int, int, int] = (220, 220, 220),
        start_color: tuple[int, int, int] = (76, 175, 80),
        goal_color: tuple[int, int, int] = (244, 67, 54),
    ) -> Image.Image:
        if not grid:
            return Image.new("RGB", (200, 200), bg)

        rows = len(grid)
        cols = len(grid[0]) if grid else 0
        width = padding * 2 + cell_px * cols
        height = padding * 2 + cell_px * rows
        img = Image.new("RGB", (width, height), bg)
        draw = ImageDraw.Draw(img)

        font_path = self.assets_dir / "DejaVuSans.ttf"
        if font_path.exists():
            font = ImageFont.truetype(str(font_path), int(cell_px * 0.4))
        else:
            logger.warning(f"Font file not found: {font_path}, using default font")
            font = ImageFont.load_default()

        for r, row in enumerate(grid):
            for c, char in enumerate(row):
                x = padding + c * cell_px
                y = padding + r * cell_px

                # Determine cell color
                if char == wall_char:
                    color = wall_color
                elif char == start_char:
                    color = start_color
                elif char == goal_char:
                    color = goal_color
                else:
                    color = path_color

                # Draw cell
                draw.rectangle(
                    [x, y, x + cell_px - 1, y + cell_px - 1],
                    fill=color,
                    outline=(100, 100, 100),
                    width=1,
                )

                # Draw start/goal labels (use actual character from maze)
                if char == start_char or char == goal_char:
                    label = char  # Use the actual character (e.g., 'E', '1')
                    bbox = draw.textbbox((0, 0), label, font=font)
                    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    cx = x + cell_px // 2
                    cy = y + cell_px // 2
                    draw.text(
                        (cx - tw // 2, cy - th // 2),
                        label,
                        fill=(255, 255, 255),
                        font=font,
                    )

        return img

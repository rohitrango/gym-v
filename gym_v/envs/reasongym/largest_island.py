"""Largest Island single-turn environment backed by reasoning-gym."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from reasoning_gym.factory import create_dataset

from gym_v import Env, Observation, get_logger

logger = get_logger()


class ReasoningGymLargestIslandEnv(Env):
    """Largest Island puzzle using reasoning-gym's dataset.

    The player finds the maximum area of an island in a binary grid.

    Args:
        dataset_kwargs: Configuration parameters for the reasoning-gym dataset
        cell_px: Size of each cell in pixels for rendering
        padding: Padding around the grid in pixels for rendering
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    def __init__(
        self,
        dataset_kwargs: dict[str, Any] | None = None,
        cell_px: int = 40,
        padding: int = 20,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._dataset_kwargs = dataset_kwargs or {}
        self._cell_px = cell_px
        self._padding = padding

        self._seed: int | None = None
        self._dataset = None
        self._entry: dict[str, Any] | None = None
        self._entry_idx: int | None = None
        self._metadata: dict[str, Any] | None = None
        self._grid: list[list[int]] | None = None
        self._oracle_answer: str | None = None

    @property
    def description(self) -> str:
        """Return description for Largest Island puzzle.

        Original reasoning-gym question format:
        ```
        You are given the following 10 x 5 binary matrix grid:
        0 0 0 0 0
        0 0 0 0 0
        ...

        An island is a group of 1's (representing land) connected 4-directionally
        (horizontal or vertical). You may assume all four edges of the grid are
        surrounded by water.

        The area of an island is the number of cells with a value 1 in the island.

        Return the maximum area of an island in grid. If there is no island, return 0.
        ```

        Original reasoning-gym answer format:
        ```
        0
        ```
        (Single integer representing the maximum island area)
        """
        return dedent("""
            Find the maximum area of an island in the grid.

            An island is a group of land connected 4-directionally (up, down, left, right).You may assume all four edges of the grid are surrounded by water
            The area of an island is the count of land in that island.

            Return the maximum area. If there is no island, return 0.

            Output format: A single integer.
        """).strip()

    def _make_dataset(self, *, seed: int | None):
        kwargs = self._dataset_kwargs.copy()
        if seed is not None and "seed" not in kwargs:
            kwargs["seed"] = seed
        if "size" not in kwargs:
            kwargs["size"] = 500

        return create_dataset("largest_island", **kwargs)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)
        self._seed = seed

        self._dataset = self._make_dataset(seed=self._seed)
        self._entry_idx = int(self.np_random.integers(0, len(self._dataset)))
        self._entry = self._dataset[self._entry_idx]

        self._oracle_answer = self._entry["answer"]
        self._metadata = self._entry.get("metadata", {})
        self._grid = self._metadata.get("grid", [])

        logger.info("Reset ReasoningGym Largest Island.")

        grid_text = self._format_grid(self._grid)

        obs = Observation(
            image=self.render(),
            text=grid_text,
            metadata=self._metadata,
        )
        info = {
            "reasoning_gym_seed": self._seed,
            "reasoning_gym_index": self._entry_idx,
            "oracle_answer": self._oracle_answer,
        }
        return obs, info

    def inner_step(
        self, action: str
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        answer = action
        reward = self._dataset.score_answer(answer=answer, entry=self._entry)

        obs = Observation(
            image=self.render(),
            text=None,
            metadata=self._metadata,
        )
        info = {
            "reasoning_gym_seed": self._seed,
            "reasoning_gym_index": self._entry_idx,
            "oracle_answer": self._oracle_answer,
        }

        return obs, reward, True, False, info

    def _format_grid(self, grid: list[list[int]]) -> str:
        if not grid:
            return ""
        return "\n".join(" ".join(str(x) for x in row) for row in grid)

    def render(self) -> Image.Image | list[Image.Image] | None:
        return self._render_island_grid(
            self._grid, cell_px=self._cell_px, padding=self._padding
        )

    def _render_island_grid(
        self,
        grid: list[list[int]],
        cell_px: int = 40,
        padding: int = 20,
        bg: tuple[int, int, int] = (245, 245, 250),
        water_color: tuple[int, int, int] = (70, 130, 220),
        land_color: tuple[int, int, int] = (100, 200, 100),
        grid_color: tuple[int, int, int] = (180, 180, 180),
    ) -> Image.Image:
        """Render grid as a clean island map."""
        if not grid:
            return Image.new("RGB", (200, 200), bg)

        rows = len(grid)
        cols = len(grid[0]) if grid else 0

        # Calculate dimensions
        legend_h = 60
        grid_width = cell_px * cols
        grid_height = cell_px * rows

        # Legend needs space for 2 items: "Water" and "Land"
        legend_item_width = 110  # 30px box + 10px gap + ~70px text
        legend_total_width = legend_item_width * 2

        # Image width is max of grid width and legend width, plus padding
        width = max(grid_width, legend_total_width) + padding * 2
        height = grid_height + padding * 2 + legend_h

        img = Image.new("RGB", (width, height), bg)
        draw = ImageDraw.Draw(img)

        font_path = self.assets_dir / "DejaVuSans.ttf"
        if font_path.exists():
            font = ImageFont.truetype(str(font_path), int(cell_px * 0.5))
            legend_font = ImageFont.truetype(str(font_path), 16)
        else:
            logger.warning(f"Font file not found: {font_path}, using default font")
            font = ImageFont.load_default()
            legend_font = font

        # Calculate grid start position (center horizontally if needed)
        grid_start_x = (width - grid_width) // 2
        grid_start_y = padding

        # Draw cells
        for r in range(rows):
            for c in range(cols):
                x = grid_start_x + c * cell_px
                y = grid_start_y + r * cell_px

                val = grid[r][c] if r < len(grid) and c < len(grid[r]) else 0
                cell_color = land_color if val == 1 else water_color

                # Draw cell
                draw.rectangle(
                    [x, y, x + cell_px - 1, y + cell_px - 1],
                    fill=cell_color,
                    outline=grid_color,
                    width=1,
                )

                # Add subtle effects
                if val == 1:
                    # Land: shadow at bottom-right
                    shadow_color = (
                        max(0, land_color[0] - 40),
                        max(0, land_color[1] - 40),
                        max(0, land_color[2] - 40),
                    )
                    draw.line(
                        [(x, y + cell_px - 1), (x + cell_px - 1, y + cell_px - 1)],
                        fill=shadow_color,
                        width=2,
                    )
                    draw.line(
                        [(x + cell_px - 1, y), (x + cell_px - 1, y + cell_px - 1)],
                        fill=shadow_color,
                        width=2,
                    )
                    # Highlight at top-left
                    highlight_color = (
                        min(255, land_color[0] + 40),
                        min(255, land_color[1] + 40),
                        min(255, land_color[2] + 40),
                    )
                    draw.line(
                        [(x, y), (x + cell_px - 1, y)], fill=highlight_color, width=2
                    )
                    draw.line(
                        [(x, y), (x, y + cell_px - 1)], fill=highlight_color, width=2
                    )
                else:
                    # Water: wave lines
                    wave_color = (
                        min(255, water_color[0] + 30),
                        min(255, water_color[1] + 30),
                        min(255, water_color[2] + 30),
                    )
                    for wave_y in [cell_px // 3, 2 * cell_px // 3]:
                        draw.line(
                            [(x + 3, y + wave_y), (x + cell_px - 4, y + wave_y)],
                            fill=wave_color,
                            width=1,
                        )

        # Draw outer border for grid
        draw.rectangle(
            [
                grid_start_x - 1,
                grid_start_y - 1,
                grid_start_x + grid_width,
                grid_start_y + grid_height,
            ],
            outline=(60, 60, 60),
            width=2,
        )

        # Draw legend at bottom (centered)
        legend_y = height - legend_h + 10
        legend_items = [("Water", water_color), ("Land", land_color)]
        legend_start_x = (width - legend_total_width) // 2

        for i, (label, color) in enumerate(legend_items):
            legend_x = legend_start_x + i * legend_item_width
            box_size = 30

            # Draw colored box
            draw.rectangle(
                [legend_x, legend_y, legend_x + box_size, legend_y + box_size],
                fill=color,
                outline=(60, 60, 60),
                width=2,
            )

            # Add same effects as in grid
            if "Land" in label:
                shadow_color = (
                    max(0, color[0] - 40),
                    max(0, color[1] - 40),
                    max(0, color[2] - 40),
                )
                draw.line(
                    [
                        (legend_x, legend_y + box_size),
                        (legend_x + box_size, legend_y + box_size),
                    ],
                    fill=shadow_color,
                    width=2,
                )
                draw.line(
                    [
                        (legend_x + box_size, legend_y),
                        (legend_x + box_size, legend_y + box_size),
                    ],
                    fill=shadow_color,
                    width=2,
                )
            else:
                wave_color = (
                    min(255, color[0] + 30),
                    min(255, color[1] + 30),
                    min(255, color[2] + 30),
                )
                for wave_y_offset in [box_size // 3, 2 * box_size // 3]:
                    draw.line(
                        [
                            (legend_x + 3, legend_y + wave_y_offset),
                            (legend_x + box_size - 3, legend_y + wave_y_offset),
                        ],
                        fill=wave_color,
                        width=1,
                    )

            # Draw label
            draw.text(
                (legend_x + box_size + 10, legend_y + 7),
                label,
                fill=(30, 30, 30),
                font=legend_font,
            )

        return img

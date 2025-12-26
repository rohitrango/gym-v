"""Rotten Oranges single-turn environment backed by reasoning-gym."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw
from reasoning_gym.factory import create_dataset

from gym_v import Env, Observation, get_logger

logger = get_logger()


class ReasoningGymRottenOrangesEnv(Env):
    """Rotten Oranges puzzle using reasoning-gym's dataset.

    The player determines minimum minutes for all oranges to rot.

    Args:
        dataset_kwargs: Configuration parameters for the reasoning-gym dataset
        cell_px: Size of each cell in pixels for rendering
        padding: Padding around the grid in pixels for rendering
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    def __init__(
        self,
        dataset_kwargs: dict[str, Any] | None = None,
        cell_px: int = 32,
        padding: int = 16,
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
        """Return description for Rotten Oranges puzzle.

        Original reasoning-gym question format:
        ```
        You are given an n x n grid where each cell can have one of three values:
        - 0 representing an empty cell
        - 1 representing a fresh orange
        - 2 representing a rotten orange

        Every minute, any fresh orange that is 4-directionally adjacent to a rotten
        orange becomes rotten.

        Your task is determine the minimum number of minutes that must elapse until
        no cell has a fresh orange. If this is impossible, return -1.

        Now, determine the minimum number of minutes...
        1 1 1 1 2 1 1 1 1 0 ...
        ```

        Original reasoning-gym answer format:
        ```
        6
        ```
        (Single integer: minutes or -1 if impossible)
        """
        return dedent("""
            Determine the minimum minutes until no fresh orange remains.

            Every minute, any fresh orange that is 4-directionally adjacent to a rotten
            orange becomes rotten.

            Your task is determine the minimum number of minutes that must elapse until
            no cell has a fresh orange. If this is impossible, return -1.

            Now, determine the minimum number of minutes. If impossible (some fresh oranges can't be reached), return -1.

            Output format: Only a single integer.
        """).strip()

    def _make_dataset(self, *, seed: int | None):
        kwargs = self._dataset_kwargs.copy()
        if seed is not None and "seed" not in kwargs:
            kwargs["seed"] = seed
        if "size" not in kwargs:
            kwargs["size"] = 500

        return create_dataset("rotten_oranges", **kwargs)

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
        # Note: reasoning-gym uses "matrix" key for this dataset
        self._grid = self._metadata.get("matrix", [])

        logger.info("Reset ReasoningGym Rotten Oranges.")

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

    def render(self) -> Image.Image:
        return self._render_orange_grid(
            self._grid, cell_px=self._cell_px, padding=self._padding
        )

    def _draw_orange(
        self,
        draw: ImageDraw.Draw,
        cx: int,
        cy: int,
        radius: int,
        is_rotten: bool = False,
    ) -> None:
        """Draw a beautiful orange at the given center position.

        Args:
            draw: ImageDraw object
            cx, cy: Center coordinates
            radius: Radius of the orange
            is_rotten: If True, draw a rotten orange with spots
        """
        if is_rotten:
            # Rotten orange - brownish green with dark spots
            base_color = (120, 90, 40)
            highlight_color = (150, 110, 50)
            shadow_color = (80, 60, 30)
        else:
            # Fresh orange - vibrant orange with highlight
            base_color = (255, 140, 0)
            highlight_color = (255, 180, 50)
            shadow_color = (230, 100, 0)

        # Draw shadow (slightly offset)
        shadow_offset = max(2, radius // 8)
        draw.ellipse(
            [
                cx - radius + shadow_offset,
                cy - radius + shadow_offset,
                cx + radius + shadow_offset,
                cy + radius + shadow_offset,
            ],
            fill=(100, 100, 100, 80),
        )

        # Draw main orange body
        draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            fill=base_color,
        )

        # Draw gradient effect (bottom shadow arc)
        inner_radius = int(radius * 0.85)
        draw.ellipse(
            [
                cx - inner_radius,
                cy - inner_radius + radius // 4,
                cx + inner_radius,
                cy + inner_radius + radius // 4,
            ],
            fill=shadow_color,
        )

        # Draw main body again but slightly smaller for layered effect
        main_radius = int(radius * 0.9)
        draw.ellipse(
            [cx - main_radius, cy - main_radius, cx + main_radius, cy + main_radius],
            fill=base_color,
        )

        # Draw highlight (top-left shine)
        highlight_radius = int(radius * 0.35)
        highlight_offset_x = -int(radius * 0.3)
        highlight_offset_y = -int(radius * 0.3)
        draw.ellipse(
            [
                cx + highlight_offset_x - highlight_radius,
                cy + highlight_offset_y - highlight_radius,
                cx + highlight_offset_x + highlight_radius,
                cy + highlight_offset_y + highlight_radius,
            ],
            fill=highlight_color,
        )

        # Draw small stem/navel at top
        stem_width = max(2, radius // 6)
        stem_height = max(2, radius // 5)
        draw.ellipse(
            [
                cx - stem_width,
                cy - radius - stem_height // 2,
                cx + stem_width,
                cy - radius + stem_height,
            ],
            fill=(80, 120, 40) if not is_rotten else (50, 70, 30),
        )

        # Add leaf for fresh oranges
        if not is_rotten:
            leaf_length = int(radius * 0.5)
            leaf_points = [
                (cx + stem_width, cy - radius),
                (cx + stem_width + leaf_length, cy - radius - leaf_length // 2),
                (cx + stem_width + leaf_length // 2, cy - radius + 2),
            ]
            draw.polygon(leaf_points, fill=(60, 140, 40))

        # Add rot spots for rotten oranges
        if is_rotten:
            # Draw several dark spots
            spot_positions = [
                (0.2, 0.1),
                (-0.3, 0.2),
                (0.1, 0.4),
                (-0.2, -0.2),
                (0.35, 0.25),
                (-0.1, 0.35),
            ]
            for sx, sy in spot_positions:
                spot_x = cx + int(radius * sx)
                spot_y = cy + int(radius * sy)
                spot_r = max(2, radius // 8)
                draw.ellipse(
                    [
                        spot_x - spot_r,
                        spot_y - spot_r,
                        spot_x + spot_r,
                        spot_y + spot_r,
                    ],
                    fill=(60, 50, 25),
                )

    def _render_orange_grid(
        self,
        grid: list[list[int]],
        cell_px: int = 32,
        padding: int = 16,
        bg: tuple[int, int, int] = (245, 245, 235),
    ) -> Image.Image:
        """Render the orange grid with beautiful fruit graphics."""
        if not grid:
            return Image.new("RGB", (200, 200), bg)

        rows = len(grid)
        cols = len(grid[0]) if grid else 0
        width = padding * 2 + cell_px * cols
        height = padding * 2 + cell_px * rows
        img = Image.new("RGB", (width, height), bg)
        draw = ImageDraw.Draw(img)

        # Draw grid lines (subtle)
        grid_color = (220, 220, 210)
        for r in range(rows + 1):
            y = padding + r * cell_px
            draw.line([(padding, y), (width - padding, y)], fill=grid_color, width=1)
        for c in range(cols + 1):
            x = padding + c * cell_px
            draw.line([(x, padding), (x, height - padding)], fill=grid_color, width=1)

        # Draw oranges
        orange_radius = int(cell_px * 0.38)
        for r in range(rows):
            for c in range(cols):
                val = grid[r][c] if r < len(grid) and c < len(grid[r]) else 0
                cx = padding + c * cell_px + cell_px // 2
                cy = padding + r * cell_px + cell_px // 2

                if val == 0:
                    # Empty cell - just a subtle dot
                    dot_r = max(2, cell_px // 12)
                    draw.ellipse(
                        [cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r],
                        fill=(200, 200, 195),
                    )
                elif val == 1:
                    # Fresh orange
                    self._draw_orange(draw, cx, cy, orange_radius, is_rotten=False)
                else:  # val == 2
                    # Rotten orange
                    self._draw_orange(draw, cx, cy, orange_radius, is_rotten=True)

        # Draw outer border
        draw.rectangle(
            [padding - 1, padding - 1, width - padding, height - padding],
            outline=(150, 150, 140),
            width=2,
        )

        return img

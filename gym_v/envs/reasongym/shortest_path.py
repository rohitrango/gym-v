"""Shortest Path single-turn environment backed by reasoning-gym."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from reasoning_gym.factory import create_dataset

from gym_v import Env, Observation, get_logger

logger = get_logger()


class ReasoningGymShortestPathEnv(Env):
    """Shortest Path puzzle using reasoning-gym's dataset.

    The player finds the shortest path from start to destination in a grid.

    Args:
        dataset_kwargs: Configuration parameters for the reasoning-gym dataset
        cell_px: Size of each cell in pixels for rendering
        padding: Padding around the grid in pixels for rendering
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    def __init__(
        self,
        dataset_kwargs: dict[str, Any] | None = None,
        cell_px: int = 50,
        padding: int = 40,
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
        self._matrix: list[list[str]] | None = None
        self._oracle_answer: str | None = None

    @property
    def description(self) -> str:
        """Return description for Shortest Path puzzle.

        Original reasoning-gym question format:
        ```
        Your task is to find the shortest path from the start to the destination
        point in a grid.

        The grid is represented as a matrix with the following types of cells:
        - *: your starting point
        - #: your destination point
        - O: an open cell
        - X: a blocked cell

        Therefore, you need to find the shortest path from * to #, moving only
        through open cells.

        You may only move in four directions: up, down, left, and right.

        If there is no path from * to #, simply write "infeasible" (without quotes).

        Your output should be a sequence of directions that leads from * to #,
        e.g. right right down down up left

        Now, find the length of the shortest path from * to # in the following grid:
        O X X X O
        O O X X X
        O O # O O
        * X O O X
        O X X O X
        ```

        Original reasoning-gym answer format:
        ```
        up right right
        ```
        (Space-separated directions, or "infeasible")
        """
        return dedent("""
            Find the shortest path from START to GOAL in the maze.

            Rules:
            - Move only in 4 directions: up, down, left, right
            - Walk only through open paths
            - Cannot pass through walls

            If no path exists, answer "infeasible".

            Output format: Space-separated directions.
            Example: right right down down up left
        """).strip()

    def _make_dataset(self, *, seed: int | None):
        kwargs = self._dataset_kwargs.copy()
        if seed is not None and "seed" not in kwargs:
            kwargs["seed"] = seed
        if "size" not in kwargs:
            kwargs["size"] = 500

        return create_dataset("shortest_path", **kwargs)

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
        self._matrix = self._metadata.get("matrix", [])

        logger.info("Reset ReasoningGym Shortest Path.")

        matrix_text = self._format_matrix(self._matrix)

        obs = Observation(
            image=self.render(),
            text=matrix_text,
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

    def _format_matrix(self, matrix: list[list[str]]) -> str:
        if not matrix:
            return ""
        return "\n".join(" ".join(row) for row in matrix)

    def render(self) -> Image.Image | list[Image.Image] | None:
        return self._render_path_grid(
            self._matrix, cell_px=self._cell_px, padding=self._padding
        )

    def _render_path_grid(
        self,
        matrix: list[list[str]],
        cell_px: int = 50,
        padding: int = 40,
        bg: tuple[int, int, int] = (245, 245, 240),
    ) -> Image.Image:
        if not matrix:
            return Image.new("RGB", (400, 400), bg)

        # Load fonts
        font_path = self.assets_dir / "DejaVuSans.ttf"
        if font_path.exists():
            title_font = ImageFont.truetype(str(font_path), 22)
            small_font = ImageFont.truetype(str(font_path), 11)
        else:
            logger.warning(f"Font file not found: {font_path}, using default font")
            title_font = ImageFont.load_default()
            small_font = title_font

        rows = len(matrix)
        cols = len(matrix[0]) if matrix else 0

        # Layout calculations
        header_h = 90
        legend_h = 60
        grid_w = cell_px * cols
        grid_h = cell_px * rows

        # Legend needs minimum width for 4 items (~100px each)
        legend_min_width = 420
        content_width = max(grid_w, legend_min_width)

        width = padding * 2 + content_width
        height = padding * 2 + header_h + grid_h + legend_h

        img = Image.new("RGB", (width, height), bg)
        draw = ImageDraw.Draw(img)

        # Draw title
        title = "Pathfinding Challenge"
        bbox = draw.textbbox((0, 0), title, font=title_font)
        tw = bbox[2] - bbox[0]
        draw.text(
            (width // 2 - tw // 2, padding),
            title,
            fill=(40, 80, 120),
            font=title_font,
        )

        # Draw subtitle
        subtitle = f"Find the shortest path from START to GOAL ({rows}×{cols} maze)"
        bbox = draw.textbbox((0, 0), subtitle, font=small_font)
        tw = bbox[2] - bbox[0]
        draw.text(
            (width // 2 - tw // 2, padding + 32),
            subtitle,
            fill=(100, 100, 100),
            font=small_font,
        )

        # Draw instruction
        instruction = "Move: Up, Down, Left, Right only"
        bbox = draw.textbbox((0, 0), instruction, font=small_font)
        tw = bbox[2] - bbox[0]
        draw.text(
            (width // 2 - tw // 2, padding + 50),
            instruction,
            fill=(120, 120, 120),
            font=small_font,
        )

        # Grid area (center horizontally if needed)
        grid_y = padding + header_h
        grid_x = padding + (content_width - grid_w) // 2

        # Draw grid background with shadow
        draw.rectangle(
            [grid_x + 3, grid_y + 3, grid_x + grid_w + 3, grid_y + grid_h + 3],
            fill=(150, 150, 150),
        )
        draw.rectangle(
            [grid_x, grid_y, grid_x + grid_w, grid_y + grid_h],
            fill=(200, 200, 200),
            outline=(100, 100, 100),
            width=2,
        )

        # Draw cells
        for r in range(rows):
            for c in range(cols):
                x = grid_x + c * cell_px
                y = grid_y + r * cell_px

                cell = matrix[r][c] if r < len(matrix) and c < len(matrix[r]) else "O"

                if cell == "X":
                    # Blocked cell - dark wall with texture
                    draw.rectangle(
                        [x, y, x + cell_px, y + cell_px],
                        fill=(50, 50, 50),
                        outline=(30, 30, 30),
                        width=1,
                    )
                    # Add brick texture
                    for i in range(0, cell_px, 10):
                        draw.line(
                            [(x, y + i), (x + cell_px, y + i)],
                            fill=(60, 60, 60),
                            width=1,
                        )
                        draw.line(
                            [(x + i, y), (x + i, y + cell_px)],
                            fill=(60, 60, 60),
                            width=1,
                        )
                else:
                    # Open cell - light blue walkable path
                    draw.rectangle(
                        [x, y, x + cell_px, y + cell_px],
                        fill=(210, 230, 240),
                        outline=(150, 170, 180),
                        width=1,
                    )

                    if cell == "*":
                        # Start - green circle
                        cx, cy = x + cell_px // 2, y + cell_px // 2
                        r_circle = int(cell_px * 0.35)

                        # Shadow
                        draw.ellipse(
                            [
                                cx - r_circle + 2,
                                cy - r_circle + 2,
                                cx + r_circle + 2,
                                cy + r_circle + 2,
                            ],
                            fill=(100, 150, 100),
                        )
                        # Circle
                        draw.ellipse(
                            [
                                cx - r_circle,
                                cy - r_circle,
                                cx + r_circle,
                                cy + r_circle,
                            ],
                            fill=(80, 200, 80),
                            outline=(50, 150, 50),
                            width=2,
                        )
                        # Highlight
                        draw.ellipse(
                            [
                                cx - r_circle // 3,
                                cy - r_circle // 3,
                                cx - r_circle // 3 + r_circle // 2,
                                cy - r_circle // 3 + r_circle // 2,
                            ],
                            fill=(120, 255, 120),
                        )
                        # Text
                        text = "START"
                        bbox = draw.textbbox((0, 0), text, font=small_font)
                        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                        draw.text(
                            (cx - tw // 2, cy - th // 2),
                            text,
                            fill=(255, 255, 255),
                            font=small_font,
                        )

                    elif cell == "#":
                        # Goal - red flag
                        cx, cy = x + cell_px // 2, y + cell_px // 2

                        # Flag pole
                        draw.rectangle(
                            [
                                cx - 2,
                                cy - int(cell_px * 0.3),
                                cx + 2,
                                cy + int(cell_px * 0.3),
                            ],
                            fill=(100, 100, 100),
                        )
                        # Flag
                        flag_points = [
                            (cx + 2, cy - int(cell_px * 0.3)),
                            (cx + int(cell_px * 0.35), cy - int(cell_px * 0.15)),
                            (cx + 2, cy),
                        ]
                        draw.polygon(flag_points, fill=(230, 30, 30))
                        draw.line(
                            flag_points + [flag_points[0]], fill=(180, 20, 20), width=1
                        )

                        # Text below
                        text = "GOAL"
                        bbox = draw.textbbox((0, 0), text, font=small_font)
                        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                        draw.text(
                            (cx - tw // 2, cy + int(cell_px * 0.15)),
                            text,
                            fill=(180, 30, 30),
                            font=small_font,
                        )

        # Draw legend (centered)
        legend_y = grid_y + grid_h + 25

        # Fixed spacing for 4 items
        item_spacing = 105
        legend_total_width = item_spacing * 4
        legend_start_x = (width - legend_total_width) // 2
        legend_item_y = legend_y + 10

        # START
        cx = legend_start_x + item_spacing * 0 + 15
        cy = legend_item_y + 15
        draw.ellipse(
            [cx - 12, cy - 12, cx + 12, cy + 12],
            fill=(80, 200, 80),
            outline=(50, 150, 50),
            width=2,
        )
        draw.text(
            (cx + 20, legend_item_y + 10), "START", fill=(60, 60, 60), font=small_font
        )

        # GOAL
        cx = legend_start_x + item_spacing * 1 + 15
        cy = legend_item_y + 15
        draw.rectangle([cx - 2, cy - 10, cx + 2, cy + 10], fill=(100, 100, 100))
        flag_points = [(cx + 2, cy - 10), (cx + 15, cy - 3), (cx + 2, cy + 4)]
        draw.polygon(flag_points, fill=(230, 30, 30))
        draw.text(
            (cx + 20, legend_item_y + 10), "GOAL", fill=(60, 60, 60), font=small_font
        )

        # Open Path
        cx = legend_start_x + item_spacing * 2 + 15
        cy = legend_item_y + 15
        draw.rectangle(
            [cx - 12, cy - 12, cx + 12, cy + 12],
            fill=(210, 230, 240),
            outline=(150, 170, 180),
            width=1,
        )
        draw.text(
            (cx + 20, legend_item_y + 10),
            "Open Path",
            fill=(60, 60, 60),
            font=small_font,
        )

        # Wall
        cx = legend_start_x + item_spacing * 3 + 15
        cy = legend_item_y + 15
        draw.rectangle(
            [cx - 12, cy - 12, cx + 12, cy + 12],
            fill=(50, 50, 50),
            outline=(30, 30, 30),
            width=1,
        )
        for i in range(-12, 13, 6):
            draw.line(
                [(cx - 12, cy + i), (cx + 12, cy + i)], fill=(60, 60, 60), width=1
            )
            draw.line(
                [(cx + i, cy - 12), (cx + i, cy + 12)], fill=(60, 60, 60), width=1
            )
        draw.text(
            (cx + 20, legend_item_y + 10), "Wall", fill=(60, 60, 60), font=small_font
        )

        # Draw outer border
        draw.rectangle(
            [2, 2, width - 3, height - 3],
            outline=(120, 140, 160),
            width=3,
        )

        return img

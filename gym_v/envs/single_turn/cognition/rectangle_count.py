"""Rectangle Count single-turn environment backed by reasoning-gym."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from reasoning_gym.factory import create_dataset

from gym_v import Env, Observation, get_logger

logger = get_logger()


class RectangleCountEnv(Env):
    # Meta: source=ReasoningGym, category=cognition, turn=single
    """Rectangle Count puzzle using reasoning-gym's dataset.

    The player counts rectangles in an ASCII grid.

    Args:
        dataset_kwargs: Configuration parameters for the reasoning-gym dataset
        cell_px: Size of each cell in pixels for rendering
        padding: Padding around the grid in pixels for rendering
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    def __init__(
        self,
        dataset_kwargs: dict[str, Any] | None = None,
        cell_px: int = 20,
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
        self._grid_str: str | None = None
        self._oracle_answer: str | None = None

    @property
    def description(self) -> str:
        """Return description for Rectangle Count puzzle.

        Original reasoning-gym question format:
        ```
        Your task is to count how many rectangles are present in an ASCII grid.

        Single rectangles are outlined with a '#', overlapping rectangles
        (max 2) are shown with '█'.

        Your output should be a single number, representing the total count
        of rectangles.

        Now, it's your turn. How many rectangles do you see in the grid below?
        [ASCII grid with # and █ characters]
        ```

        Original reasoning-gym answer format:
        ```
        2
        ```
        (Single integer count)
        """
        return dedent("""
            How many rectangles do you see in the grid below?
            Your output should be only a single number, representing the total count
        of rectangles.
        """).strip()

    def _make_dataset(self, *, seed: int | None):
        kwargs = self._dataset_kwargs.copy()
        if seed is not None and "seed" not in kwargs:
            kwargs["seed"] = seed
        if "size" not in kwargs:
            kwargs["size"] = 500

        return create_dataset("rectangle_count", **kwargs)

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

        # Extract grid from question
        self._grid_str = self._extract_grid_from_question(self._entry["question"])

        logger.info("Reset ReasoningGym Rectangle Count.")

        obs = Observation(
            image=self.render(),
            text=None,
            metadata={
                "state_text": self._grid_str,
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

    def _extract_grid_from_question(self, question: str) -> str:
        """Extract the ASCII grid from the question.

        The grid starts after 'in the grid below?' and includes all lines
        that could be part of the grid (spaces and # and █).
        """
        lines = question.split("\n")
        # Find the start marker
        start_idx = 0
        for i, line in enumerate(lines):
            if "grid below?" in line.lower():
                start_idx = i + 1
                break

        # Collect all grid lines after the marker
        grid_lines = lines[start_idx:]
        return "\n".join(grid_lines)

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
        return self._render_rectangle_grid(
            self._grid_str, cell_px=self._cell_px, padding=self._padding
        )

    def _render_rectangle_grid(
        self,
        grid_str: str,
        cell_px: int = 20,
        padding: int = 40,
        bg: tuple[int, int, int] = (250, 250, 245),
        border_color: tuple[int, int, int] = (70, 130, 220),
        overlap_color: tuple[int, int, int] = (180, 100, 220),
    ) -> Image.Image:
        if not grid_str:
            return Image.new("RGB", (400, 400), bg)

        # Load fonts
        font_path = self.assets_dir / "DejaVuSans.ttf"
        if font_path.exists():
            title_font = ImageFont.truetype(str(font_path), 20)
            small_font = ImageFont.truetype(str(font_path), 12)
        else:
            logger.warning(f"Font file not found: {font_path}, using default font")
            title_font = ImageFont.load_default()
            small_font = title_font

        lines = [line for line in grid_str.split("\n") if line.strip()]
        rows = len(lines)
        cols = max(len(line) for line in lines) if lines else 0

        # Layout calculations
        header_h = 80
        legend_h = 50
        grid_w = cell_px * cols
        grid_h = cell_px * rows

        width = padding * 2 + grid_w
        height = padding * 2 + header_h + grid_h + legend_h

        img = Image.new("RGB", (width, height), bg)
        draw = ImageDraw.Draw(img)

        # Draw title
        title = "Rectangle Count Challenge"
        bbox = draw.textbbox((0, 0), title, font=title_font)
        tw = bbox[2] - bbox[0]
        draw.text(
            (width // 2 - tw // 2, padding),
            title,
            fill=(40, 80, 120),
            font=title_font,
        )

        # Draw subtitle
        subtitle = "Count all rectangles in the grid below"
        bbox = draw.textbbox((0, 0), subtitle, font=small_font)
        tw = bbox[2] - bbox[0]
        draw.text(
            (width // 2 - tw // 2, padding + 30),
            subtitle,
            fill=(100, 100, 100),
            font=small_font,
        )

        # Draw grid background
        grid_y = padding + header_h
        grid_x = padding
        draw.rectangle(
            [grid_x - 2, grid_y - 2, grid_x + grid_w + 2, grid_y + grid_h + 2],
            fill=(255, 255, 255),
            outline=(150, 150, 150),
            width=2,
        )

        # Draw grid cells
        for r, line in enumerate(lines):
            for c, char in enumerate(line):
                x = grid_x + c * cell_px
                y = grid_y + r * cell_px

                if char == "#":
                    # Rectangle border - draw with nice blue
                    draw.rectangle(
                        [x, y, x + cell_px - 1, y + cell_px - 1],
                        fill=border_color,
                    )
                    # Add subtle highlight
                    draw.rectangle(
                        [x + 2, y + 2, x + cell_px // 3, y + cell_px // 3],
                        fill=(120, 170, 255),
                    )
                elif char == "█":
                    # Overlap area - draw with purple
                    draw.rectangle(
                        [x, y, x + cell_px - 1, y + cell_px - 1],
                        fill=overlap_color,
                    )
                    # Add subtle highlight
                    draw.rectangle(
                        [x + 2, y + 2, x + cell_px // 3, y + cell_px // 3],
                        fill=(210, 140, 255),
                    )
                else:
                    # Empty space - draw subtle grid
                    draw.rectangle(
                        [x, y, x + cell_px - 1, y + cell_px - 1],
                        outline=(235, 235, 235),
                        width=1,
                    )

        # Draw legend (only overlap symbol)
        legend_y = grid_y + grid_h + 20
        legend_x = padding + 20

        # Overlap symbol
        draw.rectangle(
            [legend_x, legend_y, legend_x + cell_px - 2, legend_y + cell_px - 2],
            fill=overlap_color,
        )
        draw.rectangle(
            [
                legend_x + 2,
                legend_y + 2,
                legend_x + cell_px // 3,
                legend_y + cell_px // 3,
            ],
            fill=(210, 140, 255),
        )
        draw.text(
            (legend_x + cell_px + 10, legend_y + 2),
            "█ = Two rectangles overlap (count as 2)",
            fill=(60, 60, 60),
            font=small_font,
        )

        # Draw outer border
        draw.rectangle(
            [2, 2, width - 3, height - 3],
            outline=(120, 140, 160),
            width=3,
        )

        return img

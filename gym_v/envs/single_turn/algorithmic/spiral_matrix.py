"""Spiral Matrix single-turn environment backed by reasoning-gym."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from reasoning_gym.factory import create_dataset

from gym_v import Env, Observation, get_logger

logger = get_logger()


class SpiralMatrixEnv(Env):
    # Meta: source=ReasoningGym, category=algorithmic, turn=single
    """Spiral Matrix puzzle using reasoning-gym's dataset.

    The player outputs elements in spiral order starting from top-left.

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
        self._matrix: list[list[int]] | None = None
        self._oracle_answer: str | None = None

    @property
    def description(self) -> str:
        """Return description for Spiral Matrix puzzle.

        Original reasoning-gym question format:
        ```
        Given a matrix, your job is to generate a list of elements in spiral order,
        starting from the top-left element.

        The spiral order is clockwise, starting from the top-left corner. More precisely:
        - Start from the top-left corner and move right.
        - Move down towards the bottom-right corner.
        - Move left towards the bottom-left corner.
        - Move up towards the top-right corner.
        - Repeat the steps for the inner elements of the matrix until every entry is visited.

        Your output should be a space-separated list of integers, e.g. 1 2 3 4 5 6

        For the matrix below, what is the list of elements in spiral order?
        3 1 3
        2 4 9
        1 0 8
        ```

        Original reasoning-gym answer format:
        ```
        3 1 3 9 8 0 1 2 4
        ```
        (Space-separated integers in spiral order)
        """
        return dedent("""
            Generate a list of elements in spiral order, starting from the top-left element.

            The spiral order is clockwise:
            - Start from the top-left corner and move right.
            - Move down towards the bottom-right corner.
            - Move left towards the bottom-left corner.
            - Move up towards the top-left corner.
            - Repeat for inner elements until every entry is visited.

            For the matrix image, what is the list of elements in spiral order?
            Output format: Space-separated list of integers, e.g. 1 2 3 4 5 6
        """).strip()

    def _make_dataset(self, *, seed: int | None):
        kwargs = self._dataset_kwargs.copy()
        if "seed" not in kwargs:
            kwargs["seed"] = seed if seed is not None else int(self.np_random.integers(0, 2**31))
        if "size" not in kwargs:
            kwargs["size"] = 500

        return create_dataset("spiral_matrix", **kwargs)

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

        logger.info("Reset ReasoningGym Spiral Matrix.")

        # obs.text = matrix as text
        matrix_text = self._format_matrix(self._matrix)

        obs = Observation(
            image=self.render(),
            text=None,
            metadata={
                "state_text": matrix_text,
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
        try:
            reward = self._dataset.score_answer(answer=answer, entry=self._entry)
        except Exception as e:
            logger.warning(f"score_answer failed for {type(self).__name__}: {e}, answer={str(answer)[:200]}")
            reward = 0.0

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

    def _format_matrix(self, matrix: list[list[int]]) -> str:
        """Format matrix as text."""
        if not matrix:
            return ""
        return "\n".join(" ".join(str(x) for x in row) for row in matrix)

    def render(self) -> Image.Image | list[Image.Image] | None:
        return self._render_matrix_grid(
            self._matrix, cell_px=self._cell_px, padding=self._padding
        )

    def _render_matrix_grid(
        self,
        matrix: list[list[int]],
        cell_px: int = 48,
        padding: int = 24,
        bg: tuple[int, int, int] = (250, 250, 250),
        cell_bg: tuple[int, int, int] = (255, 255, 255),
        grid_color: tuple[int, int, int] = (100, 100, 100),
        text_color: tuple[int, int, int] = (30, 30, 30),
    ) -> Image.Image:
        if not matrix:
            return Image.new("RGB", (200, 200), bg)

        rows = len(matrix)
        cols = len(matrix[0]) if matrix else 0
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

        # Draw cells
        for r in range(rows):
            for c in range(cols):
                x = padding + c * cell_px
                y = padding + r * cell_px

                draw.rectangle(
                    [x, y, x + cell_px - 1, y + cell_px - 1],
                    fill=cell_bg,
                    outline=grid_color,
                    width=1,
                )

                # Draw number
                val = matrix[r][c] if r < len(matrix) and c < len(matrix[r]) else 0
                txt = str(val)
                bbox = draw.textbbox((0, 0), txt, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                draw.text(
                    (x + cell_px // 2 - tw // 2, y + cell_px // 2 - th // 2),
                    txt,
                    fill=text_color,
                    font=font,
                )

        # Draw outer border
        draw.rectangle(
            [padding - 1, padding - 1, width - padding, height - padding],
            outline=(60, 60, 60),
            width=2,
        )

        return img

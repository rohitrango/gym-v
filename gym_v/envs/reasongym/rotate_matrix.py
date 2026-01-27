"""Rotate Matrix single-turn environment backed by reasoning-gym."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from reasoning_gym.factory import create_dataset

from gym_v import Env, Observation, get_logger

logger = get_logger()


class ReasoningGymRotateMatrixEnv(Env):
    """Rotate Matrix puzzle using reasoning-gym's dataset.

    The player rotates a matrix by specified degrees clockwise.

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
        self._num_rotations: int = 0
        self._oracle_answer: str | None = None

    @property
    def description(self) -> str:
        """Return description for Rotate Matrix puzzle.

        Original reasoning-gym question format:
        ```
        Given a square matrix, your job is to rotate it clockwise.

        Your output should be a matrix in the same format as the input.

        Rotate the matrix below by 540 degrees clockwise:
        0 4 3
        3 2 1
        8 1 9
        ```

        Original reasoning-gym answer format:
        ```
        9 1 8
        1 2 3
        3 4 0
        ```
        (Matrix with space-separated numbers, newlines between rows)
        """
        degrees = self._num_rotations * 90

        return dedent(f"""
            Given a square matrix, your job is to rotate it clockwise by {degrees} degrees.

            Output format: Matrix with space-separated numbers, newlines between rows.
            Example for a 2x2 result:
            1 2
            3 4
        """).strip()

    def _make_dataset(self, *, seed: int | None):
        kwargs = self._dataset_kwargs.copy()
        if seed is not None and "seed" not in kwargs:
            kwargs["seed"] = seed
        if "size" not in kwargs:
            kwargs["size"] = 500

        return create_dataset("rotate_matrix", **kwargs)

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
        self._num_rotations = self._metadata.get("num_rotations", 0)

        logger.info("Reset ReasoningGym Rotate Matrix.")

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

    def _format_matrix(self, matrix: list[list[int]]) -> str:
        if not matrix:
            return ""
        return "\n".join(" ".join(str(x) for x in row) for row in matrix)

    def render(self) -> Image.Image | list[Image.Image] | None:
        return self._render_matrix_grid(
            self._matrix,
            self._num_rotations,
            cell_px=self._cell_px,
            padding=self._padding,
        )

    def _render_matrix_grid(
        self,
        matrix: list[list[int]],
        num_rotations: int,
        cell_px: int = 48,
        padding: int = 24,
        bg: tuple[int, int, int] = (250, 250, 250),
        cell_bg: tuple[int, int, int] = (240, 240, 250),
        grid_color: tuple[int, int, int] = (150, 150, 150),
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
            font = ImageFont.truetype(str(font_path), int(cell_px * 0.5))
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

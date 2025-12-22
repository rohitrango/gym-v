"""Survo single-turn environment backed by reasoning-gym."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from reasoning_gym.factory import create_dataset

from gym_v import Env, Observation, get_logger

logger = get_logger()


class ReasoningGymSurvoEnv(Env):
    """Survo puzzle using reasoning-gym's Survo dataset.

    In Survo, the last element of each row and column equals the sum of the
    other elements in that row or column. The player fills in zeros with
    candidate numbers.

    Args:
        dataset_kwargs: Configuration parameters for the reasoning-gym dataset
        cell_px: Size of each cell in pixels for rendering
        padding: Padding around the grid in pixels for rendering
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    def __init__(
        self,
        dataset_kwargs: dict[str, Any] | None = None,
        cell_px: int = 64,
        padding: int = 24,
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
        self._puzzle: list[list[int]] | None = None
        self._candidate_numbers: list[int] | None = None
        self._oracle_answer: str | None = None

    @property
    def description(self) -> str:
        n = self._metadata.get("board_size", 4) if self._metadata else 4
        candidates = self._candidate_numbers if self._candidate_numbers else []
        return dedent(f"""
            Survo Puzzle:

            Given a {n}x{n} matrix where the last element of each row and column equals
            the sum of the other elements in that row or column.

            Some elements are replaced with 0. You have a set of numbers {candidates}
            that can be filled into the 0 positions to satisfy the rules.
            Each number can only be used once.

            Please fill in the matrix. Output the completed matrix with spaces
            separating each number within a row, and newlines separating rows.
        """).strip()

    def _make_dataset(self, *, seed: int | None):
        kwargs = self._dataset_kwargs.copy()
        if seed is not None and "seed" not in kwargs:
            kwargs["seed"] = seed
        if "size" not in kwargs:
            kwargs["size"] = 500

        return create_dataset("survo", **kwargs)

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
        self._puzzle = self._metadata.get("puzzle", [])
        self._candidate_numbers = self._metadata.get("candidate_numbers", [])

        logger.info("Reset ReasoningGym Survo.")

        # obs.text = only the board state (caption)
        board_text = self._board_to_string(self._puzzle)

        obs = Observation(
            image=self.render(),
            text=board_text,
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

    def _board_to_string(self, board: list[list[int]]) -> str:
        """Convert board to string representation."""
        return "\n".join(" ".join(str(x) for x in row) for row in board)

    def render(self) -> Image.Image:
        return self._render_survo_grid(
            self._puzzle, cell_px=self._cell_px, padding=self._padding
        )

    def _render_survo_grid(
        self,
        puzzle: list[list[int]],
        cell_px: int = 64,
        padding: int = 24,
        bg: tuple[int, int, int] = (250, 250, 250),
        fg: tuple[int, int, int] = (20, 20, 20),
        empty_fg: tuple[int, int, int] = (150, 150, 150),
        grid_color: tuple[int, int, int] = (30, 30, 30),
        sum_bg: tuple[int, int, int] = (200, 220, 240),
    ) -> Image.Image:
        if not puzzle:
            return Image.new("RGB", (200, 200), bg)

        n_rows = len(puzzle)
        n_cols = len(puzzle[0]) if puzzle else 0

        width = padding * 2 + cell_px * n_cols
        height = padding * 2 + cell_px * n_rows
        img = Image.new("RGB", (width, height), bg)
        draw = ImageDraw.Draw(img)

        font_path = self.assets_dir / "DejaVuSans.ttf"
        if font_path.exists():
            font = ImageFont.truetype(str(font_path), int(cell_px * 0.4))
        else:
            logger.warning(f"Font file not found: {font_path}, using default font")
            font = ImageFont.load_default()

        # Draw cells
        for r in range(n_rows):
            for c in range(n_cols):
                x = padding + c * cell_px
                y = padding + r * cell_px

                # Last row/col are sum cells
                is_sum_cell = (r == n_rows - 1) or (c == n_cols - 1)
                cell_bg = sum_bg if is_sum_cell else bg

                draw.rectangle(
                    [x, y, x + cell_px - 1, y + cell_px - 1],
                    fill=cell_bg,
                    outline=grid_color,
                    width=1,
                )

                v = puzzle[r][c] if r < len(puzzle) and c < len(puzzle[r]) else 0
                if v == 0 and not is_sum_cell:
                    # Empty cell, draw placeholder
                    txt = "_"
                    color = empty_fg
                else:
                    txt = str(v)
                    color = fg

                bbox = draw.textbbox((0, 0), txt, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                cx = x + cell_px // 2
                cy = y + cell_px // 2
                draw.text((cx - tw // 2, cy - th // 2), txt, fill=color, font=font)

        # Draw border
        draw.rectangle(
            [padding - 2, padding - 2, width - padding + 1, height - padding + 1],
            outline=(60, 60, 60),
            width=2,
        )

        return img

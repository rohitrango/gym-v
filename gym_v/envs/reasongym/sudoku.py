"""Sudoku single-turn environment backed by reasoning-gym."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from reasoning_gym.factory import create_dataset

from gym_v import Env, Observation, get_logger

logger = get_logger()


class ReasoningGymSudokuEnv(Env):
    """Sudoku number placement puzzle using reasoning-gym's Sudoku dataset.

    The player fills a 9x9 grid with digits 1-9 such that each row, column, and
    3x3 subgrid contains all digits without repetition.

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
        self._oracle_answer: str | None = None

    @property
    def description(self) -> str:
        return dedent("""
            Solve this Sudoku puzzle.
            Respond with only your answer, formatted as the puzzle, a 9x9 grid with numbers separated by spaces, and rows separated by newlines.
        """).strip()

    def _make_dataset(self, *, seed: int | None):
        kwargs = self._dataset_kwargs
        # reasoning-gym SudokuConfig supports seed/size
        if seed is not None and "seed" not in kwargs:
            kwargs["seed"] = seed
        if "size" not in kwargs:
            kwargs["size"] = 500

        return create_dataset("sudoku", **kwargs)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)
        self._seed = seed

        self._dataset = self._make_dataset(seed=self._seed)
        self._entry_idx = int(self.np_random.integers(0, len(self._dataset)))
        self._entry = self._dataset[self._entry_idx]

        # reasoning-gym entries are expected to always provide these fields.
        self._oracle_answer = self._entry["answer"]
        self._metadata = self._entry.get("metadata", {})
        self._puzzle = self._metadata.get("puzzle")

        logger.info("Reset ReasoningGym Sudoku.")

        obs = Observation(
            image=self.render(),
            text=self._dataset._board_to_string(self._puzzle),
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

    def render(self) -> Image.Image:
        return self._render_sudoku_grid(
            self._puzzle, cell_px=self._cell_px, padding=self._padding
        )

    def _render_sudoku_grid(
        self,
        puzzle: list[list[int]],
        cell_px: int = 64,
        padding: int = 24,
        bg: tuple[int, int, int] = (250, 250, 250),
        fg: tuple[int, int, int] = (20, 20, 20),
        grid: tuple[int, int, int] = (30, 30, 30),
    ) -> Image.Image:
        n = 9
        size = padding * 2 + cell_px * n
        img = Image.new("RGB", (size, size), bg)
        draw = ImageDraw.Draw(img)

        font_path = self.assets_dir / "DejaVuSans.ttf"
        if font_path.exists():
            font = ImageFont.truetype(str(font_path), int(cell_px * 0.5))
        else:
            logger.warning(f"Font file not found: {font_path}, using default font")
            font = ImageFont.load_default()

        # Cells background
        draw.rectangle([0, 0, size - 1, size - 1], outline=grid, width=2)

        # Grid lines (thick lines every 3)
        for i in range(n + 1):
            x = padding + i * cell_px
            y = padding + i * cell_px
            w = 3 if i % 3 == 0 else 1
            draw.line([(x, padding), (x, padding + n * cell_px)], fill=grid, width=w)
            draw.line([(padding, y), (padding + n * cell_px, y)], fill=grid, width=w)

        # Digits
        for r in range(n):
            for c in range(n):
                v = int(puzzle[r][c])
                if v <= 0:
                    continue
                txt = str(v)
                bbox = draw.textbbox((0, 0), txt, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                cx = padding + c * cell_px + cell_px // 2
                cy = padding + r * cell_px + cell_px // 2
                draw.text((cx - tw // 2, cy - th // 2), txt, fill=fg, font=font)

        return img

"""Sudoku single-turn environment backed by reasoning-gym."""

from __future__ import annotations

from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation, get_logger

logger = get_logger()


def _render_sudoku_grid(
    puzzle: list[list[int]],
    *,
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

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", int(cell_px * 0.5))
    except Exception:
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


class ReasoningGymSudokuEnv(Env):
    """Single-turn Sudoku environment.

    reset(): samples one Sudoku puzzle and returns (image=grid, text=question)
    step(action): scores the submitted full-grid answer using reasoning-gym scoring, then terminates.
    """

    def __init__(
        self,
        dataset_kwargs: dict[str, Any] | None = None,
        include_oracle_in_info: bool = False,
        cell_px: int = 64,
        padding: int = 24,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._dataset_kwargs = dataset_kwargs or {}
        self._include_oracle_in_info = include_oracle_in_info
        self._cell_px = cell_px
        self._padding = padding

        self._dataset = None
        self._entry: dict[str, Any] | None = None
        self._entry_idx: int | None = None
        self._puzzle: list[list[int]] | None = None
        self._question: str | None = None
        self._oracle_answer: str | None = None

    @property
    def description(self) -> str:
        return (
            "Solve the Sudoku. Reply with ONLY your answer, formatted as a 9x9 grid: "
            "numbers separated by spaces, rows separated by newlines."
        )

    def _make_dataset(self, *, seed: int | None):
        import reasoning_gym
        from reasoning_gym.factory import create_dataset

        kwargs = dict(self._dataset_kwargs)
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

        logger.info("Reset ReasoningGym Sudoku.")
        self._dataset = self._make_dataset(seed=seed)
        ds_len = len(self._dataset)
        if ds_len <= 0:
            raise ValueError(f"reasoning_gym sudoku dataset length is {ds_len}")

        self._entry_idx = int(self.np_random.integers(0, ds_len))
        self._entry = self._dataset[self._entry_idx]

        # reasoning-gym entries are expected to always provide these fields.
        self._question = self._entry["question"]
        self._oracle_answer = self._entry["answer"]
        md = self._entry.get("metadata", {})
        self._puzzle = md.get("puzzle")
        if not isinstance(self._puzzle, list):
            raise ValueError("reasoning_gym sudoku entry missing metadata['puzzle']")

        obs = Observation(
            image=self.render(),
            text=self._question,
            metadata={
                **md,
                "reasoning_gym_dataset": "sudoku",
                "reasoning_gym_index": self._entry_idx,
            },
        )
        info = {"dataset_name": "sudoku", "dataset_index": self._entry_idx}
        return obs, info

    def inner_step(
        self, action: str
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        if self._dataset is None or self._entry is None:
            raise RuntimeError("Call reset() before step().")

        answer = action if isinstance(action, str) else str(action)
        reward = float(self._dataset.score_answer(answer=answer, entry=self._entry))

        info: dict[str, Any] = {
            "dataset_name": "sudoku",
            "dataset_index": self._entry_idx,
            "score": reward,
        }
        if self._include_oracle_in_info:
            info["oracle_answer"] = self._oracle_answer

        obs_text = (
            f"{self._question}\n\n"
            f"Your answer:\n{answer}\n\n"
            f"Score: {reward}"
        )
        obs = Observation(
            image=self.render(),
            text=obs_text,
            metadata={
                **(self._entry.get("metadata") or {}),
                "reasoning_gym_dataset": "sudoku",
                "reasoning_gym_index": self._entry_idx,
            },
        )
        return obs, reward, True, False, info

    def render(self) -> Image.Image:
        if self._puzzle is None:
            # blank grid if called before reset
            empty = [[0] * 9 for _ in range(9)]
            return _render_sudoku_grid(empty, cell_px=self._cell_px, padding=self._padding)
        return _render_sudoku_grid(self._puzzle, cell_px=self._cell_px, padding=self._padding)



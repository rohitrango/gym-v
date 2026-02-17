"""Mini Sudoku (4x4) single-turn environment backed by reasoning-gym."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from reasoning_gym.factory import create_dataset

from gym_v import Env, Observation, get_logger

logger = get_logger()


class MiniSudokuEnv(Env):
    # Meta: source=ReasoningGym, category=logic, turn=single
    # Difficulty: grid_size=4x4
    """Mini Sudoku (4x4) puzzle using reasoning-gym's Mini Sudoku dataset.

    The player fills a 4x4 grid with digits 1-4 such that each row, column, and
    2x2 subgrid contains all digits without repetition.

    Args:
        dataset_kwargs: Configuration parameters for the reasoning-gym dataset
        cell_px: Size of each cell in pixels for rendering
        padding: Padding around the grid in pixels for rendering
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    def __init__(
        self,
        dataset_kwargs: dict[str, Any] | None = None,
        cell_px: int = 80,
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
        self._puzzle: list[list[int]] | None = None
        self._oracle_answer: str | None = None

    @property
    def description(self) -> str:
        """Return description for Mini Sudoku puzzle.

        Original reasoning-gym question format:
        ```
        In 4x4 Mini Sudoku:
        - Each row must contain each number from 1-4 exactly once
        - Each column must contain each number 1-4 exactly once
        - Each 2x2 subgrid must contain each number 1-4 exactly once
        Solve this 4x4 Mini Sudoku puzzle:
        _ 2 3 4
        3 _ 1 2
        2 1 4 _
        4 3 _ 1
        Format your response as the puzzle above, with spaces separating each number
        within a row, and newlines separating rows.
        ```

        Original reasoning-gym answer format:
        ```
        1 2 3 4
        3 4 1 2
        2 1 4 3
        4 3 2 1
        ```
        (4x4 grid, numbers 1-4 separated by spaces, rows separated by newlines)
        """
        return dedent("""
            Solve this 4x4 Mini Sudoku puzzle.

            In the image:
            - Cells with numbers are pre-filled clues
            - Empty cells need to be filled with digits 1-4

            Rules:
            - Each row must contain digits 1-4 exactly once
            - Each column must contain digits 1-4 exactly once
            - Each 2x2 subgrid must contain digits 1-4 exactly once

            Output format: A 4x4 grid with numbers separated by spaces within rows,
            and newlines separating rows. Example:
            1 2 3 4
            3 4 1 2
            2 1 4 3
            4 3 2 1
        """).strip()

    def _make_dataset(self, *, seed: int | None):
        kwargs = self._dataset_kwargs.copy()
        if seed is not None and "seed" not in kwargs:
            kwargs["seed"] = seed
        if "size" not in kwargs:
            kwargs["size"] = 500

        return create_dataset("mini_sudoku", **kwargs)

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
        self._puzzle = self._metadata.get("puzzle", [])

        logger.info("Reset ReasoningGym Mini Sudoku.")

        # obs.text = only the board state (caption)
        board_text = self._board_to_string(self._puzzle)

        obs = Observation(
            image=self.render(),
            text=None,
            metadata={
                "state_text": board_text,
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

    def _board_to_string(self, board: list[list[int]]) -> str:
        """Convert board to string representation."""
        return "\n".join(
            " ".join(str(x) if x != 0 else "_" for x in row) for row in board
        )

    def render(self) -> Image.Image | list[Image.Image] | None:
        return self._render_mini_sudoku_grid(
            self._puzzle, cell_px=self._cell_px, padding=self._padding
        )

    def _render_mini_sudoku_grid(
        self,
        puzzle: list[list[int]],
        cell_px: int = 80,
        padding: int = 24,
        bg: tuple[int, int, int] = (250, 250, 250),
        fg: tuple[int, int, int] = (20, 20, 20),
        grid_color: tuple[int, int, int] = (30, 30, 30),
    ) -> Image.Image:
        if not puzzle:
            return Image.new("RGB", (200, 200), bg)

        n = 4
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
        draw.rectangle([0, 0, size - 1, size - 1], outline=grid_color, width=2)

        # Grid lines (thick lines every 2 for 2x2 subgrids)
        for i in range(n + 1):
            x = padding + i * cell_px
            y = padding + i * cell_px
            w = 3 if i % 2 == 0 else 1
            draw.line(
                [(x, padding), (x, padding + n * cell_px)], fill=grid_color, width=w
            )
            draw.line(
                [(padding, y), (padding + n * cell_px, y)], fill=grid_color, width=w
            )

        # Digits
        for r in range(n):
            for c in range(n):
                v = int(puzzle[r][c]) if r < len(puzzle) and c < len(puzzle[r]) else 0
                if v <= 0:
                    continue
                txt = str(v)
                bbox = draw.textbbox((0, 0), txt, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                cx = padding + c * cell_px + cell_px // 2
                cy = padding + r * cell_px + cell_px // 2
                draw.text((cx - tw // 2, cy - th // 2), txt, fill=fg, font=font)

        return img

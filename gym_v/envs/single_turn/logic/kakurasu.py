"""Kakurasu single-turn environment backed by reasoning-gym."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from reasoning_gym.factory import create_dataset

from gym_v import Env, Observation, get_logger

logger = get_logger()


class KakurasuEnv(Env):
    # Meta: source=ReasoningGym, category=logic, turn=single
    """Kakurasu puzzle using reasoning-gym's Kakurasu dataset.

    In Kakurasu (also known as Kukurasu), the player fills a grid with 1s and 0s
    such that the weighted sum of each row and column matches the given constraints.

    Args:
        dataset_kwargs: Configuration parameters for the reasoning-gym dataset
        cell_px: Size of each cell in pixels for rendering
        padding: Padding around the grid in pixels for rendering
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    def __init__(
        self,
        dataset_kwargs: dict[str, Any] | None = None,
        cell_px: int = 56,
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
        self._puzzle: list[list[int]] | None = None
        self._row_sums: list[int] | None = None
        self._col_sums: list[int] | None = None
        self._oracle_answer: str | None = None

    @property
    def description(self) -> str:
        """Return description for Kakurasu puzzle.

        Original reasoning-gym question format (one of several templates):
        ```
        You are given a {n_rows} x {n_cols} grid representing a Kukurasu puzzle.
        In this puzzle, you need to place 1s in the grid so that the weighted sum
        of each row and column matches the given constraints.
        The row sums are [3, 5, 7, 4] and the column sums are [6, 4, 5, 4].

        1. Rules:
          1. Each cell can contain either a 1 or an 0.
          2. The weight of a 1 in a row is its column position (1 to {n_cols}).
          3. The weight of a 1 in a column is its row position (1 to {n_rows}).
          4. The weighted sum of each row must match the corresponding row constraint.
          5. The weighted sum of each column must match the corresponding column constraint.
        2. Input:
        0 0 0 0
        0 0 0 0
        0 0 0 0
        0 0 0 0
        ```

        Original reasoning-gym answer format:
        ```
        1 0 1 0
        0 1 0 1
        1 1 0 0
        0 0 1 1
        ```
        (Grid with 0s and 1s, spaces between cells, newlines between rows)
        """
        n_rows = self._metadata.get("n_rows", 4) if self._metadata else 4
        n_cols = self._metadata.get("n_cols", 4) if self._metadata else 4

        return dedent(f"""
            Kakurasu Puzzle: Fill the {n_rows}x{n_cols} grid with 0s and 1s.

            In the image:
            - Each column has a weight equal to its position (column 1 = weight 1, column 2 = weight 2, etc.)
            - Each row has a weight equal to its position (row 1 = weight 1, row 2 = weight 2, etc.)
            - The target sum for each row is shown on the right side
            - The target sum for each column is shown at the bottom

            Rules:
            1. Fill each cell with 0 or 1.
            2. For each row: sum of (cell value x column weight) must equal the row's target.
            3. For each column: sum of (cell value x row weight) must equal the column's target.

            Output format: {n_rows} lines, each with {n_cols} numbers (0 or 1) separated by spaces.
            Example for a 3x3 grid:
            1 0 1
            0 1 0
            1 1 0
        """).strip()

    def _make_dataset(self, *, seed: int | None):
        kwargs = self._dataset_kwargs.copy()
        if "seed" not in kwargs:
            kwargs["seed"] = seed if seed is not None else int(self.np_random.integers(0, 2**31))
        if "size" not in kwargs:
            kwargs["size"] = 500

        return create_dataset("kakurasu", **kwargs)

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
        self._row_sums = self._metadata.get("row_sums", [])
        self._col_sums = self._metadata.get("col_sums", [])

        logger.info("Reset ReasoningGym Kakurasu.")

        # obs.text = only the board state (empty grid with constraints)
        board_text = self._format_puzzle_text()

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

    def _format_puzzle_text(self) -> str:
        """Format puzzle with row/col sums."""
        if not self._puzzle:
            return ""
        lines = []
        for row in self._puzzle:
            lines.append(" ".join(str(x) for x in row))
        return "\n".join(lines)

    def render(self) -> Image.Image | list[Image.Image] | None:
        return self._render_kakurasu_grid(
            self._puzzle,
            self._row_sums,
            self._col_sums,
            cell_px=self._cell_px,
            padding=self._padding,
        )

    def _render_kakurasu_grid(
        self,
        puzzle: list[list[int]],
        row_sums: list[int],
        col_sums: list[int],
        cell_px: int = 56,
        padding: int = 40,
        bg: tuple[int, int, int] = (250, 250, 250),
        fg: tuple[int, int, int] = (20, 20, 20),
        grid_color: tuple[int, int, int] = (30, 30, 30),
        header_bg: tuple[int, int, int] = (220, 230, 240),
        sum_bg: tuple[int, int, int] = (240, 220, 200),
    ) -> Image.Image:
        if not puzzle:
            return Image.new("RGB", (200, 200), bg)

        n_rows = len(puzzle)
        n_cols = len(puzzle[0]) if puzzle else 0

        # Extra space for row/col weights and sums
        header_px = cell_px  # For column weights (1,2,3...)
        sum_px = cell_px  # For row/col sums

        width = padding * 2 + header_px + n_cols * cell_px + sum_px
        height = padding * 2 + header_px + n_rows * cell_px + sum_px
        img = Image.new("RGB", (width, height), bg)
        draw = ImageDraw.Draw(img)

        font_path = self.assets_dir / "DejaVuSans.ttf"
        if font_path.exists():
            font = ImageFont.truetype(str(font_path), int(cell_px * 0.4))
            small_font = ImageFont.truetype(str(font_path), int(cell_px * 0.3))
        else:
            logger.warning(f"Font file not found: {font_path}, using default font")
            font = ImageFont.load_default()
            small_font = font

        # Draw column weight headers (1, 2, 3, ...)
        for c in range(n_cols):
            x = padding + header_px + c * cell_px
            y = padding
            draw.rectangle(
                [x, y, x + cell_px - 1, y + header_px - 1],
                fill=header_bg,
                outline=grid_color,
                width=1,
            )
            txt = str(c + 1)
            bbox = draw.textbbox((0, 0), txt, font=small_font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text(
                (x + cell_px // 2 - tw // 2, y + header_px // 2 - th // 2),
                txt,
                fill=fg,
                font=small_font,
            )

        # Draw row weight headers (1, 2, 3, ...)
        for r in range(n_rows):
            x = padding
            y = padding + header_px + r * cell_px
            draw.rectangle(
                [x, y, x + header_px - 1, y + cell_px - 1],
                fill=header_bg,
                outline=grid_color,
                width=1,
            )
            txt = str(r + 1)
            bbox = draw.textbbox((0, 0), txt, font=small_font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text(
                (x + header_px // 2 - tw // 2, y + cell_px // 2 - th // 2),
                txt,
                fill=fg,
                font=small_font,
            )

        # Draw column sums (at bottom)
        for c in range(n_cols):
            x = padding + header_px + c * cell_px
            y = padding + header_px + n_rows * cell_px
            draw.rectangle(
                [x, y, x + cell_px - 1, y + sum_px - 1],
                fill=sum_bg,
                outline=grid_color,
                width=1,
            )
            if col_sums and c < len(col_sums):
                txt = str(col_sums[c])
                bbox = draw.textbbox((0, 0), txt, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                draw.text(
                    (x + cell_px // 2 - tw // 2, y + sum_px // 2 - th // 2),
                    txt,
                    fill=fg,
                    font=font,
                )

        # Draw row sums (at right)
        for r in range(n_rows):
            x = padding + header_px + n_cols * cell_px
            y = padding + header_px + r * cell_px
            draw.rectangle(
                [x, y, x + sum_px - 1, y + cell_px - 1],
                fill=sum_bg,
                outline=grid_color,
                width=1,
            )
            if row_sums and r < len(row_sums):
                txt = str(row_sums[r])
                bbox = draw.textbbox((0, 0), txt, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                draw.text(
                    (x + sum_px // 2 - tw // 2, y + cell_px // 2 - th // 2),
                    txt,
                    fill=fg,
                    font=font,
                )

        # Draw main grid cells (empty - to be filled)
        for r in range(n_rows):
            for c in range(n_cols):
                x = padding + header_px + c * cell_px
                y = padding + header_px + r * cell_px

                draw.rectangle(
                    [x, y, x + cell_px - 1, y + cell_px - 1],
                    fill=bg,
                    outline=grid_color,
                    width=1,
                )

                # Draw a subtle "?" to indicate empty cell
                txt = "?"
                bbox = draw.textbbox((0, 0), txt, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                draw.text(
                    (x + cell_px // 2 - tw // 2, y + cell_px // 2 - th // 2),
                    txt,
                    fill=(200, 200, 200),
                    font=font,
                )

        # Draw corner cells (empty decorative)
        # Top-left corner
        draw.rectangle(
            [padding, padding, padding + header_px - 1, padding + header_px - 1],
            fill=(230, 230, 230),
            outline=grid_color,
            width=1,
        )
        # Bottom-right corner
        draw.rectangle(
            [
                padding + header_px + n_cols * cell_px,
                padding + header_px + n_rows * cell_px,
                width - padding - 1,
                height - padding - 1,
            ],
            fill=(230, 230, 230),
            outline=grid_color,
            width=1,
        )

        return img

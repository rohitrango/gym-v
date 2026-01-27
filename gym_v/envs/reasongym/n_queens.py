"""N Queens single-turn environment backed by reasoning-gym."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from reasoning_gym.factory import create_dataset

from gym_v import Env, Observation, get_logger

logger = get_logger()


class ReasoningGymNQueensEnv(Env):
    """N Queens puzzle using reasoning-gym's N Queens dataset.

    The player must place queens on a chess board such that no two queens
    attack each other (not in the same row, column, or diagonal).

    Args:
        dataset_kwargs: Configuration parameters for the reasoning-gym dataset
        cell_px: Size of each cell in pixels for rendering
        padding: Padding around the board in pixels for rendering
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    def __init__(
        self,
        dataset_kwargs: dict[str, Any] | None = None,
        cell_px: int = 64,
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
        self._puzzle: list[list[str]] | None = None
        self._oracle_answer: str | None = None

    @property
    def description(self) -> str:
        """Return description with current puzzle parameters.

        Original reasoning-gym question format:
        ```
        Your job is to complete an n x n chess board with n Queens in total, such that no two attack each other.

        No two queens attack each other if they are not in the same row, column, or diagonal.

        You can place a queen by replacing an underscore (_) with a Q.

        Your output should be also a board in the same format as the input, with queens placed on the board by replacing underscores with the letter Q.

        Given the below board of size {n} x {n} your job is to place {num_removed} queen(s) on the board such that no two queens attack each other.
        _ Q _ _ _ _ _ _
        _ _ _ _ Q _ _ _
        ...
        ```

        Original reasoning-gym answer format:
        ```
        _ Q _ _ _ _ _ _
        _ _ _ _ Q _ _ _
        _ _ _ _ _ _ Q _
        Q _ _ _ _ _ _ _
        _ _ Q _ _ _ _ _
        _ _ _ _ _ _ _ Q
        _ _ _ _ _ Q _ _
        _ _ _ Q _ _ _ _
        ```
        (Board with Q for queens, _ for empty, spaces between cells, newlines between rows)
        """
        n = len(self._puzzle) if self._puzzle else 8
        num_removed = self._metadata.get("num_removed", 0) if self._metadata else 0

        return dedent(f"""
            Your job is to complete an {n} x {n} chess board with {n} Queens in total, ssuch that no two attack each other.

            No two queens attack each other if they are not in the same row, column, or diagonal.
            Your job is to place {num_removed} more queen(s) on the board such that no two queens attack each other.

            Output format: A {n}x{n} board where each cell is either Q (queen) or _ (empty),
            with spaces separating cells in a row, and newlines separating rows.
            Example for a 4x4 board:
            _ Q _ _
            _ _ _ Q
            Q _ _ _
            _ _ Q _
        """).strip()

    def _make_dataset(self, *, seed: int | None):
        kwargs = self._dataset_kwargs.copy()
        if seed is not None and "seed" not in kwargs:
            kwargs["seed"] = seed
        if "size" not in kwargs:
            kwargs["size"] = 500

        return create_dataset("n_queens", **kwargs)

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

        logger.info("Reset ReasoningGym N Queens.")

        # obs.text = only the board state (caption), not the full question
        board_text = self._board_to_string(self._puzzle) if self._puzzle else ""

        obs = Observation(
            image=self.render(),
            text=board_text,
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

    def render(self) -> Image.Image | list[Image.Image] | None:
        return self._render_chess_board(
            self._puzzle, cell_px=self._cell_px, padding=self._padding
        )

    def _render_chess_board(
        self,
        puzzle: list[list[str]],
        cell_px: int = 64,
        padding: int = 24,
        light_sq: tuple[int, int, int] = (240, 217, 181),
        dark_sq: tuple[int, int, int] = (181, 136, 99),
        queen_color: tuple[int, int, int] = (50, 50, 50),
        empty_color: tuple[int, int, int] = (180, 180, 180),
    ) -> Image.Image:
        if not puzzle:
            return Image.new("RGB", (200, 200), (250, 250, 250))

        n = len(puzzle)
        size = padding * 2 + cell_px * n
        img = Image.new("RGB", (size, size), (250, 250, 250))
        draw = ImageDraw.Draw(img)

        font_path = self.assets_dir / "DejaVuSans.ttf"
        if font_path.exists():
            font = ImageFont.truetype(str(font_path), int(cell_px * 0.6))
        else:
            logger.warning(f"Font file not found: {font_path}, using default font")
            font = ImageFont.load_default()

        # Draw board and pieces
        for r in range(n):
            for c in range(n):
                x = padding + c * cell_px
                y = padding + r * cell_px

                # Alternating square colors (chess pattern)
                sq_color = light_sq if (r + c) % 2 == 0 else dark_sq
                draw.rectangle(
                    [x, y, x + cell_px - 1, y + cell_px - 1],
                    fill=sq_color,
                    outline=(100, 100, 100),
                    width=1,
                )

                cell = puzzle[r][c] if r < len(puzzle) and c < len(puzzle[r]) else "_"
                if cell == "Q":
                    # Draw queen symbol
                    self._draw_queen(draw, x, y, cell_px, queen_color, font)
                # Empty cells (_) are left blank - no markers

        # Draw border
        draw.rectangle(
            [padding - 2, padding - 2, size - padding + 1, size - padding + 1],
            outline=(60, 60, 60),
            width=2,
        )

        return img

    def _draw_queen(
        self,
        draw: ImageDraw.Draw,
        x: int,
        y: int,
        cell_px: int,
        color: tuple[int, int, int],
        font: ImageFont.FreeTypeFont,
    ):
        """Draw a queen symbol (♛) at the specified cell."""
        queen_symbol = "♛"
        bbox = draw.textbbox((0, 0), queen_symbol, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        cx = x + cell_px // 2
        cy = y + cell_px // 2
        draw.text((cx - tw // 2, cy - th // 2), queen_symbol, fill=color, font=font)

    def _board_to_string(self, board: list[list[str]]) -> str:
        """Convert board to string representation."""
        return "\n".join(" ".join(x for x in row) for row in board)

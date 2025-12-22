"""Knight Swap single-turn environment backed by reasoning-gym."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from reasoning_gym.factory import create_dataset

from gym_v import Env, Observation, get_logger

logger = get_logger()


class ReasoningGymKnightSwapEnv(Env):
    """Knight Swap puzzle using reasoning-gym's Knight Swap dataset.

    The player must swap white and black knights' positions through valid knight moves.

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
        self._board: dict[str, list[str]] | None = None
        self._pieces: dict[str, str | None] | None = None
        self._oracle_answer: str | None = None

    @property
    def description(self) -> str:
        """Return description with current puzzle parameters."""
        start_turn = self._metadata.get("start_turn", "w") if self._metadata else "w"
        return dedent(f"""
            Knight Swap Challenge:

            Legend:
            - 'w' = White Knight
            - 'B' = Black Knight
            - Empty squares are marked with '.'

            Objective:
            Swap the positions of all white knights with all black knights through valid moves.

            Rules:
            1. Knights move in L-shape (2 squares + 1 square perpendicular)
            2. Knights can only move to empty squares
            3. {start_turn} moves first, then players alternate
            4. All knights must reach their target positions (white ↔ black)

            Question:
            Is it possible to swap all knights' positions? If yes, list the moves.

            Answer Format:
            - For impossible puzzles: "No"
            - For possible puzzles: List moves as ["color,from,to", ...]
              Example: ["w,A1,B3"] means white knight moves A1→B3
        """).strip()

    def _make_dataset(self, *, seed: int | None):
        kwargs = self._dataset_kwargs.copy()
        if seed is not None and "seed" not in kwargs:
            kwargs["seed"] = seed
        if "size" not in kwargs:
            kwargs["size"] = 500

        return create_dataset("knight_swap", **kwargs)

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
        self._board = self._metadata.get("board", {})
        self._pieces = self._metadata.get("pieces", {})

        logger.info("Reset ReasoningGym Knight Swap.")

        # obs.text = only the board state (caption)
        board_text = self._format_board(self._board, self._pieces)

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

    def _format_board(
        self, board: dict[str, list[str]], pieces: dict[str, str | None]
    ) -> str:
        """Format the board state as a string."""
        if not board:
            return ""

        positions = list(board.keys())
        columns = sorted(set(pos[0] for pos in positions))
        rows = sorted(set(int(pos[1:]) for pos in positions), reverse=True)

        lines = []
        # Header
        lines.append("    " + "   ".join(columns))
        lines.append("   " + "----" * len(columns))

        # Board rows
        for row in rows:
            line = f"{row} |"
            for col in columns:
                pos = col + str(row)
                if pos in pieces:
                    piece = pieces[pos] if pieces[pos] is not None else "."
                    line += f" {piece} |"
                else:
                    line += "   |"
            lines.append(line)
            lines.append("   " + "----" * len(columns))

        return "\n".join(lines)

    def render(self) -> Image.Image:
        return self._render_knight_board(
            self._board,
            self._pieces,
            cell_px=self._cell_px,
            padding=self._padding,
        )

    def _render_knight_board(
        self,
        board: dict[str, list[str]],
        pieces: dict[str, str | None],
        cell_px: int = 64,
        padding: int = 24,
        light_sq: tuple[int, int, int] = (240, 217, 181),
        dark_sq: tuple[int, int, int] = (181, 136, 99),
        white_knight_color: tuple[int, int, int] = (255, 255, 255),
        black_knight_color: tuple[int, int, int] = (30, 30, 30),
    ) -> Image.Image:
        if not board:
            return Image.new("RGB", (200, 200), (250, 250, 250))

        positions = list(board.keys())
        columns = sorted(set(pos[0] for pos in positions))
        rows = sorted(set(int(pos[1:]) for pos in positions), reverse=True)

        n_cols = len(columns)
        n_rows = len(rows)

        width = padding * 2 + cell_px * n_cols
        height = padding * 2 + cell_px * n_rows
        img = Image.new("RGB", (width, height), (250, 250, 250))
        draw = ImageDraw.Draw(img)

        font_path = self.assets_dir / "DejaVuSans.ttf"
        if font_path.exists():
            font = ImageFont.truetype(str(font_path), int(cell_px * 0.5))
            small_font = ImageFont.truetype(str(font_path), int(cell_px * 0.25))
        else:
            logger.warning(f"Font file not found: {font_path}, using default font")
            font = ImageFont.load_default()
            small_font = font

        # Draw board and pieces
        for r_idx, row in enumerate(rows):
            for c_idx, col in enumerate(columns):
                pos = col + str(row)
                x = padding + c_idx * cell_px
                y = padding + r_idx * cell_px

                # Check if position is valid in the board
                if pos not in board:
                    # Draw as invalid/non-existent cell
                    draw.rectangle(
                        [x, y, x + cell_px - 1, y + cell_px - 1],
                        fill=(128, 128, 128),
                        outline=(100, 100, 100),
                        width=1,
                    )
                    continue

                # Alternating square colors (chess pattern)
                sq_color = light_sq if (r_idx + c_idx) % 2 == 0 else dark_sq
                draw.rectangle(
                    [x, y, x + cell_px - 1, y + cell_px - 1],
                    fill=sq_color,
                    outline=(100, 100, 100),
                    width=1,
                )

                # Draw coordinate label
                label = pos
                bbox = draw.textbbox((0, 0), label, font=small_font)
                _, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                draw.text(
                    (x + 2, y + cell_px - th - 2),
                    label,
                    fill=(80, 80, 80),
                    font=small_font,
                )

                # Draw piece if present
                piece = pieces.get(pos)
                if piece == "w":
                    self._draw_knight(
                        draw, x, y, cell_px, white_knight_color, font, outline=(0, 0, 0)
                    )
                elif piece == "B":
                    self._draw_knight(
                        draw,
                        x,
                        y,
                        cell_px,
                        black_knight_color,
                        font,
                        outline=(200, 200, 200),
                    )

        # Draw border
        draw.rectangle(
            [padding - 2, padding - 2, width - padding + 1, height - padding + 1],
            outline=(60, 60, 60),
            width=2,
        )

        return img

    def _draw_knight(
        self,
        draw: ImageDraw.Draw,
        x: int,
        y: int,
        cell_px: int,
        color: tuple[int, int, int],
        font: ImageFont.FreeTypeFont,
        outline: tuple[int, int, int] = (0, 0, 0),
    ):
        """Draw a knight symbol (♞) at the specified cell."""
        knight_symbol = "♞"
        bbox = draw.textbbox((0, 0), knight_symbol, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        cx = x + cell_px // 2
        cy = y + cell_px // 2
        # Draw outline
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx != 0 or dy != 0:
                    draw.text(
                        (cx - tw // 2 + dx, cy - th // 2 + dy),
                        knight_symbol,
                        fill=outline,
                        font=font,
                    )
        # Draw knight
        draw.text((cx - tw // 2, cy - th // 2), knight_symbol, fill=color, font=font)

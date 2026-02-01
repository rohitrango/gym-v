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
        self._board: dict[str, list[str]] | None = None
        self._pieces: dict[str, str | None] | None = None
        self._oracle_answer: str | None = None

    @property
    def description(self) -> str:
        """Return description with current puzzle parameters.

        Original reasoning-gym question format:
        ```
        Knight Swap Challenge:

        ```
            A   B   C   D
           ----------------
        3 |   | . |   | . |
           ----------------
        2 | B | w |   |   |
           ----------------
        1 |   |   | B | w |
           ----------------
        ```

        Legend:
        - 'w' = White Knight
        - 'B' = Black Knight
        - Empty squares are marked with '.'

        Objective:
        Swap the positions of all white knights with all black knights through valid moves.

        Rules:
        1. Knights move in L-shape (2 squares + 1 square perpendicular)
        2. Knights can only move to empty squares
        3. w moves first, then players alternate
        4. All knights must reach their target positions (white ↔ black)

        Question:
        Is it possible to swap all knights' positions? If yes, list the moves.

        Answer Format:
        - For impossible puzzles: "No"
        - For possible puzzles: List moves as ["color,from,to", ...]
          Example: ["w,A1,B3"] means white knight moves A1→B3
        ```

        Original reasoning-gym answer format:
        - For impossible puzzles: "No"
        - For possible puzzles: '["w,A1,C2", "B,D1,B2", ...]'
          (JSON array of move strings)
        """
        start_turn = self._metadata.get("start_turn", "w") if self._metadata else "w"
        start_color = "White" if start_turn == "w" else "Black"
        return dedent(f"""
            Knight Swap Challenge:

            In the image:
            - White knights are shown in white color
            - Black knights are shown in black color
            - Light squares are valid positions (knights can move here)
            - Dark gray squares are invalid positions (knights cannot move here)
            - Empty valid squares have no knight piece

            Objective:
            Swap the positions of all white knights with all black knights through valid moves.

            Rules:
            1. Knights move in L-shape (2 squares + 1 square perpendicular)
            2. Knights can only move to empty squares
            3. {start_color} ('{start_turn}') moves first, then players alternate
            4. All knights must reach their target positions (white ↔ black)

            Question:
            Is it possible to swap all knights' positions? If yes, list the moves.

            Answer Format:
            - For impossible puzzles: "No"
            - For possible puzzles: List moves as a JSON array ["color,from,to", ...]
              where color is 'w' for white or 'B' for black
              Example: ["w,A1,C2", "B,D1,B2"] means white knight A1→C2, then black knight D1→B2
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
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
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

    def render(self) -> Image.Image | list[Image.Image] | None:
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
    ) -> Image.Image:
        if not board:
            return Image.new("RGB", (200, 200), (250, 250, 250))

        positions = list(board.keys())
        columns = sorted(set(pos[0] for pos in positions))
        rows = sorted(set(int(pos[1:]) for pos in positions), reverse=True)

        n_cols = len(columns)
        n_rows = len(rows)

        # Calculate dimensions
        label_margin = 35
        board_width = cell_px * n_cols
        board_height = cell_px * n_rows
        width = padding * 2 + label_margin + board_width
        height = padding * 2 + label_margin + board_height

        bg_color = (245, 245, 250)
        img = Image.new("RGB", (width, height), bg_color)
        draw = ImageDraw.Draw(img)

        font_path = self.assets_dir / "DejaVuSans.ttf"
        if font_path.exists():
            knight_font = ImageFont.truetype(str(font_path), int(cell_px * 0.6))
            label_font = ImageFont.truetype(str(font_path), 16)
        else:
            logger.warning(f"Font file not found: {font_path}, using default font")
            knight_font = ImageFont.load_default()
            label_font = knight_font

        board_x0 = padding + label_margin
        board_y0 = padding

        # Draw labels
        for r_idx, row in enumerate(rows):
            bbox = draw.textbbox((0, 0), str(row), font=label_font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text(
                (
                    padding + (label_margin - tw) // 2,
                    board_y0 + r_idx * cell_px + (cell_px - th) // 2,
                ),
                str(row),
                fill=(100, 100, 100),
                font=label_font,
            )

        for c_idx, col in enumerate(columns):
            bbox = draw.textbbox((0, 0), col, font=label_font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text(
                (
                    board_x0 + c_idx * cell_px + (cell_px - tw) // 2,
                    board_y0 + board_height + 10,
                ),
                col,
                fill=(100, 100, 100),
                font=label_font,
            )

        # Draw board squares
        square_color = (220, 230, 240)
        invalid_color = (180, 180, 180)

        for r_idx, row in enumerate(rows):
            for c_idx, col in enumerate(columns):
                pos = col + str(row)
                x = board_x0 + c_idx * cell_px
                y = board_y0 + r_idx * cell_px

                if pos not in board:
                    # Invalid square
                    draw.rectangle(
                        [x, y, x + cell_px - 1, y + cell_px - 1],
                        fill=invalid_color,
                        outline=(150, 150, 150),
                        width=1,
                    )
                else:
                    # Valid square
                    draw.rectangle(
                        [x, y, x + cell_px - 1, y + cell_px - 1],
                        fill=square_color,
                        outline=(180, 180, 180),
                        width=1,
                    )

                    # Draw piece
                    piece = pieces.get(pos)
                    if piece == "w":
                        self._draw_knight(
                            draw, x, y, cell_px, (255, 255, 255), knight_font, (0, 0, 0)
                        )
                    elif piece == "B":
                        self._draw_knight(
                            draw,
                            x,
                            y,
                            cell_px,
                            (40, 40, 40),
                            knight_font,
                            (200, 200, 200),
                        )

        # Draw border
        draw.rectangle(
            [
                board_x0 - 2,
                board_y0 - 2,
                board_x0 + board_width + 1,
                board_y0 + board_height + 1,
            ],
            outline=(100, 100, 100),
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

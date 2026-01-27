"""Tsumego (Go problem) single-turn environment backed by reasoning-gym."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from reasoning_gym.factory import create_dataset

from gym_v import Env, Observation, get_logger

logger = get_logger()


class ReasoningGymTsumegoEnv(Env):
    """Tsumego (Go problem) using reasoning-gym's Tsumego dataset.

    The player must find the key move to capture opponent's stones.
    Black to play.

    Args:
        dataset_kwargs: Configuration parameters for the reasoning-gym dataset
        cell_px: Size of each cell in pixels for rendering
        padding: Padding around the board in pixels for rendering
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    def __init__(
        self,
        dataset_kwargs: dict[str, Any] | None = None,
        cell_px: int = 36,
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
        self._board: list[list[str]] | None = None
        self._oracle_answer: str | None = None

    @property
    def description(self) -> str:
        """Return description with current puzzle parameters.

        Original reasoning-gym question format:
        ```
        Black to play and capture some stones. Find the key move.

           A B C D E F G H I
         9 . . . . . . . . .
         8 . . . . . . . . .
         7 . . X X X . . . .
         6 . . X O O X . . .
         5 . . X O . O X . .
         4 . . . X O O X . .
         3 . . . . X X X . .
         2 . . . . . . . . .
         1 . . . . . . . . .

        X - Black
        O - White

        Specify your move in coordinates (e.g. 'C4' for column C, row 4)
        ```

        Original reasoning-gym answer format:
        - A coordinate string like "E5", "C4", etc.
        """
        return dedent("""
            Tsumego (Go Problem):

            Black to play and capture some stones. Find the key move.

            In the image:
            - Black stones are shown as black circles
            - White stones are shown as white circles
            - Empty intersections show just the board grid lines

            Specify your move in coordinates (e.g. 'C4' for column C, row 4).
            Output only the coordinate, nothing else, like "E5", "C4", etc.
        """).strip()

    def _make_dataset(self, *, seed: int | None):
        kwargs = self._dataset_kwargs.copy()
        if seed is not None and "seed" not in kwargs:
            kwargs["seed"] = seed
        if "size" not in kwargs:
            kwargs["size"] = 500

        return create_dataset("tsumego", **kwargs)

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
        self._board = self._metadata.get("board", [])

        logger.info("Reset ReasoningGym Tsumego.")

        # obs.text = only the board state (caption)
        board_text = self._board_to_string(self._board)

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

    def _board_to_string(self, board: list[list[str]]) -> str:
        """Convert board to string representation with coordinates."""
        if not board:
            return ""

        size = len(board)
        # Column labels
        cols = "   " + " ".join(chr(ord("A") + i) for i in range(size))
        # Board with row numbers
        rows = [f"{size-i:2d} {' '.join(row)}" for i, row in enumerate(board)]
        return cols + "\n" + "\n".join(rows)

    def render(self) -> Image.Image | list[Image.Image] | None:
        return self._render_go_board(
            self._board, cell_px=self._cell_px, padding=self._padding
        )

    def _render_go_board(
        self,
        board: list[list[str]],
        cell_px: int = 36,
        padding: int = 24,
        board_color: tuple[int, int, int] = (220, 179, 92),
        line_color: tuple[int, int, int] = (30, 30, 30),
        black_color: tuple[int, int, int] = (20, 20, 20),
        white_color: tuple[int, int, int] = (245, 245, 245),
    ) -> Image.Image:
        if not board:
            return Image.new("RGB", (200, 200), board_color)

        size = len(board)
        # Board dimensions - grid lines are at cell centers
        board_size = padding * 2 + cell_px * (size - 1)
        label_space = 30  # Space for labels

        width = board_size + label_space
        height = board_size + label_space
        img = Image.new("RGB", (width, height), board_color)
        draw = ImageDraw.Draw(img)

        font_path = self.assets_dir / "DejaVuSans.ttf"
        if font_path.exists():
            font = ImageFont.truetype(str(font_path), int(cell_px * 0.4))
        else:
            logger.warning(f"Font file not found: {font_path}, using default font")
            font = ImageFont.load_default()

        offset_x = label_space
        offset_y = 0

        # Draw grid lines
        for i in range(size):
            # Vertical lines
            x = offset_x + padding + i * cell_px
            draw.line(
                [
                    (x, offset_y + padding),
                    (x, offset_y + padding + (size - 1) * cell_px),
                ],
                fill=line_color,
                width=1,
            )
            # Horizontal lines
            y = offset_y + padding + i * cell_px
            draw.line(
                [
                    (offset_x + padding, y),
                    (offset_x + padding + (size - 1) * cell_px, y),
                ],
                fill=line_color,
                width=1,
            )

        # Draw star points (hoshi) for 9x9 and larger boards
        star_points = []
        if size == 9:
            star_points = [(2, 2), (2, 6), (4, 4), (6, 2), (6, 6)]
        elif size == 13:
            star_points = [(3, 3), (3, 9), (6, 6), (9, 3), (9, 9)]
        elif size == 19:
            star_points = [
                (3, 3),
                (3, 9),
                (3, 15),
                (9, 3),
                (9, 9),
                (9, 15),
                (15, 3),
                (15, 9),
                (15, 15),
            ]

        for r, c in star_points:
            if r < size and c < size:
                x = offset_x + padding + c * cell_px
                y = offset_y + padding + r * cell_px
                dot_r = cell_px // 8
                draw.ellipse(
                    [x - dot_r, y - dot_r, x + dot_r, y + dot_r],
                    fill=line_color,
                )

        # Draw column labels (A, B, C, ...)
        for i in range(size):
            x = offset_x + padding + i * cell_px
            label = chr(ord("A") + i)
            bbox = draw.textbbox((0, 0), label, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text(
                (x - tw // 2, height - label_space + (label_space - th) // 2),
                label,
                fill=line_color,
                font=font,
            )

        # Draw row labels (numbers, from top)
        for i in range(size):
            y = offset_y + padding + i * cell_px
            label = str(size - i)
            bbox = draw.textbbox((0, 0), label, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text(
                ((label_space - tw) // 2, y - th // 2),
                label,
                fill=line_color,
                font=font,
            )

        # Draw stones
        stone_radius = int(cell_px * 0.42)
        for r in range(size):
            for c in range(size):
                cell = board[r][c] if r < len(board) and c < len(board[r]) else "."
                if cell == ".":
                    continue

                x = offset_x + padding + c * cell_px
                y = offset_y + padding + r * cell_px

                if cell == "X":  # Black stone
                    # Draw shadow
                    draw.ellipse(
                        [
                            x - stone_radius + 2,
                            y - stone_radius + 2,
                            x + stone_radius + 2,
                            y + stone_radius + 2,
                        ],
                        fill=(100, 80, 50),
                    )
                    # Draw stone
                    draw.ellipse(
                        [
                            x - stone_radius,
                            y - stone_radius,
                            x + stone_radius,
                            y + stone_radius,
                        ],
                        fill=black_color,
                    )
                    # Draw highlight
                    highlight_r = stone_radius // 3
                    draw.ellipse(
                        [
                            x - stone_radius // 2 - highlight_r,
                            y - stone_radius // 2 - highlight_r,
                            x - stone_radius // 2 + highlight_r,
                            y - stone_radius // 2 + highlight_r,
                        ],
                        fill=(60, 60, 60),
                    )
                elif cell == "O":  # White stone
                    # Draw shadow
                    draw.ellipse(
                        [
                            x - stone_radius + 2,
                            y - stone_radius + 2,
                            x + stone_radius + 2,
                            y + stone_radius + 2,
                        ],
                        fill=(100, 80, 50),
                    )
                    # Draw stone
                    draw.ellipse(
                        [
                            x - stone_radius,
                            y - stone_radius,
                            x + stone_radius,
                            y + stone_radius,
                        ],
                        fill=white_color,
                        outline=(180, 180, 180),
                        width=1,
                    )
                    # Draw highlight
                    highlight_r = stone_radius // 3
                    draw.ellipse(
                        [
                            x - stone_radius // 2 - highlight_r,
                            y - stone_radius // 2 - highlight_r,
                            x - stone_radius // 2 + highlight_r,
                            y - stone_radius // 2 + highlight_r,
                        ],
                        fill=(255, 255, 255),
                    )

        return img

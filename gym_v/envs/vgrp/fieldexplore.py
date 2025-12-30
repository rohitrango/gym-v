"""Field Explore single-turn environment backed by VGRP-Bench."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation, get_logger

from .vgrp_factories import FieldExplorePuzzleFactory

logger = get_logger()


class VGRPFieldExploreEnv(Env):
    """Field Explore puzzle using VGRP-Bench's factory.

    Determine the location of mines based on numeric clues.
    Similar to Minesweeper but static: all clues are given upfront.

    Args:
        size: Grid size (default 8)
        cell_px: Size of each cell in pixels for rendering
        padding: Padding around the grid
        num_mines: Number of mines to place
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    def __init__(
        self,
        size: int = 8,
        num_mines: int = 10,
        cell_px: int = 50,
        padding: int = 30,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._size = size
        self._num_mines = num_mines
        self._cell_px = cell_px
        self._padding = padding

        self._seed: int | None = None
        self._factory = FieldExplorePuzzleFactory(size)

        # State
        self._solution_board: list[list[str]] | None = (
            None  # 's' for mine, 'e' for empty
        )
        self._puzzle_board: list[list[Any]] | None = None  # int for clue, 0 for unknown

    @property
    def description(self) -> str:
        return dedent(f"""
            Solve this {self._size}x{self._size} Field Explore puzzle.

            Rules:
            1. Identify all cells containing mines.
            2. Numbered cells indicate how many mines are adjacent (including diagonals).
            3. Numbered cells themselves do NOT contain mines.
            4. Find all mines based on these clues.

            Output format: A {self._size}x{self._size} grid with 's' for Mine (Selected) or 'e' for Empty.
        """).strip()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)
        self._seed = seed
        if seed is not None:
            np.random.seed(seed)

        # 1. Generate solution (mines)
        self._solution_board = self._generate_solution()

        # 2. Generate puzzle (clues)
        self._puzzle_board = self._generate_clues(self._solution_board)

        logger.info("Reset VGRP Field Explore.")

        obs = Observation(
            image=self.render(),
            text=self._board_to_text_puzzle(),
            metadata={"size": self._size},
        )
        info = {
            "oracle_answer": self._board_to_text(self._solution_board),
        }
        return obs, info

    def inner_step(
        self, action: str
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        try:
            answer_board = self._text_to_board(action)
            reward = 1.0 if self._check_solution(answer_board) else 0.0
        except Exception as e:
            logger.warning(f"Failed to parse answer: {e}")
            reward = 0.0

        obs = Observation(
            image=self.render(),
            text=self._board_to_text_puzzle(),
            metadata={"size": self._size},
        )
        info = {
            "oracle_answer": self._board_to_text(self._solution_board),
        }
        return obs, reward, True, False, info

    def _generate_solution(self) -> list[list[str]]:
        board = [["e" for _ in range(self._size)] for _ in range(self._size)]
        mines_placed = 0
        while mines_placed < self._num_mines:
            r, c = np.random.randint(0, self._size), np.random.randint(0, self._size)
            if board[r][c] == "e":
                board[r][c] = "s"
                mines_placed += 1
        return board

    def _generate_clues(self, solution: list[list[str]]) -> list[list[Any]]:
        puzzle = [[0 for _ in range(self._size)] for _ in range(self._size)]

        # In Minesweeper, non-mine cells usually show count.
        # Here we reveal counts for ALL empty cells? Or just some?
        # Standard logic: All empty cells revealed = trivial.
        # But this is "Field Explore", maybe we only reveal SOME numbers.
        # Let's assume standard static Minesweeper puzzle: some cells are revealed as clues.
        # Let's reveal 50% of empty cells as clues.

        for r in range(self._size):
            for c in range(self._size):
                if solution[r][c] == "s":
                    puzzle[r][c] = -1  # Hidden (could be mine)
                else:
                    # Calculate mine count
                    count = 0
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < self._size and 0 <= nc < self._size:
                                if solution[nr][nc] == "s":
                                    count += 1

                    # Randomly decide to show this clue or keep it hidden
                    if np.random.random() < 0.4:  # Show 40% clues
                        puzzle[r][c] = count
                    else:
                        puzzle[r][c] = -1  # Unknown

        return puzzle

    def _check_solution(self, answer_board: list[list[str]]) -> bool:
        if len(answer_board) != self._size:
            return False
        for i in range(self._size):
            if len(answer_board[i]) != self._size:
                return False
            for j in range(self._size):
                if answer_board[i][j] != self._solution_board[i][j]:
                    return False
        return True

    def _board_to_text(self, board: list[list[str]]) -> str:
        return "\n".join(" ".join(row) for row in board)

    def _board_to_text_puzzle(self) -> str:
        lines = []
        for row in self._puzzle_board:
            lines.append(" ".join(str(x) if x != -1 else "?" for x in row))
        return "\n".join(lines)

    def _text_to_board(self, text: str) -> list[list[str]]:
        lines = text.strip().split("\n")
        board = []
        for line in lines:
            line = line.strip()
            row = []
            for val in line.split():
                val = val.lower()
                if val in ["s", "e"]:
                    row.append(val)
                else:
                    row.append("e")  # default
            if row:
                board.append(row)
        return board

    def render(self) -> Image.Image:
        size_px = self._cell_px * self._size + 2 * self._padding
        img = Image.new("RGB", (size_px, size_px), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        font = ImageFont.load_default()
        try:
            font_path = self.assets_dir / "DejaVuSans.ttf"
            if font_path.exists():
                font = ImageFont.truetype(str(font_path), size=20)
        except Exception:
            pass

        # Draw Cells
        for r in range(self._size):
            for c in range(self._size):
                val = self._puzzle_board[r][c]
                x = self._padding + c * self._cell_px
                y = self._padding + r * self._cell_px

                # Draw cell background
                if val != -1:
                    # Revealed clue
                    draw.rectangle(
                        [x, y, x + self._cell_px, y + self._cell_px],
                        fill=(230, 230, 230),
                        outline=(150, 150, 150),
                    )
                    # Draw number (including 0)
                    colors = [
                        (0, 0, 0),
                        (0, 0, 255),
                        (0, 128, 0),
                        (255, 0, 0),
                        (0, 0, 128),
                        (128, 0, 0),
                        (0, 128, 128),
                        (0, 0, 0),
                        (128, 128, 128),
                    ]
                    color = colors[min(val, 8)]  # 0 is black

                    # Optional: Don't draw 0 if we want classic minesweeper look,
                    # but for VGRP alignment with text "0", we should probably draw it or make it very clear.
                    # Let's draw it for clarity.
                    txt = str(val)
                    if val == 0:
                        txt = "0"  # Explicitly draw 0

                    bbox = draw.textbbox((0, 0), txt, font=font)
                    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    draw.text(
                        (x + (self._cell_px - w) / 2, y + (self._cell_px - h) / 2),
                        txt,
                        fill=color,
                        font=font,
                    )
                else:
                    # Unknown / Hidden
                    draw.rectangle(
                        [x, y, x + self._cell_px, y + self._cell_px],
                        fill=(200, 200, 200),
                        outline=(100, 100, 100),
                    )
                    # draw bevel effect
                    draw.line(
                        [(x, y), (x + self._cell_px, y)], fill=(255, 255, 255), width=2
                    )
                    draw.line(
                        [(x, y), (x, y + self._cell_px)], fill=(255, 255, 255), width=2
                    )
                    draw.line(
                        [
                            (x, y + self._cell_px),
                            (x + self._cell_px, y + self._cell_px),
                        ],
                        fill=(100, 100, 100),
                        width=2,
                    )
                    draw.line(
                        [
                            (x + self._cell_px, y),
                            (x + self._cell_px, y + self._cell_px),
                        ],
                        fill=(100, 100, 100),
                        width=2,
                    )

        return img

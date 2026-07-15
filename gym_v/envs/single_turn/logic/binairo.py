"""Binairo single-turn environment backed by VGRP-Bench."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from gym_v import Env, Observation, get_logger
from gym_v.utils.vgrp_base import Constraint, PuzzleFactory
from gym_v.utils.vgrp_utils import generate_puzzle

logger = get_logger()


class ConstraintRowBalance(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_row_balance"

    def check(self, game_state: dict[str, Any]) -> bool:
        board = game_state["board"]
        size = len(board)
        expected_count = size // 2

        assert all(
            all(cell != "*" for cell in row) for row in board
        ), "'*' should be replaced by '0' in the initialization board"

        for row in board:
            if 0 not in row:  # Only check completed rows
                white_count = sum(1 for x in row if x == "w")
                black_count = sum(1 for x in row if x == "b")
                if white_count != black_count or white_count != expected_count:
                    return False
        return True


class ConstraintColBalance(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_col_balance"

    def check(self, game_state: dict[str, Any]) -> bool:
        board = game_state["board"]
        size = len(board)
        expected_count = size // 2

        for col in range(size):
            column = [board[row][col] for row in range(size)]
            if 0 not in column and "*" not in column:  # Only check completed columns
                white_count = sum(1 for x in column if x == "w")
                black_count = sum(1 for x in column if x == "b")
                if white_count != black_count or white_count != expected_count:
                    return False
        return True


class ConstraintNoTripleAdjacent(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_no_triple_adjacent"

    def check(self, game_state: dict[str, Any]) -> bool:
        board = game_state["board"]
        size = len(board)

        # Check rows
        for row in range(size):
            for col in range(size - 2):
                if (
                    board[row][col] != 0
                    and board[row][col] == board[row][col + 1] == board[row][col + 2]
                ):
                    return False

        # Check columns
        for col in range(size):
            for row in range(size - 2):
                if (
                    board[row][col] != 0
                    and board[row][col] == board[row + 1][col] == board[row + 2][col]
                ):
                    return False
        return True


class BinairoPuzzleFactory(PuzzleFactory):
    def __init__(self, size: int) -> None:
        super().__init__()
        if size < 4 or size % 2 != 0:
            raise ValueError("Size must be an even number greater than or equal to 4")

        self.game_name = "binairo"
        self.size = size
        self.constraints = [
            ConstraintRowBalance(),
            ConstraintColBalance(),
            ConstraintNoTripleAdjacent(),
        ]

        self.all_possible_values = ["w", "b"]

    def get_possible_values(
        self, game_state: dict[str, Any], row: int, col: int
    ) -> list[int]:
        possible_values = []
        board = game_state["board"]
        original_value = board[row][col]

        for value in self.all_possible_values:
            board[row][col] = value
            if self.check(game_state):
                possible_values.append(value)
        board[row][col] = original_value
        return possible_values


class BinairoEnv(Env):
    # Meta: source=VGRP, category=logic, turn=single
    """Binairo puzzle using VGRP-Bench's Binairo puzzle generator.

    Fill the grid with white (w) and black (b) circles following the rules:
    - Each row and column must have equal white and black circles
    - No more than two consecutive circles of the same color in any row/column

    Args:
        size: Grid size (must be even, default 6)
        num_hints: Number of pre-filled cells
        cell_px: Size of each cell in pixels for rendering
        padding: Padding around the grid in pixels for rendering
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""You are given a {size}×{size} Binairo puzzle.

Rules:
+ Each row and column must contain exactly {half_size} white circles ('w') and {half_size} black circles ('b')
+ No more than two consecutive circles of the same color are allowed in any row or column

Current puzzle state:
{puzzle_state}

Note: '?' represents empty cells that need to be filled.

**Output Format:**
Your answer should be a {size}×{size} grid where:
- Use 'w' for white circles
- Use 'b' for black circles
- Separate values with spaces within rows
- Separate rows with newlines

Example output for a 4×4 puzzle:
w b w b
b w b w
w b b w
b w w b
"""

    def __init__(
        self,
        size: int = 6,
        num_hints: int = 12,
        cell_px: int = 60,
        padding: int = 24,
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._size = size
        self._num_hints = num_hints
        self._cell_px = cell_px
        self._padding = padding
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._seed: int | None = None
        self._factory = BinairoPuzzleFactory(size)
        self._puzzle_board: list[list[str]] | None = None
        self._solution_board: list[list[str]] | None = None
        self._prompt: str | None = None

    @property
    def description(self) -> str:
        """Return description for Binairo puzzle."""
        return dedent(f"""
            Fill this {self._size}x{self._size} Binairo grid with white (w) and black (b) circles.

            In the image:
            - Empty cells need to be filled

            Rules:
            1. Each row and column must have exactly {self._size // 2} white and {self._size // 2} black circles
            2. No more than two consecutive circles of the same color in any row or column

            Output format: A {self._size}x{self._size} grid with 'w' for White circles or 'b' for Black circles separated by spaces within rows,
            and newlines separating rows. Example for 6x6:
            w b w b w b
            b w b w b w
            w b b w w b
            b w w b b w
            w b w w b b
            b w b b w w
        """).strip()

    def _prompt_generate(self) -> str:
        """Generate complete text prompt for the puzzle."""
        puzzle_text = self._board_to_text(self._puzzle_board)
        return self.prompt_template.format(
            size=self._size,
            half_size=self._size // 2,
            puzzle_state=puzzle_text,
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        super().reset(seed=seed)
        self._seed = seed

        # Generate a new puzzle
        result = generate_puzzle(self._factory, self._size, self._num_hints)
        if result is None:
            raise RuntimeError(
                f"Failed to generate Binairo puzzle with size {self._size} and {self._num_hints} hints"
            )

        self._puzzle_board, self._solution_board = result
        logger.info(f"Reset VGRP Binairo with {self._num_hints} hints.")

        # Generate prompt
        self._prompt = self._prompt_generate()

        # Convert board to text format
        self._board_to_text(self._puzzle_board)

        obs = Observation(
            image=self.render(),
            text=None,
            metadata={
                "text_prompt": None,
                "state_text": None,
                "size": self._size,
                "num_hints": self._num_hints,
            },
        )
        info = {
            "oracle_answer": self._board_to_text(self._solution_board),
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
        action_str = action[agent_id]
        # Parse answer and check correctness
        try:
            answer_board = self._text_to_board(action_str)
            reward = 1.0 if self._check_solution(answer_board) else 0.0
        except Exception as e:
            logger.warning(f"Failed to parse answer: {e}")
            reward = 0.0

        obs = Observation(
            image=self.render(),
            text=None,
            metadata={
                "text_prompt": None,
                "state_text": None,
                "size": self._size,
            },
        )
        info = {
            "oracle_answer": self._board_to_text(self._solution_board),
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

    def _check_solution(self, answer_board: list[list[str]]) -> bool:
        """Check if the answer matches the solution or satisfies constraints."""
        if len(answer_board) != self._size:
            return False
        for i in range(self._size):
            if len(answer_board[i]) != self._size:
                return False

        # 1. Exact match with generated solution
        matches_solution = True
        for i in range(self._size):
            for j in range(self._size):
                if answer_board[i][j] != self._solution_board[i][j]:
                    matches_solution = False
                    break
            if not matches_solution:
                break

        if matches_solution:
            return True

        # 2. Check validity using Factory (VGRP logic)
        # Ensure board is fully filled (no 0s)
        for row in answer_board:
            if 0 in row:
                return False

        game_state = {"board": answer_board}
        return self._factory.check(game_state)

    def _board_to_text(self, board: list[list[str]]) -> str:
        """Convert board to text format."""
        lines = []
        for row in board:
            lines.append(" ".join(str(v) if v != 0 else "?" for v in row))
        return "\n".join(lines)

    def _text_to_board(self, text: str) -> list[list[str]]:
        """Convert text to board."""
        lines = text.strip().split("\n")
        board = []
        for line in lines:
            row = []
            for val in line.strip().split():
                if val in ["w", "b"]:
                    row.append(val)
                elif val in ["○", "o", "O", "white", "0"]:
                    row.append("w")
                elif val in ["●", "x", "X", "black", "1"]:
                    row.append("b")
                else:
                    row.append(0)
            board.append(row)
        return board

    def render(self) -> Image.Image | list[Image.Image] | None:
        return self._render_binairo_grid(
            self._puzzle_board, cell_px=self._cell_px, padding=self._padding
        )

    def _render_binairo_grid(
        self,
        board: list[list[str]],
        cell_px: int = 60,
        padding: int = 24,
        bg: tuple[int, int, int] = (250, 250, 250),
        grid: tuple[int, int, int] = (30, 30, 30),
    ) -> Image.Image:
        n = self._size
        size = padding * 2 + cell_px * n
        img = Image.new("RGB", (size, size), bg)
        draw = ImageDraw.Draw(img)

        # Draw grid
        for i in range(n + 1):
            x = padding + i * cell_px
            y = padding + i * cell_px
            draw.line([(x, padding), (x, padding + n * cell_px)], fill=grid, width=2)
            draw.line([(padding, y), (padding + n * cell_px, y)], fill=grid, width=2)

        # Draw circles with 3D effect
        for i in range(n):
            for j in range(n):
                cell = board[i][j]
                x = padding + j * cell_px
                y = padding + i * cell_px

                if cell == "w":
                    # White circle with shadow and highlight
                    # Shadow
                    draw.ellipse(
                        [x + 13, y + 13, x + cell_px - 7, y + cell_px - 7],
                        fill=(200, 200, 200),
                    )
                    # Main circle
                    draw.ellipse(
                        [x + 10, y + 10, x + cell_px - 10, y + cell_px - 10],
                        fill=(255, 255, 255),
                        outline=(80, 80, 80),
                        width=3,
                    )
                    # Highlight for 3D effect
                    highlight_size = cell_px // 5
                    draw.ellipse(
                        [
                            x + cell_px // 4,
                            y + cell_px // 4,
                            x + cell_px // 4 + highlight_size,
                            y + cell_px // 4 + highlight_size,
                        ],
                        fill=(240, 240, 240),
                    )
                elif cell == "b":
                    # Black circle with gradient effect
                    # Shadow
                    draw.ellipse(
                        [x + 13, y + 13, x + cell_px - 7, y + cell_px - 7],
                        fill=(100, 100, 100),
                    )
                    # Main circle
                    draw.ellipse(
                        [x + 10, y + 10, x + cell_px - 10, y + cell_px - 10],
                        fill=(30, 30, 30),
                        outline=(20, 20, 20),
                        width=2,
                    )
                    # Highlight
                    highlight_size = cell_px // 6
                    draw.ellipse(
                        [
                            x + cell_px // 4,
                            y + cell_px // 4,
                            x + cell_px // 4 + highlight_size,
                            y + cell_px // 4 + highlight_size,
                        ],
                        fill=(100, 100, 100),
                    )
                # else: empty cell (0)

        return img

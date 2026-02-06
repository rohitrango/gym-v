"""Futoshiki single-turn environment backed by VGRP-Bench."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation, get_logger
from gym_v.utils.vgrp_base import (
    Constraint,
    ConstraintColNoRepeat,
    ConstraintRowNoRepeat,
    PuzzleFactory,
)
from gym_v.utils.vgrp_utils import generate_puzzle

logger = get_logger()


class ConstraintInequality(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_inequality"

    def check(self, game_state: dict[str, Any]) -> bool:
        board = game_state["board"]
        size = len(board)
        inequalities = game_state.get("inequalities", {"row": [], "col": []})

        row_ineq = inequalities.get(
            "row", [["" for _ in range(size - 1)] for _ in range(size)]
        )
        if not row_ineq:
            row_ineq = [["" for _ in range(size - 1)] for _ in range(size)]

        for row in range(size):
            for col in range(size - 1):
                if row_ineq[row][col] == "<":
                    if board[row][col] != 0 and board[row][col + 1] != 0:
                        if board[row][col] >= board[row][col + 1]:
                            return False
                elif row_ineq[row][col] == ">":
                    if board[row][col] != 0 and board[row][col + 1] != 0:
                        if board[row][col] <= board[row][col + 1]:
                            return False

        col_ineq = inequalities.get(
            "col", [["" for _ in range(size)] for _ in range(size - 1)]
        )
        if not col_ineq:
            col_ineq = [["" for _ in range(size)] for _ in range(size - 1)]

        for row in range(size - 1):
            for col in range(size):
                if col_ineq[row][col] == "^":
                    if board[row][col] != 0 and board[row + 1][col] != 0:
                        if board[row][col] >= board[row + 1][col]:
                            return False
                elif col_ineq[row][col] == "v":
                    if board[row][col] != 0 and board[row + 1][col] != 0:
                        if board[row][col] <= board[row + 1][col]:
                            return False
        return True


class FutoshikiPuzzleFactory(PuzzleFactory):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.game_name = "futoshiki"
        self.size = size
        self.constraints = [
            ConstraintRowNoRepeat(),
            ConstraintColNoRepeat(),
            ConstraintInequality(),
        ]
        self.all_possible_values = [i for i in range(1, size + 1)]

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


class FutoshikiEnv(Env):
    # Meta: source=VGRP, category=logic, turn=single
    """Futoshiki puzzle using VGRP-Bench's factory.

    Fill the grid with numbers 1-N.
    Inequality signs (<, >) between cells must be respected.
    Each row and column must contain each number exactly once.

    Args:
        size: Grid size (default 5)
        cell_px: Size of each cell in pixels for rendering
        padding: Padding around the grid
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""You are given a {size}×{size} Futoshiki puzzle.

Rules:
+ Fill the grid with numbers 1 to {size}
+ Each row and column must contain each number exactly once
+ Respect inequality signs between cells:
  - '<' means left cell < right cell
  - '>' means left cell > right cell
  - '^' means top cell < bottom cell
  - 'v' means top cell > bottom cell

Current puzzle state:
{puzzle_state}

Inequality constraints:
{inequalities}

Note: '.' represents empty cells that need to be filled.

**Output Format:**
Your answer should be a {size}×{size} grid where:
- Use numbers 1 to {size}
- Separate values with spaces within rows
- Separate rows with newlines

Example output for a 5×5 puzzle:
1 2 3 4 5
2 3 4 5 1
3 4 5 1 2
4 5 1 2 3
5 1 2 3 4
"""

    def __init__(
        self,
        size: int = 5,
        cell_px: int = 60,
        padding: int = 30,
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._size = size
        self._cell_px = cell_px
        self._padding = padding
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._seed: int | None = None
        self._solution_board: list[list[int]] | None = None
        self._inequalities: dict[str, list[list[str]]] | None = None
        self._puzzle_board: list[list[int]] | None = None
        self._factory = FutoshikiPuzzleFactory(size)
        self._prompt: str | None = None

    @property
    def description(self) -> str:
        return dedent(f"""
            Solve this {self._size}x{self._size} Futoshiki puzzle.

            Rules:
            1. Fill the grid with numbers 1 to {self._size}.
            2. Each row and column must contain each number exactly once.
            3. Respect the inequality signs between cells (if any).
               - '<' or '>' between horizontal neighbors.
               - '^' or 'v' between vertical neighbors.

            Output format: A {self._size}x{self._size} grid with numbers separated by spaces.
            separated by spaces within rows, and newlines separating rows.
            Example for 5x5:
            1 2 3 4 5
            2 3 4 5 1
            3 4 5 1 2
            4 5 1 2 3
            5 1 2 3 4
        """).strip()

    def _prompt_generate(self) -> str:
        """Generate complete text prompt for the puzzle."""
        puzzle_text = self._board_to_text_puzzle()
        inequalities_text = self._inequalities_to_text()
        return self.prompt_template.format(
            size=self._size,
            puzzle_state=puzzle_text,
            inequalities=inequalities_text,
        )

    def _inequalities_to_text(self) -> str:
        """Convert inequalities to text description."""
        lines = []
        # Row inequalities (horizontal)
        for r in range(self._size):
            for c in range(self._size - 1):
                sign = self._inequalities["row"][r][c]
                if sign:
                    lines.append(
                        f"Row {r + 1}: cell({r + 1},{c + 1}) {sign} cell({r + 1},{c + 2})"
                    )
        # Col inequalities (vertical)
        for r in range(self._size - 1):
            for c in range(self._size):
                sign = self._inequalities["col"][r][c]
                if sign:
                    lines.append(
                        f"Col {c + 1}: cell({r + 1},{c + 1}) {sign} cell({r + 2},{c + 1})"
                    )
        return "\n".join(lines) if lines else "No inequality constraints"

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        super().reset(seed=seed)
        self._seed = seed
        if seed is not None:
            np.random.seed(seed)

        # 1. Generate Latin Square using generate_puzzle (official solver)
        # Empty board, no inequalities = Latin Square
        result = generate_puzzle(
            self._factory, self._size, num_hints=0, max_attempts=1000
        )
        if result is None:
            raise RuntimeError("Failed to generate Futoshiki Latin Square")

        _, self._solution_board = result

        # 2. Generate Inequalities (Structure)
        self._inequalities = self._generate_inequalities(self._solution_board)

        # 3. Create Puzzle (hide numbers, keep some hints?)
        # Standard Futoshiki has some initial numbers. Let's keep ~20%.
        self._puzzle_board = [[0 for _ in range(self._size)] for _ in range(self._size)]
        num_hints = max(1, int(self._size * self._size * 0.2))
        cells = [(r, c) for r in range(self._size) for c in range(self._size)]
        np.random.shuffle(cells)
        for i in range(num_hints):
            r, c = cells[i]
            self._puzzle_board[r][c] = self._solution_board[r][c]

        logger.info("Reset VGRP Futoshiki.")

        # Generate prompt
        self._prompt = self._prompt_generate()

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

    def _generate_inequalities(self, board: list[list[int]]) -> dict:
        row_ineq = [["" for _ in range(self._size - 1)] for _ in range(self._size)]
        col_ineq = [["" for _ in range(self._size)] for _ in range(self._size - 1)]

        # Add random inequalities (density ~30-50%)
        density = 0.4

        for r in range(self._size):
            for c in range(self._size - 1):
                if np.random.random() < density:
                    if board[r][c] < board[r][c + 1]:
                        row_ineq[r][c] = "<"
                    else:
                        row_ineq[r][c] = ">"

        for r in range(self._size - 1):
            for c in range(self._size):
                if np.random.random() < density:
                    if board[r][c] < board[r + 1][c]:
                        col_ineq[r][c] = (
                            "^"  # Top is smaller (standard notation?) or just use visual logic
                        )
                        # Factory uses '^' and 'v'
                    else:
                        col_ineq[r][c] = "v"

        return {"row": row_ineq, "col": col_ineq}

    def _check_solution(self, answer_board: list[list[int]]) -> bool:
        """Check if the answer matches the solution or satisfies constraints."""
        if len(answer_board) != self._size:
            return False
        for i in range(self._size):
            if len(answer_board[i]) != self._size:
                return False

        # 1. Exact match
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

        # 2. VGRP check
        # Must be complete (no 0s)
        for row in answer_board:
            if 0 in row:
                return False

        # Check initial hints
        for i in range(self._size):
            for j in range(self._size):
                if self._puzzle_board[i][j] != 0:
                    if answer_board[i][j] != self._puzzle_board[i][j]:
                        return False

        game_state = {"board": answer_board, "inequalities": self._inequalities}
        return self._factory.check(game_state)

    def _board_to_text(self, board: list[list[int]]) -> str:
        return "\n".join(" ".join(str(x) for x in row) for row in board)

    def _board_to_text_puzzle(self) -> str:
        lines = []
        for row in self._puzzle_board:
            lines.append(" ".join(str(x) if x > 0 else "." for x in row))
        return "\n".join(lines)

    def _text_to_board(self, text: str) -> list[list[int]]:
        lines = text.strip().split("\n")
        board = []
        for line in lines:
            line = line.strip()
            row = []
            for val in line.split():
                try:
                    row.append(int(val))
                except Exception:
                    row.append(0)
            if row:
                board.append(row)
        return board

    def render(self) -> Image.Image | list[Image.Image] | None:
        size_px = self._cell_px * self._size + 2 * self._padding
        img = Image.new("RGB", (size_px, size_px), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        font = ImageFont.load_default()
        try:
            font_path = self.assets_dir / "DejaVuSans.ttf"
            if font_path.exists():
                font = ImageFont.truetype(str(font_path), size=24)
        except Exception:
            pass

        # Cells
        for r in range(self._size):
            for c in range(self._size):
                x = self._padding + c * self._cell_px
                y = self._padding + r * self._cell_px

                draw.rectangle(
                    [x, y, x + self._cell_px, y + self._cell_px],
                    outline=(0, 0, 0),
                    width=2,
                )

                val = self._puzzle_board[r][c]
                if val > 0:
                    bbox = draw.textbbox((0, 0), str(val), font=font)
                    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    draw.text(
                        (x + (self._cell_px - w) / 2, y + (self._cell_px - h) / 2),
                        str(val),
                        fill=(0, 0, 0),
                        font=font,
                    )

        # Inequalities
        sign_font = font  # Reuse font

        for r in range(self._size):
            for c in range(self._size - 1):
                sign = self._inequalities["row"][r][c]
                if sign:
                    x = self._padding + (c + 1) * self._cell_px
                    y = self._padding + r * self._cell_px + self._cell_px // 2
                    # Draw sign centered on the grid line
                    bbox = draw.textbbox((0, 0), sign, font=sign_font)
                    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    # Offset slightly to center on line
                    draw.rectangle(
                        [x - 10, y - 10, x + 10, y + 10], fill=(255, 255, 255)
                    )  # Clear background
                    draw.text(
                        (x - w / 2, y - h / 2), sign, fill=(0, 0, 0), font=sign_font
                    )

        for r in range(self._size - 1):
            for c in range(self._size):
                sign = self._inequalities["col"][r][c]
                if sign:
                    x = self._padding + c * self._cell_px + self._cell_px // 2
                    y = self._padding + (r + 1) * self._cell_px
                    bbox = draw.textbbox((0, 0), sign, font=sign_font)
                    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    draw.rectangle(
                        [x - 10, y - 10, x + 10, y + 10], fill=(255, 255, 255)
                    )
                    draw.text(
                        (x - w / 2, y - h / 2), sign, fill=(0, 0, 0), font=sign_font
                    )

        return img

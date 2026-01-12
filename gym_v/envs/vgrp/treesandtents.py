"""Trees and Tents single-turn environment backed by VGRP-Bench."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation, get_logger

from .utils import generate_puzzle
from .vgrp_base import Constraint, PuzzleFactory

logger = get_logger()


class ConstraintRowTents(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_row_tents"

    def check(self, game_state: dict[str, Any]) -> bool:
        board = game_state["board"]
        clues = game_state.get("clues", None)
        if not clues:
            return True
        for i, row in enumerate(board):
            if 0 not in row:
                if row.count("tt") != clues["row_clues"][i]:
                    return False
            else:
                if row.count("tt") > clues["row_clues"][i]:
                    return False
        return True


class ConstraintColTents(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_col_tents"

    def check(self, game_state: dict[str, Any]) -> bool:
        board = game_state["board"]
        clues = game_state.get("clues", None)
        if not clues:
            return True
        size = len(board)
        for j in range(size):
            col = [board[i][j] for i in range(size)]
            if 0 not in col:
                if col.count("tt") != clues["col_clues"][j]:
                    return False
            else:
                if col.count("tt") > clues["col_clues"][j]:
                    return False
        return True


class ConstraintTentTree(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_tent_tree"

    def check(self, game_state: dict[str, Any]) -> bool:
        board = game_state["board"]
        size = len(board)
        for i in range(size):
            for j in range(size):
                if board[i][j] == "tt":
                    adjacent_trees = []
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < size and 0 <= nj < size:
                            if board[ni][nj] == "tr":
                                adjacent_trees.append((ni, nj))
                    if len(adjacent_trees) != 1:
                        return False
        for i in range(size):
            for j in range(size):
                if board[i][j] == "tr":
                    adjacent_tents = 0
                    adjacent_non_allocated = 0
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < size and 0 <= nj < size:
                            if board[ni][nj] == "tt":
                                adjacent_tents += 1
                            elif board[ni][nj] == 0:
                                adjacent_non_allocated += 1
                    if adjacent_tents > 1:
                        return False
                    if adjacent_tents == 0 and adjacent_non_allocated == 0:
                        return False
        return True


class ConstraintAdjacentTents(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_adjacent_tents"

    def check(self, game_state: dict[str, Any]) -> bool:
        board = game_state["board"]
        size = len(board)
        for i in range(size):
            for j in range(size):
                if board[i][j] == "tt":
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if 0 <= ni < size and 0 <= nj < size:
                                if board[ni][nj] == "tt":
                                    return False
        return True


class ConstraintTentTreeCount(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_tent_tree_count"

    def check(self, game_state: dict[str, Any]) -> bool:
        board = game_state["board"]
        num_trees = sum(row.count("tr") for row in board)
        num_tents = sum(row.count("tt") for row in board)
        num_unallocated = sum(row.count(0) for row in board)
        if num_unallocated == 0:
            return num_tents == num_trees
        return (num_tents + num_unallocated) >= num_trees


class TreesAndTentsPuzzleFactory(PuzzleFactory):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.game_name = "treesandtents"
        self.size = size
        self.constraints = [
            ConstraintRowTents(),
            ConstraintColTents(),
            ConstraintTentTree(),
            ConstraintAdjacentTents(),
            ConstraintTentTreeCount(),
        ]
        self.all_possible_values = ["tt", "e"]

    def get_possible_values(
        self, game_state: dict[str, Any], row: int, col: int
    ) -> list[str]:
        board = game_state["board"]
        if board[row][col] != 0:
            return []
        possible = []
        original_value = board[row][col]
        for value in self.all_possible_values:
            board[row][col] = value
            if self.check(game_state):
                possible.append(value)
        board[row][col] = original_value
        return possible


class VGRPTreesAndTentsEnv(Env):
    """Trees and Tents puzzle using VGRP-Bench's TreesAndTents puzzle generator.

    Place tents next to trees according to clues.

    Args:
        size: Grid size (default 5)
        num_hints: Not used (clues derived from solution)
        cell_px: Size of each cell in pixels for rendering
        padding: Padding around the grid in pixels for rendering
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    def __init__(
        self,
        size: int = 5,
        num_hints: int = 0,
        cell_px: int = 60,
        padding: int = 50,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._size = size
        self._cell_px = cell_px
        self._padding = padding

        self._seed: int | None = None
        self._puzzle_board: list[list[str]] | None = None
        self._solution_board: list[list[str]] | None = None
        self._tree_positions: list[tuple[int, int]] | None = None
        self._row_clues: list[int] | None = None
        self._col_clues: list[int] | None = None
        self._factory = TreesAndTentsPuzzleFactory(size)

    @property
    def description(self) -> str:
        """Return description for Trees and Tents puzzle."""
        return dedent(f"""
            Solve this {self._size}x{self._size} Trees and Tents puzzle.

            In the image:
            - Numbers on left show required tents per row
            - Numbers on top show required tents per column

            Rules:
            1. Place exactly one tent horizontally or vertically adjacent to each tree
            2. Tents cannot touch each other, even diagonally
            3. Row and column tent counts must match the given clues
            4. Each tree must have exactly one tent next to it (not diagonal)

            Output format: A {self._size}x{self._size} grid where:
            - 'tr' = tree (given, don't output)
            - 'tt' = tent (your answer)
            - 'e' = empty
            separated by spaces within rows, and newlines separating rows.
        """).strip()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed, options=options)
        self._seed = seed
        if seed is not None:
            np.random.seed(seed)

        # Generate tree positions
        self._tree_positions = self._generate_trees()

        # Prepare initial board with trees
        init_board = [[0 for _ in range(self._size)] for _ in range(self._size)]
        for r, c in self._tree_positions:
            init_board[r][c] = "tr"

        original_constraints = self._factory.constraints
        self._factory.constraints = [
            c
            for c in original_constraints
            if c.name
            in [
                "constraint_tent_tree",
                "constraint_adjacent_tents",
                "constraint_tent_tree_count",
            ]
        ]

        result = generate_puzzle(
            self._factory, self._size, num_hints=0, initial_board=init_board
        )

        # Restore constraints
        self._factory.constraints = original_constraints

        if result is None:
            raise RuntimeError("Failed to generate Trees and Tents solution")

        _, self._solution_board = result

        # Calculate row/col clues from solution
        self._row_clues = [
            sum(1 for cell in row if cell == "tt") for row in self._solution_board
        ]
        self._col_clues = [
            sum(1 for i in range(self._size) if self._solution_board[i][j] == "tt")
            for j in range(self._size)
        ]

        # Puzzle board shows trees
        self._puzzle_board = [
            ["e" for _ in range(self._size)] for _ in range(self._size)
        ]
        for r, c in self._tree_positions:
            self._puzzle_board[r][c] = "tr"

        logger.info("Reset VGRP Trees and Tents.")

        obs = Observation(
            image=self.render(),
            text=self._board_to_text_with_clues(),
            metadata={"size": self._size},
        )
        info = {
            "oracle_answer": self._board_to_text(self._solution_board),
        }
        return obs, info

    def _generate_trees(self) -> list[tuple[int, int]]:
        """Generate random tree positions."""
        trees = []
        num_trees = np.random.randint(max(2, self._size - 2), self._size + 1)

        available_cells = [(i, j) for i in range(self._size) for j in range(self._size)]
        np.random.shuffle(available_cells)

        for cell in available_cells[:num_trees]:
            trees.append(cell)

        return trees

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
            text=self._board_to_text_with_clues(),
            metadata={"size": self._size},
        )
        terminated = True
        truncated = False
        info = {
            "oracle_answer": self._board_to_text(self._solution_board),
        }
        return obs, reward, terminated, truncated, info

    def _check_solution(self, answer_board: list[list[str]]) -> bool:
        """Check if the answer matches the solution."""
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
        merged_board = [row[:] for row in answer_board]
        for r in range(self._size):
            for c in range(self._size):
                if self._puzzle_board[r][c] == "tr":
                    merged_board[r][c] = "tr"

        game_state = {
            "board": merged_board,
            "clues": {"row_clues": self._row_clues, "col_clues": self._col_clues},
        }
        return self._factory.check(game_state)

    def _board_to_text_with_clues(self) -> str:
        """Convert board to text with clues."""
        lines = []
        lines.append(
            f"Row clues (tents per row): {' '.join(map(str, self._row_clues))}"
        )
        lines.append(
            f"Col clues (tents per col): {' '.join(map(str, self._col_clues))}"
        )
        lines.append("")
        lines.append("Trees are marked in the grid below:")
        for row in self._puzzle_board:
            lines.append(" ".join(row))
        return "\n".join(lines)

    def _board_to_text(self, board: list[list[str]]) -> str:
        """Convert board to text."""
        lines = []
        for row in board:
            lines.append(" ".join(row))
        return "\n".join(lines)

    def _text_to_board(self, text: str) -> list[list[str]]:
        """Parse text to board."""
        lines = text.strip().split("\n")
        board = []
        for line in lines:
            line = line.strip()
            if not line or ":" in line or "Trees" in line:
                continue
            row = []
            for val in line.split():
                val = val.strip().lower()
                if val in ["tt", "tr", "e"]:
                    row.append(val)
                else:
                    row.append("e")
            if row:
                board.append(row)
        return board

    def render(self) -> Image.Image | list[Image.Image] | None:
        return self._render_trees_and_tents(
            self._puzzle_board,
            self._row_clues,
            self._col_clues,
            cell_px=self._cell_px,
            padding=self._padding,
        )

    def _render_trees_and_tents(
        self,
        puzzle: list[list[str]],
        row_clues: list[int],
        col_clues: list[int],
        cell_px: int = 60,
        padding: int = 50,
        bg: tuple[int, int, int] = (220, 245, 220),  # Light green grass
        fg: tuple[int, int, int] = (20, 20, 20),
        grid: tuple[int, int, int] = (150, 180, 150),
        tree_color: tuple[int, int, int] = (34, 139, 34),
    ) -> Image.Image:
        n = self._size
        size = padding * 2 + cell_px * n
        img = Image.new("RGB", (size, size), bg)
        draw = ImageDraw.Draw(img)

        font_path = self.assets_dir / "DejaVuSans.ttf"
        if font_path.exists():
            font = ImageFont.truetype(str(font_path), int(cell_px * 0.35))
        else:
            font = ImageFont.load_default()

        # Draw grid
        for i in range(n + 1):
            x = padding + i * cell_px
            y = padding + i * cell_px
            draw.line([(padding, y), (padding + n * cell_px, y)], fill=grid, width=2)
            draw.line([(x, padding), (x, padding + n * cell_px)], fill=grid, width=2)

        # Draw trees (simple: trunk + round foliage)
        for r in range(n):
            for c in range(n):
                if puzzle[r][c] == "tr":
                    x = padding + c * cell_px
                    y = padding + r * cell_px
                    cx = x + cell_px // 2
                    cy = y + cell_px // 2

                    # Tree trunk (brown rectangle)
                    trunk_w = cell_px // 5
                    draw.rectangle(
                        [
                            cx - trunk_w // 2,
                            cy + cell_px // 6,
                            cx + trunk_w // 2,
                            cy + cell_px // 2.5,
                        ],
                        fill=(101, 67, 33),
                        outline=(70, 45, 20),
                        width=2,
                    )

                    # Tree foliage (big green circle)
                    foliage_r = cell_px // 3
                    draw.ellipse(
                        [
                            cx - foliage_r,
                            cy - foliage_r - cell_px // 6,
                            cx + foliage_r,
                            cy + foliage_r - cell_px // 6,
                        ],
                        fill=(50, 150, 50),
                        outline=(30, 100, 30),
                        width=3,
                    )

                elif puzzle[r][c] == "tt":
                    # Draw tent (triangle ⛺)
                    x = padding + c * cell_px
                    y = padding + r * cell_px
                    cx = x + cell_px // 2
                    margin = cell_px // 5

                    # Tent triangle (orange/red)
                    points = [
                        (cx, y + margin),  # top
                        (x + margin, y + cell_px - margin),  # bottom-left
                        (x + cell_px - margin, y + cell_px - margin),  # bottom-right
                    ]
                    draw.polygon(
                        points, fill=(255, 140, 0), outline=(200, 100, 0), width=3
                    )

                    # Tent entrance (dark rectangle)
                    entrance_w = cell_px // 6
                    entrance_h = cell_px // 4
                    draw.rectangle(
                        [
                            cx - entrance_w // 2,
                            y + cell_px - margin - entrance_h,
                            cx + entrance_w // 2,
                            y + cell_px - margin,
                        ],
                        fill=(80, 50, 30),
                    )

        # Draw row clues (left) with background circle
        for i in range(n):
            text = str(row_clues[i])
            y = padding + i * cell_px + cell_px // 2
            # Circle background
            circle_r = cell_px // 5
            draw.ellipse(
                [
                    padding // 3 - circle_r,
                    y - circle_r,
                    padding // 3 + circle_r,
                    y + circle_r,
                ],
                fill=(255, 200, 100),
                outline=(200, 150, 50),
                width=2,
            )
            draw.text(
                (padding // 3, y), text, fill=(40, 30, 20), font=font, anchor="mm"
            )

        # Draw col clues (top) with background circle
        for j in range(n):
            text = str(col_clues[j])
            x = padding + j * cell_px + cell_px // 2
            # Circle background
            circle_r = cell_px // 5
            draw.ellipse(
                [
                    x - circle_r,
                    padding // 3 - circle_r,
                    x + circle_r,
                    padding // 3 + circle_r,
                ],
                fill=(255, 200, 100),
                outline=(200, 150, 50),
                width=2,
            )
            draw.text(
                (x, padding // 3), text, fill=(40, 30, 20), font=font, anchor="mm"
            )

        return img

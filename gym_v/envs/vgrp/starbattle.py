"""Star Battle single-turn environment backed by VGRP-Bench."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from gym_v import Env, Observation, get_logger

from .utils import generate_puzzle
from .vgrp_base import Constraint, PuzzleFactory

logger = get_logger()


class ConstraintRowStar(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_row_star"

    def check(self, game_state: dict[str, Any]) -> bool:
        board = game_state["board"]
        for row in board:
            if 0 not in row:
                if sum(1 for cell in row if cell == "s") != 1:
                    return False
            else:
                if sum(1 for cell in row if cell == "s") > 1:
                    return False
        return True


class ConstraintColStar(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_col_star"

    def check(self, game_state: dict[str, Any]) -> bool:
        board = game_state["board"]
        size = len(board)
        for col in range(size):
            col_values = [board[row][col] for row in range(size)]
            if 0 not in col_values:
                if sum(1 for val in col_values if val == "s") != 1:
                    return False
            else:
                if sum(1 for val in col_values if val == "s") > 1:
                    return False
        return True


class ConstraintRegionStar(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_region_star"

    def check(self, game_state: dict[str, Any]) -> bool:
        board = game_state["board"]
        regions = game_state["regions"]
        size = len(board)
        region_counts = {}
        for i in range(size):
            for j in range(size):
                if board[i][j] == "s":
                    region = regions[i][j]
                    region_counts[region] = region_counts.get(region, 0) + 1
                    if region_counts[region] > 1:
                        return False
        for region_num in set(cell for row in regions for cell in row):
            region_cells = [
                (i, j)
                for i in range(size)
                for j in range(size)
                if regions[i][j] == region_num
            ]
            if all(board[i][j] != 0 for i, j in region_cells):
                if region_counts.get(region_num, 0) != 1:
                    return False
        return True


class ConstraintAdjacentStar(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_adjacent_star"

    def check(self, game_state: dict[str, Any]) -> bool:
        board = game_state["board"]
        size = len(board)
        for row in range(size):
            for col in range(size):
                if board[row][col] == "s":
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            new_row, new_col = row + dr, col + dc
                            if (
                                0 <= new_row < size
                                and 0 <= new_col < size
                                and board[new_row][new_col] == "s"
                            ):
                                return False
        return True


class StarBattlePuzzleFactory(PuzzleFactory):
    def __init__(self, size: int, num_stars: int = 1) -> None:
        super().__init__()
        self.game_name = "starbattle"
        self.size = size
        self.constraints = [
            ConstraintRowStar(),
            ConstraintColStar(),
            ConstraintAdjacentStar(),
            ConstraintRegionStar(),
        ]
        self.all_possible_values = ["s", "e"]

    def get_possible_values(
        self, game_state: dict[str, Any], row: int, col: int
    ) -> list[str]:
        board = game_state["board"]
        if board[row][col] in ["s", "e"]:
            return []
        possible = []
        for val in ["s", "e"]:
            board[row][col] = val
            if self.check(game_state):
                possible.append(val)
            board[row][col] = 0
        return possible


class VGRPStarBattleEnv(Env):
    """Star Battle puzzle using VGRP-Bench's factory.

    Place stars in the grid such that:
    - Each row, column, and outlined region contains exactly N stars (usually 1 or 2).
    - Stars cannot touch each other, not even diagonally.

    Args:
        size: Grid size (default 8)
        stars_per_group: Number of stars per row/col/region (default 1)
        cell_px: Size of each cell in pixels for rendering
        padding: Padding around the grid
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""You are given a {size}×{size} Star Battle puzzle.

Rules:
+ Place exactly {stars_per_group} star(s) in each row
+ Place exactly {stars_per_group} star(s) in each column
+ Place exactly {stars_per_group} star(s) in each outlined region
+ Stars cannot touch each other, not even diagonally (no two stars can be adjacent in any direction)

Region layout:
{regions}

Note: Each number represents a different region. Cells with the same number belong to the same region.

**Output Format:**
Your answer should be a {size}×{size} grid where:
- Use 's' for star cells
- Use 'e' for empty cells
- Separate values with spaces within rows
- Separate rows with newlines

Example output for an 8×8 puzzle with 1 star per group:
e e e s e e e e
e s e e e e e e
e e e e e e s e
s e e e e e e e
e e e e s e e e
e e s e e e e e
e e e e e e e s
e e e e e s e e
"""

    def __init__(
        self,
        size: int = 8,
        stars_per_group: int = 1,
        cell_px: int = 50,
        padding: int = 20,
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._size = size
        self._stars_per_group = stars_per_group
        self._cell_px = cell_px
        self._padding = padding
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._seed: int | None = None
        self._solution_board: list[list[str]] | None = None  # 's' (star), 'e' (empty)
        self._regions: list[list[int]] | None = None  # Grid of region IDs
        self._puzzle_board: list[list[str]] | None = None  # Empty initially
        self._factory = StarBattlePuzzleFactory(size, stars_per_group)
        self._prompt: str | None = None

    @property
    def description(self) -> str:
        return dedent(f"""
            Solve this {self._size}x{self._size} Star Battle puzzle.

            Rules:
            1. Place stars ('s') in the grid.
            2. Each row, column, and outlined region must contain exactly {self._stars_per_group} star(s).
            3. Stars cannot touch each other, even diagonally.

            Output format: A {self._size}x{self._size} grid with 's' for Star and 'e' for Empty.
            separated by spaces within rows, and newlines separating rows.
            Example for 6x6:
            e e s s e e
            s s s e e e
            e e e e e e
            s e s s s s
            e e e e e e
            e e s s s e
        """).strip()

    def _prompt_generate(self) -> str:
        """Generate complete text prompt for the puzzle."""
        regions_text = self._board_to_text_regions()
        return self.prompt_template.format(
            size=self._size,
            stars_per_group=self._stars_per_group,
            regions=regions_text,
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        super().reset(seed=seed)
        self._seed = seed
        if seed is not None:
            np.random.seed(seed)

        # 1. Generate Regions first (Structure)
        # We need regions to validate stars in StarBattle logic
        # But usually one places stars then grows regions.
        # If we use generate_puzzle (solver), we MUST have regions defined.
        # So we must generate regions first.
        # But generating valid regions that SUPPORT a valid star placement is hard without stars.
        # Chicken and egg.
        # However, we can generate random regions, then try to solve.
        # If solve fails (impossible regions), retry region generation.

        for _retry in range(10):
            # Use a dummy stars placement just to seed regions, then discard stars?
            # Or use my region generator which is robust.
            # My _generate_regions needed a solution to guarantee validity.
            # "We need K regions... Each region needs N stars... group stars... grow regions"
            # This construction method GUARANTEES a solution exists.
            # If I generate purely random regions (Voronoi?), they might be unsolvable.
            # So I WILL keep the construction logic: Generate Stars -> Generate Regions -> (Optional: Re-solve to verify?)
            # The user said "use official generation mechanism".
            # Official mechanism is: Solver.
            # Solver requires Regions.
            # So I must provide regions.
            # Providing regions that have at least one solution is best done by construction.
            # So I will keep _generate_stars and _generate_regions to CREATE the problem instance.
            # Then, I can optionally run generate_puzzle to "prove" it's solvable or just use the stars I generated.
            # Actually, if I already generated stars to make regions, I have the solution.
            # Calling generate_puzzle again is redundant but proves it works with the official solver.
            # Let's do that to ensure "official mechanism" compatibility.

            temp_stars = self._generate_stars_heuristic()
            if temp_stars is None:
                continue

            self._regions = self._generate_regions(temp_stars)

            # Now solve using official solver
            result = generate_puzzle(
                self._factory, self._size, num_hints=0, regions=self._regions
            )

            if result is not None:
                _, self._solution_board = result
                break
        else:
            raise RuntimeError("Failed to generate StarBattle puzzle after 10 attempts")

        # 3. Puzzle State
        self._puzzle_board = [
            ["e" for _ in range(self._size)] for _ in range(self._size)
        ]

        logger.info("Reset VGRP Star Battle.")

        # Generate prompt
        self._prompt = self._prompt_generate()

        obs = Observation(
            image=self.render(),
            text=None,
            metadata={
                "text_prompt": self._prompt,
                "state_text": self._board_to_text_regions(),
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
                "text_prompt": self._prompt,
                "state_text": self._board_to_text_regions(),
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

    def _generate_stars_heuristic(self) -> list[list[str]] | None:
        # Simplified generation for seeding regions:
        attempts = 0
        while attempts < 100:
            board = [["e" for _ in range(self._size)] for _ in range(self._size)]
            # Try to place row by row
            success = True
            for r in range(self._size):
                placed = 0
                cols = list(range(self._size))
                np.random.shuffle(cols)
                for c in cols:
                    # Check safety (simplified copy of backtracking logic inside env?
                    # No, we can use a local helper or just inline simple checks)
                    # Check row, col, neighbors
                    if board[r].count("s") >= self._stars_per_group:
                        break
                    col_count = sum(1 for i in range(self._size) if board[i][c] == "s")
                    if col_count >= self._stars_per_group:
                        continue

                    safe = True
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = r + dr, c + dc
                            if (
                                0 <= nr < self._size
                                and 0 <= nc < self._size
                                and board[nr][nc] == "s"
                            ):
                                safe = False
                                break
                        if not safe:
                            break

                    if safe:
                        board[r][c] = "s"
                        placed += 1
                        if placed == self._stars_per_group:
                            break
                if placed < self._stars_per_group:
                    success = False
                    break
            if success:
                return board
            attempts += 1
        return None

    def _generate_regions(self, solution: list[list[str]]) -> list[list[int]]:
        regions = [[-1 for _ in range(self._size)] for _ in range(self._size)]

        # Find stars
        stars = []
        for r in range(self._size):
            for c in range(self._size):
                if solution[r][c] == "s":
                    stars.append((r, c))

        # We need K regions (usually K=Size). Each region needs N stars.
        # So we group stars into K groups, each size N.
        # If N=1, each star is a seed for one region.
        # If N=2, pair stars up.

        np.random.shuffle(stars)
        seeds = []

        for i in range(0, len(stars), self._stars_per_group):
            chunk = stars[i : i + self._stars_per_group]
            seeds.append(chunk)

        # Grow regions
        # Use a queue for each region
        queues = []
        for region_id, seed_cells in enumerate(seeds):
            q = []
            for r, c in seed_cells:
                regions[r][c] = region_id
                q.append((r, c))
            queues.append(q)

        # Round-robin expansion
        active = True
        while active:
            active = False
            for i in range(len(queues)):
                if not queues[i]:
                    continue

                # Expand one step
                # Pick random cell from queue? Or BFS order
                # To make shapes irregular, maybe random pick
                curr_idx = np.random.randint(0, len(queues[i]))
                r, c = queues[i][curr_idx]  # peek

                # Find neighbors
                valid_neighbors = []
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < self._size
                        and 0 <= nc < self._size
                        and regions[nr][nc] == -1
                    ):
                        valid_neighbors.append((nr, nc))

                if valid_neighbors:
                    nr, nc = valid_neighbors[np.random.randint(0, len(valid_neighbors))]
                    regions[nr][nc] = i
                    queues[i].append((nr, nc))
                    active = True
                else:
                    # No growth from this cell
                    queues[i].pop(curr_idx)
                    if queues[i]:
                        active = True  # still has cells

        # Fill any remaining -1 (should be few if any)
        for r in range(self._size):
            for c in range(self._size):
                if regions[r][c] == -1:
                    # Assign to neighbor
                    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        nr, nc = r + dr, c + dc
                        if (
                            0 <= nr < self._size
                            and 0 <= nc < self._size
                            and regions[nr][nc] != -1
                        ):
                            regions[r][c] = regions[nr][nc]
                            break
                    if regions[r][c] == -1:
                        regions[r][c] = 0  # fallback

        return regions

    def _check_solution(self, answer_board: list[list[str]]) -> bool:
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
        game_state = {"board": answer_board, "regions": self._regions}
        return self._factory.check(game_state)

    def _board_to_text(self, board: list[list[str]]) -> str:
        return "\n".join(" ".join(row) for row in board)

    def _board_to_text_regions(self) -> str:
        lines = []
        lines.append("Regions Grid:")
        for row in self._regions:
            lines.append(" ".join(f"{x:2d}" for x in row))
        return "\n".join(lines)

    def _text_to_board(self, text: str) -> list[list[str]]:
        lines = text.strip().split("\n")
        board = []
        for line in lines:
            line = line.strip()
            if not line or "Regions" in line:
                continue
            row = []
            for val in line.split():
                val = val.lower()
                if val in ["s", "e"]:
                    row.append(val)
                else:
                    row.append("e")
            if row:
                board.append(row)
        return board

    def render(self) -> Image.Image | list[Image.Image] | None:
        size_px = self._cell_px * self._size + 2 * self._padding
        img = Image.new("RGB", (size_px, size_px), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Draw region backgrounds (optional, faint colors)
        colors = [
            (255, 240, 240),
            (240, 255, 240),
            (240, 240, 255),
            (255, 255, 240),
            (255, 240, 255),
            (240, 255, 255),
            (250, 250, 230),
            (230, 250, 250),
        ]

        for r in range(self._size):
            for c in range(self._size):
                rid = self._regions[r][c]
                color = colors[rid % len(colors)]
                x = self._padding + c * self._cell_px
                y = self._padding + r * self._cell_px
                draw.rectangle([x, y, x + self._cell_px, y + self._cell_px], fill=color)

        # Draw thick region borders
        for r in range(self._size):
            for c in range(self._size):
                rid = self._regions[r][c]
                x = self._padding + c * self._cell_px
                y = self._padding + r * self._cell_px

                # Top
                if r == 0 or self._regions[r - 1][c] != rid:
                    draw.line([(x, y), (x + self._cell_px, y)], fill=(0, 0, 0), width=3)
                else:
                    draw.line(
                        [(x, y), (x + self._cell_px, y)], fill=(100, 100, 100), width=1
                    )

                # Bottom
                if r == self._size - 1 or self._regions[r + 1][c] != rid:
                    draw.line(
                        [
                            (x, y + self._cell_px),
                            (x + self._cell_px, y + self._cell_px),
                        ],
                        fill=(0, 0, 0),
                        width=3,
                    )

                # Left
                if c == 0 or self._regions[r][c - 1] != rid:
                    draw.line([(x, y), (x, y + self._cell_px)], fill=(0, 0, 0), width=3)
                else:
                    draw.line(
                        [(x, y), (x, y + self._cell_px)], fill=(100, 100, 100), width=1
                    )

                # Right
                if c == self._size - 1 or self._regions[r][c + 1] != rid:
                    draw.line(
                        [
                            (x + self._cell_px, y),
                            (x + self._cell_px, y + self._cell_px),
                        ],
                        fill=(0, 0, 0),
                        width=3,
                    )

        return img

"""Thermometers single-turn environment backed by VGRP-Bench."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation, get_logger
from gym_v.utils.vgrp_base import Constraint, PuzzleFactory
from gym_v.utils.vgrp_utils import generate_puzzle

logger = get_logger()


class ConstraintThermometerFill(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_thermometer_fill"

    def check(self, game_state: dict[str, Any]) -> bool:
        board = game_state["board"]
        thermometers = game_state.get("clues", {}).get("thermometers", [])
        thermometer_positions = {(r, c) for therm in thermometers for r, c in therm}

        for i in range(len(board)):
            for j in range(len(board[i])):
                if (i, j) not in thermometer_positions and board[i][j] == "s":
                    return False

        for thermometer in thermometers:
            first_empty = -1
            for i, (r, c) in enumerate(thermometer):
                if board[r][c] == "e":
                    first_empty = i
                    break
            if first_empty != -1:
                for i, (r, c) in enumerate(thermometer):
                    if i > first_empty and board[r][c] == "s":
                        return False
        return True


class ConstraintThermometerCount(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_thermometer_count"

    def check(self, game_state: dict[str, Any]) -> bool:
        board = game_state["board"]
        clues = game_state.get("clues", None)
        if not clues:
            return True
        size = len(board)
        row_counts = clues.get("row_counts")
        col_counts = clues.get("col_counts")

        if row_counts is None or col_counts is None:
            return True

        for i in range(size):
            row_selected = sum(1 for j in range(size) if board[i][j] == "s")
            row_undefined = sum(1 for j in range(size) if board[i][j] == 0)
            if 0 not in board[i]:
                if row_selected != row_counts[i]:
                    return False
            else:
                if row_selected > row_counts[i]:
                    return False
                if row_selected + row_undefined < row_counts[i]:
                    return False

        for j in range(size):
            col_selected = sum(1 for i in range(size) if board[i][j] == "s")
            col_undefined = sum(1 for i in range(size) if board[i][j] == 0)
            if all(board[i][j] != 0 for i in range(size)):
                if col_selected != col_counts[j]:
                    return False
            else:
                if col_selected > col_counts[j]:
                    return False
                if col_selected + col_undefined < col_counts[j]:
                    return False
        return True


class ThermometersPuzzleFactory(PuzzleFactory):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.game_name = "thermometers"
        self.size = size
        self.constraints = [ConstraintThermometerFill(), ConstraintThermometerCount()]
        self.all_possible_values = ["e", "s"]

    def get_possible_values(
        self, game_state: dict[str, Any], row: int, col: int
    ) -> list[str]:
        possible_values = []
        board = game_state["board"]
        original_value = board[row][col]
        for value in self.all_possible_values:
            board[row][col] = value
            if self.check(game_state):
                possible_values.append(value)
        board[row][col] = original_value
        return possible_values


class ThermometersEnv(Env):
    # Meta: source=VGRP, category=logic, turn=single
    """Thermometers puzzle using VGRP-Bench's Thermometers puzzle generator.

    Fill thermometers from the bulb according to row/column clues.

    Args:
        size: Grid size (default 5)
        num_hints: Not used (clues derived from solution)
        cell_px: Size of each cell in pixels for rendering
        padding: Padding around the grid in pixels for rendering
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""You are given a {size}×{size} Thermometers puzzle.

Rules:
+ Fill thermometers starting from the bulb (marked with a red circle in the image)
+ Each thermometer can be filled from 0 cells to its full length
+ If filling, must start from the bulb and fill continuously (no gaps allowed)
+ Only cells that are part of a thermometer can be filled
+ Numbers on the left indicate how many filled cells are required in each row
+ Numbers on the top indicate how many filled cells are required in each column

Clues:
Row clues (filled cells per row): {row_clues}
Column clues (filled cells per column): {col_clues}
Number of thermometers: {num_thermometers}

Thermometer positions (bulb listed first, then cells in order):
{thermometer_positions}

**Output Format:**
Your answer should be a {size}×{size} grid where:
- Use 's' for filled (shaded) cells
- Use 'e' for empty cells
- Separate values with spaces within rows
- Separate rows with newlines

Example output for a 5×5 puzzle:
e e s s e
s s e e e
e e e e e
s e s s e
e e e e e
"""

    def __init__(
        self,
        size: int = 5,
        num_hints: int = 0,
        cell_px: int = 60,
        padding: int = 50,
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
        self._puzzle_board: list[list[str]] | None = None
        self._solution_board: list[list[str]] | None = None
        self._thermometers: list[list[tuple[int, int]]] | None = None
        self._row_counts: list[int] | None = None
        self._col_counts: list[int] | None = None
        self._factory = ThermometersPuzzleFactory(size)
        self._prompt: str | None = None

    @property
    def description(self) -> str:
        """Return description for Thermometers puzzle."""
        return dedent(f"""
            Solve this {self._size}x{self._size} Thermometers puzzle.

            In the image:
            - Thermometer shapes are shown with bulbs (red circles) at one end
            - Numbers on the left show the required number of filled cells in each row
            - Numbers on top show the required number of filled cells in each column

            Rules:
            1. Each thermometer can be partially or fully filled, starting from the bulb end.
            2. Filling must be continuous from the bulb - you cannot skip cells.
            3. A thermometer can also be left completely empty.
            4. The total filled cells in each row must equal the row clue number.
            5. The total filled cells in each column must equal the column clue number.

            Output format: {self._size} lines, each with {self._size} characters separated by spaces.
            Use 's' for filled cells and 'e' for empty cells.
            Example for a 3x3 grid:
            s e e
            s s e
            e s s
        """).strip()

    def _prompt_generate(self) -> str:
        """Generate complete text prompt for the puzzle."""
        row_clues = " ".join(map(str, self._row_counts))
        col_clues = " ".join(map(str, self._col_counts))
        thermometer_positions = self._thermometers_to_text()
        return self.prompt_template.format(
            size=self._size,
            row_clues=row_clues,
            col_clues=col_clues,
            num_thermometers=len(self._thermometers),
            thermometer_positions=thermometer_positions,
        )

    def _thermometers_to_text(self) -> str:
        """Convert thermometer positions to text description."""
        lines = []
        for i, thermo in enumerate(self._thermometers):
            cells = " -> ".join(f"({r + 1},{c + 1})" for r, c in thermo)
            lines.append(f"Thermometer {i + 1}: {cells}")
        return "\n".join(lines)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed, options=options)
        self._seed = seed

        # Generate thermometers configuration
        self._thermometers = self._generate_thermometers()

        # Generate solution using generate_puzzle
        # We need to pass the thermometers structure to the factory constraints
        # Note: ConstraintThermometerCount uses row_counts/col_counts.
        # If we pass 0 counts, it will try to generate empty board?
        # No, "If board is incomplete, check <= count". "If complete, check == count".
        # If we pass 0 counts, it will enforce empty solution.
        # But we want to GENERATE a filled solution.
        # Catch 22: We need counts to generate solution, but we calculate counts FROM solution.
        # Solution: Use random filling (my custom logic) OR
        # Temporarily disable Count constraint by passing None?
        # vgrp_base checks: `if not clues: return True`.
        # So if we pass only thermometers in clues (or just don't pass counts),
        # ConstraintThermometerCount might fail key error or we need to handle it.
        # In vgrp_base: `row_counts = clues["row_counts"]`. It will crash if missing.
        # So we MUST calculate counts first? No, we don't know solution.

        # We can implement a "relaxed" generation where we only check ThermometerFill.
        # But we can't easily modify the factory instance's constraints dynamically without side effects?
        # Actually we can:
        original_constraints = self._factory.constraints
        # Filter out Count constraint
        self._factory.constraints = [
            c for c in original_constraints if c.name != "constraint_thermometer_count"
        ]

        # Need to ensure clues dict is present for ConstraintThermometerFill
        clues_only_thermo = {"thermometers": self._thermometers}
        result = generate_puzzle(
            self._factory, self._size, num_hints=0, clues=clues_only_thermo
        )

        # Restore constraints
        self._factory.constraints = original_constraints

        if result is None:
            raise RuntimeError("Failed to generate Thermometers solution")

        _, self._solution_board = result

        # Calculate row/col counts from solution
        self._row_counts = [
            sum(1 for cell in row if cell == "s") for row in self._solution_board
        ]
        self._col_counts = [
            sum(1 for i in range(self._size) if self._solution_board[i][j] == "s")
            for j in range(self._size)
        ]

        self._puzzle_board = [[0 for _ in range(self._size)] for _ in range(self._size)]

        logger.info("Reset VGRP Thermometers.")

        # Generate prompt
        self._prompt = self._prompt_generate()

        obs = Observation(
            image=self.render(),
            text=self.description,
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

    def _generate_thermometers(self) -> list[list[tuple[int, int]]]:
        """Generate random thermometer configurations."""
        thermometers = []
        used_cells = set()

        # Generate 3-5 thermometers
        num_thermos = self.np_random.integers(3, min(6, self._size + 1))

        for _ in range(num_thermos):
            # Random starting position
            attempts = 0
            while attempts < 50:
                start_r = self.np_random.integers(0, self._size)
                start_c = self.np_random.integers(0, self._size)

                if (start_r, start_c) in used_cells:
                    attempts += 1
                    continue

                # Build thermometer path
                thermo = [(start_r, start_c)]
                used_cells.add((start_r, start_c))

                # Random length 2-4
                length = self.np_random.integers(2, min(5, self._size))

                # Random direction
                for _step in range(1, length):
                    last_r, last_c = thermo[-1]

                    # Try directions: right, down, left, up
                    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
                    self.np_random.shuffle(directions)

                    added = False
                    for dr, dc in directions:
                        new_r, new_c = last_r + dr, last_c + dc
                        if (
                            0 <= new_r < self._size
                            and 0 <= new_c < self._size
                            and (new_r, new_c) not in used_cells
                        ):
                            thermo.append((new_r, new_c))
                            used_cells.add((new_r, new_c))
                            added = True
                            break

                    if not added:
                        break

                if len(thermo) >= 2:
                    thermometers.append(thermo)
                    break
                else:
                    # Remove from used if failed
                    for cell in thermo:
                        used_cells.discard(cell)

                attempts += 1

        return thermometers if thermometers else [[(0, 0), (0, 1)]]

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
        terminated = True
        truncated = False
        info = {
            "oracle_answer": self._board_to_text(self._solution_board),
        }
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

        # 2. VGRP Check
        game_state = {
            "board": answer_board,
            "clues": {
                "thermometers": self._thermometers,
                "row_counts": self._row_counts,
                "col_counts": self._col_counts,
            },
        }
        return self._factory.check(game_state)

    def _board_to_text_with_clues(self) -> str:
        """Convert board to text with clues."""
        lines = []
        lines.append(f"Row clues: {' '.join(map(str, self._row_counts))}")
        lines.append(f"Col clues: {' '.join(map(str, self._col_counts))}")
        lines.append("")
        lines.append(f"Thermometers count: {len(self._thermometers)}")
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
            if not line or ":" in line:
                continue
            row = []
            for val in line.split():
                val = val.strip().lower()
                if val in ["s", "e"]:
                    row.append(val)
                else:
                    row.append("e")
            if row:
                board.append(row)
        return board

    def render(self) -> Image.Image | list[Image.Image] | None:
        return self._render_thermometers(
            self._puzzle_board,
            self._thermometers,
            self._row_counts,
            self._col_counts,
            cell_px=self._cell_px,
            padding=self._padding,
        )

    def _render_thermometers(
        self,
        puzzle: list[list[str | int]],
        thermometers: list[list[tuple[int, int]]],
        row_counts: list[int],
        col_counts: list[int],
        cell_px: int = 60,
        padding: int = 50,
        bg: tuple[int, int, int] = (245, 245, 250),
        fg: tuple[int, int, int] = (20, 20, 20),
        grid: tuple[int, int, int] = (180, 180, 200),
    ) -> Image.Image:
        n = self._size
        size = padding * 2 + cell_px * n
        img = Image.new("RGB", (size, size), bg)
        draw = ImageDraw.Draw(img)

        font_path = self.assets_dir / "DejaVuSans.ttf"
        if font_path.exists():
            font = ImageFont.truetype(str(font_path), int(cell_px * 0.4))
        else:
            font = ImageFont.load_default()

        # Draw thermometers with gradient effect
        for thermo_idx, thermo in enumerate(thermometers):
            # Alternate colors for different thermometers
            colors = [
                (180, 210, 255),  # Light blue
                (255, 210, 180),  # Light orange
                (210, 255, 180),  # Light green
                (255, 180, 210),  # Light pink
                (210, 180, 255),  # Light purple
            ]
            thermo_color = colors[thermo_idx % len(colors)]

            for idx, (r, c) in enumerate(thermo):
                x = padding + c * cell_px
                y = padding + r * cell_px

                # Draw thermometer tube with rounded corners and shadow
                # Shadow
                draw.rectangle(
                    [x + 3, y + 3, x + cell_px - 1, y + cell_px - 1],
                    fill=(200, 200, 210),
                    outline=None,
                )
                # Main tube
                draw.rounded_rectangle(
                    [x + 1, y + 1, x + cell_px - 3, y + cell_px - 3],
                    radius=8,
                    fill=thermo_color,
                    outline=(120, 120, 150),
                    width=2,
                )

                # Draw bulb at start (glossy circle with highlight)
                if idx == 0:
                    margin = cell_px // 5
                    # Shadow
                    draw.ellipse(
                        [
                            x + margin + 2,
                            y + margin + 2,
                            x + cell_px - margin,
                            y + cell_px - margin,
                        ],
                        fill=(180, 80, 80),
                    )
                    # Main bulb
                    draw.ellipse(
                        [
                            x + margin,
                            y + margin,
                            x + cell_px - margin - 2,
                            y + cell_px - margin - 2,
                        ],
                        fill=(255, 100, 100),
                        outline=(180, 50, 50),
                        width=3,
                    )
                    # Highlight for glossy effect
                    highlight_size = cell_px // 8
                    draw.ellipse(
                        [
                            x + margin + cell_px // 6,
                            y + margin + cell_px // 6,
                            x + margin + cell_px // 6 + highlight_size,
                            y + margin + cell_px // 6 + highlight_size,
                        ],
                        fill=(255, 200, 200),
                    )

        # Draw grid
        for i in range(n + 1):
            x = padding + i * cell_px
            y = padding + i * cell_px
            draw.line([(padding, y), (padding + n * cell_px, y)], fill=grid, width=2)
            draw.line([(x, padding), (x, padding + n * cell_px)], fill=grid, width=2)

        # Draw row clues (left)
        for i in range(n):
            text = str(row_counts[i])
            y = padding + i * cell_px + cell_px // 2
            bbox = draw.textbbox((0, 0), text, font=font)
            text_height = bbox[3] - bbox[1]
            draw.text((padding // 3, y - text_height // 2), text, fill=fg, font=font)

        # Draw col clues (top)
        for j in range(n):
            text = str(col_counts[j])
            x = padding + j * cell_px + cell_px // 2
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            draw.text((x - text_width // 2, padding // 3), text, fill=fg, font=font)

        return img

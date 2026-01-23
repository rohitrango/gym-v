"""Binario (No Adjacency Requirement) puzzle environment for gym-v (self-contained)."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation
from gym_v.logger import get_logger

logger = get_logger()


class RLVEBinarioNoAdjacencyRequirementEnv(Env):
    """RLVE Binario (No Adjacency Requirement) puzzle as a single-turn environment.

    This is a simplified version of Binario that only requires:
    1. No three consecutive identical digits in any row or column
    2. Each row has equal number of 0s and 1s
    3. Each column has equal number of 0s and 1s
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""You are given a (2 × {N}) × (2 × {M}) matrix. Each cell contains either '0', '1', or '*' ('*' means the cell is empty). Please fill all '*' cells with either '0' or '1' such that:
1. Each **row** contains exactly {M} '0's and {M} '1's.
2. Each **column** contains exactly {N} '0's and {N} '1's.

The matrix is given in **row-major order**, with each row represented as a string of '0', '1', and '*':
{matrix}

**Output Format:** Output (2 × {N}) lines, each containing (2 × {M}) characters, where each character is either '0' or '1'. The output should match the format of the input (i.e., one row per line, no separators)."""

    def __init__(
        self,
        max_n_m: int = 4,
        sparsity: float = 0.5,
        cell_px: int = 56,
        padding: int = 24,
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._max_n_m = max_n_m
        self._sparsity = sparsity
        self._cell_px = cell_px
        self._padding = padding
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._matrix: list[str] | None = None
        self._N: int | None = None
        self._M: int | None = None
        self._prompt: str | None = None
        self._reference_answer: str | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        if self._N and self._M:
            rows = 2 * self._N
            cols = 2 * self._M
            size_hint = f"{rows} x {cols}"
        else:
            size_hint = "(2N) x (2M)"
        return dedent(
            f"""
            Binario puzzle rules (simplified - no adjacency requirement):
            1) Fill all empty cells ('*') with '0' or '1'.
            2) Each row must contain exactly M '0's and M '1's (equal amounts).
            3. Each column must contain exactly N '0's and N '1's (equal amounts).
            4) No three consecutive identical digits in any row or column.

            In the image:
            - The grid is {size_hint}
            - Pre-filled cells are shown with their values (0 or 1)
            - Empty cells are shown with '?'
            - Blue cells represent '0', orange cells represent '1'

            Output format: (2N) lines with (2M) characters each ('0' or '1'), no separators.
            """
        ).strip()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        super().reset(seed=seed)

        self._generate()
        self._prompt = self._prompt_generate()
        self._last_image = self.render()

        obs = Observation(
            image=self._last_image,
            text=self._prompt,
            metadata={
                "rlve_prompt": self._prompt,
                "rlve_reference_answer": self._reference_answer,
            },
        )
        info = {
            "reference_answer": self._reference_answer,
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
        reward = float(self._score_answer(action_str))
        obs = Observation(
            image=self._last_image,
            text=None,
            metadata={
                "rlve_prompt": self._prompt,
                "rlve_reference_answer": self._reference_answer,
            },
        )
        info = {
            "reference_answer": self._reference_answer,
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

    def _generate(self) -> None:
        """Generate a valid Binario puzzle by creating a solution first."""
        max_n_m = self._max_n_m
        if max_n_m < 2:
            raise ValueError("max_n_m must be >= 2")

        N = int(self.np_random.integers(2, max_n_m + 1))
        M = int(self.np_random.integers(2, max_n_m + 1))
        self._N = N
        self._M = M

        # Generate a valid solution using permutations (checkerboard pattern)
        row_permutation = list(range(2 * N))
        col_permutation = list(range(2 * M))
        self.np_random.shuffle(row_permutation)
        self.np_random.shuffle(col_permutation)

        # Create solution matrix
        matrix = [
            [str((row_permutation[i] + col_permutation[j]) % 2) for j in range(2 * M)]
            for i in range(2 * N)
        ]
        self._reference_answer = "\n".join("".join(row) for row in matrix)

        # Apply sparsity to create puzzle
        sparsity = self._sparsity
        if not (0 < sparsity < 1):
            raise ValueError("sparsity must be between 0 and 1")

        num_cells = (2 * N) * (2 * M)
        num_empty = max(1, int(num_cells * sparsity))
        empty_cells = self.np_random.choice(num_cells, size=num_empty, replace=False)

        for cell in empty_cells:
            row, column = divmod(cell, 2 * M)
            matrix[row][column] = "*"

        self._matrix = ["".join(row) for row in matrix]

    def _prompt_generate(self) -> str:
        """Generate the prompt string for the puzzle."""
        if self._matrix is None or self._N is None or self._M is None:
            raise RuntimeError("No matrix generated")
        return self.prompt_template.format(
            N=self._N,
            M=self._M,
            matrix="\n".join(self._matrix),
        )

    def _process(self, answer: str | None) -> list[str] | None:
        """Process the answer string into a list of rows."""
        if answer is None:
            return None
        answer = answer.strip()
        try:
            matrix = []
            for line in answer.splitlines():
                line = line.strip()
                if line:
                    matrix.append(line.strip())
            return matrix
        except ValueError:
            return None

    def _score_answer(self, answer: str) -> float:
        """Score the answer based on correctness.

        Returns:
            -1.0: wrong format
            -0.5: invalid solution (changed pre-filled cells)
            0.0: wrong solution (doesn't satisfy constraints)
            1.0: correct solution
        """
        processed_result = self._process(answer)
        if processed_result is None:
            return -1.0

        N, M = self._N, self._M
        solution = processed_result

        # Check format
        if len(solution) != 2 * N or any(len(row) != 2 * M for row in solution):
            return -1.0
        for row in solution:
            if not all(c in "01" for c in row):
                return -1.0

        # Check that pre-filled cells are not changed
        for row, original_row in zip(solution, self._matrix):
            for cell, original_cell in zip(row, original_row):
                if original_cell != "*" and cell != original_cell:
                    return -0.5

        # Check row constraints (equal 0s and 1s)
        for i in range(2 * N):
            if solution[i].count("1") != solution[i].count("0"):
                return 0.0
            if solution[i].count("1") != M or solution[i].count("0") != M:
                return 0.0

        # Check column constraints (equal 0s and 1s)
        for j in range(2 * M):
            count_ones = sum(solution[i][j] == "1" for i in range(2 * N))
            count_zeros = sum(solution[i][j] == "0" for i in range(2 * N))
            if count_ones != count_zeros:
                return 0.0
            if count_ones != N or count_zeros != N:
                return 0.0

        return 1.0

    def render(self) -> Image.Image | list[Image.Image] | None:
        """Render the Binario puzzle as an image."""
        if self._matrix is None or self._N is None or self._M is None:
            raise RuntimeError("No matrix generated")

        rows = 2 * self._N
        cols = 2 * self._M
        cell_px = self._cell_px
        padding = self._padding

        width = padding * 2 + cols * cell_px
        height = padding * 2 + rows * cell_px
        img = Image.new("RGB", (width, height), (250, 250, 250))
        draw = ImageDraw.Draw(img)

        # Load font
        font_path = None
        font_path_candidate = self.assets_dir / "DejaVuSans.ttf"
        if font_path_candidate.exists():
            font_path = str(font_path_candidate)

        if font_path:
            font = ImageFont.truetype(font_path, int(cell_px * 0.45))
        else:
            font = ImageFont.load_default()

        # Define colors for 0 and 1
        color_0 = (173, 216, 230)  # Light blue for 0
        color_1 = (255, 200, 124)  # Light orange for 1
        color_empty = (255, 255, 255)  # White for empty cells

        # Draw cells with background colors
        for r in range(rows):
            for c in range(cols):
                x = padding + c * cell_px
                y = padding + r * cell_px
                cell_value = self._matrix[r][c]

                # Fill cell background based on value
                if cell_value == "0":
                    fill_color = color_0
                elif cell_value == "1":
                    fill_color = color_1
                else:
                    fill_color = color_empty

                draw.rectangle(
                    [x, y, x + cell_px, y + cell_px],
                    fill=fill_color,
                    outline=None,
                )

        # Draw grid lines
        for r in range(rows + 1):
            y = padding + r * cell_px
            draw.line(
                (padding, y, padding + cols * cell_px, y), fill=(30, 30, 30), width=2
            )
        for c in range(cols + 1):
            x = padding + c * cell_px
            draw.line(
                (x, padding, x, padding + rows * cell_px), fill=(30, 30, 30), width=2
            )

        # Draw cell values
        for r in range(rows):
            for c in range(cols):
                cell_value = self._matrix[r][c]
                if cell_value == "*":
                    v = "?"
                else:
                    v = cell_value

                bbox = draw.textbbox((0, 0), v, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                cx = padding + c * cell_px + cell_px // 2
                cy = padding + r * cell_px + cell_px // 2
                draw.text((cx - tw // 2, cy - th // 2), v, fill=(10, 10, 10), font=font)

        return img

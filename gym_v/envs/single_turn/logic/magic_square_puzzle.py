"""Magic square puzzle environment for gym-v (self-contained)."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation
from gym_v.logger import get_logger

logger = get_logger()


def magic_square(n: int) -> np.ndarray:
    """Generate a magic square of size n x n."""
    if n == 1:
        return np.array([[1]], dtype=int)

    if n % 2 == 1:
        return _magic_odd(n)
    elif n % 4 == 0:
        return _magic_doubly_even(n)
    else:
        raise NotImplementedError(
            "Magic square for singly even n (e.g., 6, 10) is not implemented."
        )


def _magic_odd(n: int) -> np.ndarray:
    """Generate magic square for odd n using Siamese method."""
    magic = np.zeros((n, n), dtype=int)
    num = 1
    i, j = 0, n // 2
    while num <= n * n:
        magic[i, j] = num
        num += 1
        ni, nj = (i - 1) % n, (j + 1) % n
        if magic[ni, nj] != 0:
            i = (i + 1) % n
        else:
            i, j = ni, nj
    return magic


def _magic_doubly_even(n: int) -> np.ndarray:
    """Generate magic square for doubly even n."""
    magic = np.arange(1, n * n + 1, dtype=int).reshape(n, n)
    for i in range(n):
        for j in range(n):
            if (i % 4 == j % 4) or ((i % 4) + (j % 4) == 3):
                magic[i, j] = n * n + 1 - magic[i, j]
    return magic


class MagicSquarePuzzleEnv(Env):
    # Meta: source=RLVE, category=logic, turn=single
    """RLVE Magic Square puzzle as a single-turn environment."""

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""Given a grid of size {N} × {N} filled with integers, some cells may be empty (represented by `0`). Please complete the grid to form a **magic square**, such that:
1. Each integer from `1` to `{N}^2` appears **exactly once**.
2. The sum of each row, each column, and both main diagonals is equal to {N} * ({N}^2 + 1) / 2 = {magic_constant}.

The grid is given as follows:
{grid}

**Output Format:** Your final answer should contain {N} lines, each with {N} numbers, separated by spaces. The numbers should represent the completed magic square in **row-major order**, matching the format of the given input."""

    def __init__(
        self,
        min_n: int = 3,
        max_n: int = 5,
        sparsity: float = 0.5,
        cell_px: int = 64,
        padding: int = 32,
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._min_n = min_n
        self._max_n = max_n
        self._sparsity = sparsity
        self._cell_px = cell_px
        self._padding = padding
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._N: int | None = None
        self._grid: list[list[int]] | None = None
        self._reference_grid: list[list[int]] | None = None
        self._prompt: str | None = None
        self._oracle_answer: str | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        if self._N:
            size_hint = f"{self._N} x {self._N}"
            magic_constant = self._N * (self._N * self._N + 1) // 2
        else:
            size_hint = "N x N"
            magic_constant = "target sum"

        return dedent(
            f"""
            Given a grid filled with integers where some cells may be empty, complete
            the grid to form a magic square.

            Magic square puzzle rules:
            1) Fill the grid so each integer from 1 to N^2 appears exactly once.
            2) Each row, column, and both main diagonals must sum to the magic constant: {magic_constant}.
            3) Some cells are pre-filled, others are empty (marked with '0' or '?').

            In the image:
            - Grid size: {size_hint}
            - Pre-filled cells show their numbers
            - Empty cells are marked with '?'
            - Row sums are shown on the right (in blue)
            - Column sums are shown at the bottom (in green)
            - Diagonal sums are indicated in the corners (in red)

            Output Format: Your final answer should contain N lines, each with N
            numbers, separated by spaces. The numbers should represent the completed
            magic square in row-major order, matching the format of the given input.
            """
        ).strip()

    def _get_state_text(self) -> str:
        """Return the text representation of the current puzzle state."""
        if self._prompt is None:
            return ""
        return self._prompt

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        super().reset(seed=seed)

        self._generate()
        self._prompt = self._prompt_generate()
        self._last_image = self.render()

        state_text = self._get_state_text()
        obs = Observation(
            image=self._last_image,
            text=None,
            metadata={
                "state_text": state_text,
                "text_prompt": self._prompt,
            },
        )
        info = {
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
        action_str = action[agent_id]
        reward = float(self._score_answer(action_str))
        state_text = self._get_state_text()
        obs = Observation(
            image=self._last_image,
            text=None,
            metadata={
                "state_text": state_text,
                "text_prompt": self._prompt,
            },
        )
        info = {
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

    def _generate(self) -> None:
        """Generate a magic square puzzle with some cells removed."""
        # Choose N within the allowed range (only odd or doubly even)
        valid_ns = [
            n for n in range(self._min_n, self._max_n + 1) if n % 2 == 1 or n % 4 == 0
        ]
        if not valid_ns:
            raise ValueError(
                f"No valid N values between {self._min_n} and {self._max_n}"
            )

        N = int(self.np_random.choice(valid_ns))
        self._N = N

        # Generate a magic square
        grid = magic_square(N)

        # Apply transformations to randomize the magic square
        operation_distribution = [0.1, 0.1, 0.8]
        for _ in range(N * N):
            operation = self.np_random.choice(
                ["rotate", "mirror", "swap_rows"], p=operation_distribution
            )
            if operation == "rotate":
                grid = self._rotate(grid)
            elif operation == "mirror":
                grid = self._mirror(grid)
            elif operation == "swap_rows":
                row1 = int(self.np_random.integers(0, N))
                row2 = int(self.np_random.integers(0, N))
                while row1 == row2:
                    row2 = int(self.np_random.integers(0, N))
                grid = self._swap_rows(grid, row1, row2)

        # Store the reference answer
        self._reference_grid = [[int(cell) for cell in row] for row in grid]
        self._oracle_answer = "\n".join(
            " ".join(map(str, row)) for row in self._reference_grid
        )

        # Create puzzle by removing cells
        self._grid = [[int(cell) for cell in row] for row in grid]
        num_empty = max(1, int(N * N * self._sparsity))
        empty_cells = self.np_random.choice(N * N, size=num_empty, replace=False)

        for cell_idx in empty_cells:
            row, col = divmod(int(cell_idx), N)
            self._grid[row][col] = 0

    def _rotate(self, square: np.ndarray) -> np.ndarray:
        """Rotate the square 90, 180, or 270 degrees."""
        k = int(self.np_random.integers(1, 4))
        return np.rot90(square, k)

    def _mirror(self, square: np.ndarray) -> np.ndarray:
        """Mirror the square horizontally."""
        return np.fliplr(square)

    def _swap_rows(self, square: np.ndarray, i: int, j: int) -> np.ndarray:
        """Swap rows i and j, and corresponding columns to maintain magic property."""
        n = square.shape[0]
        A = square.copy()
        A[[i, j], :] = A[[j, i], :]
        c1, c2 = n - 1 - i, n - 1 - j
        A[:, [c1, c2]] = A[:, [c2, c1]]
        return square

    def _prompt_generate(self) -> str:
        """Generate the text prompt for the puzzle."""
        if self._grid is None or self._N is None:
            raise RuntimeError("No puzzle generated")

        magic_constant = self._N * (self._N * self._N + 1) // 2
        grid_str = "\n".join(" ".join(map(str, row)) for row in self._grid)

        return self.prompt_template.format(
            N=self._N,
            magic_constant=magic_constant,
            grid=grid_str,
        )

    def _process(self, answer: str | None) -> list[list[int]] | None:
        """Process the answer string into a grid."""
        if answer is None:
            return None
        answer = answer.strip()
        try:
            grid = []
            for line in answer.splitlines():
                line = line.strip()
                if line:
                    grid.append(list(map(int, line.split())))
            return grid
        except ValueError:
            return None

    def _score_answer(self, answer: str) -> float:
        """Score the answer based on correctness."""
        processed_result = self._process(answer)
        if processed_result is None:
            return 0.0
        if self._grid is None or self._N is None:
            raise RuntimeError("No puzzle generated")

        N = self._N
        solution = processed_result

        # Check format
        if len(solution) != N or any(len(row) != N for row in solution):
            return 0.0
        flat_solution = [cell for row in solution for cell in row]
        if set(flat_solution) != set(range(1, N * N + 1)):
            return 0.0
        for i in range(N):
            for j in range(N):
                if self._grid[i][j] != 0 and solution[i][j] != self._grid[i][j]:
                    return 0.0
        magic_constant = N * (N * N + 1) // 2
        satisfied = 0

        # Check rows
        for row in solution:
            if sum(row) == magic_constant:
                satisfied += 1

        # Check columns
        for j in range(N):
            if sum(solution[i][j] for i in range(N)) == magic_constant:
                satisfied += 1

        # Check main diagonal
        if sum(solution[i][i] for i in range(N)) == magic_constant:
            satisfied += 1

        # Check anti-diagonal
        if sum(solution[i][N - i - 1] for i in range(N)) == magic_constant:
            satisfied += 1

        total_constraints = 2 * N + 2
        return (satisfied / total_constraints) ** 10

    def render(self) -> Image.Image | list[Image.Image] | None:
        """Render the magic square puzzle."""
        if self._grid is None or self._N is None:
            raise RuntimeError("No puzzle generated")

        N = self._N
        cell_px = self._cell_px
        padding = self._padding

        # Extra space for sum labels
        label_space = cell_px

        width = padding * 2 + N * cell_px + label_space
        height = padding * 2 + N * cell_px + label_space

        img = Image.new("RGB", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Load font
        font_path = None
        font_path_candidate = self.assets_dir / "DejaVuSans.ttf"
        if font_path_candidate.exists():
            font_path = str(font_path_candidate)

        if font_path:
            font = ImageFont.truetype(font_path, int(cell_px * 0.4))
            small_font = ImageFont.truetype(font_path, int(cell_px * 0.3))
        else:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()

        magic_constant = N * (N * N + 1) // 2

        # Draw grid lines
        for r in range(N + 1):
            y = padding + r * cell_px
            draw.line(
                (padding, y, padding + N * cell_px, y),
                fill=(80, 80, 80),
                width=2,
            )
        for c in range(N + 1):
            x = padding + c * cell_px
            draw.line(
                (x, padding, x, padding + N * cell_px),
                fill=(80, 80, 80),
                width=2,
            )

        # Draw cell numbers or question marks
        for r in range(N):
            for c in range(N):
                if self._grid[r][c] == 0:
                    v = "?"
                    color = (150, 150, 150)
                else:
                    v = str(self._grid[r][c])
                    color = (20, 20, 20)

                bbox = draw.textbbox((0, 0), v, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                cx = padding + c * cell_px + cell_px // 2
                cy = padding + r * cell_px + cell_px // 2
                draw.text((cx - tw // 2, cy - th // 2), v, fill=color, font=font)

        # Draw row sum labels (right side, blue)
        for r in range(N):
            sum_text = str(magic_constant)
            bbox = draw.textbbox((0, 0), sum_text, font=small_font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            x = padding + N * cell_px + label_space // 2
            y = padding + r * cell_px + cell_px // 2
            draw.text(
                (x - tw // 2, y - th // 2),
                sum_text,
                fill=(30, 100, 220),
                font=small_font,
            )

        # Draw column sum labels (bottom, green)
        for c in range(N):
            sum_text = str(magic_constant)
            bbox = draw.textbbox((0, 0), sum_text, font=small_font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            x = padding + c * cell_px + cell_px // 2
            y = padding + N * cell_px + label_space // 2
            draw.text(
                (x - tw // 2, y - th // 2),
                sum_text,
                fill=(40, 160, 60),
                font=small_font,
            )

        # Draw diagonal indicators in corners (red)
        # Top-left corner for main diagonal
        diag_text = "\\"
        bbox = draw.textbbox((0, 0), diag_text, font=small_font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(
            (padding // 2 - tw // 2, padding // 2 - th // 2),
            diag_text,
            fill=(220, 50, 50),
            font=small_font,
        )

        # Top-right corner for anti-diagonal
        diag_text = "/"
        bbox = draw.textbbox((0, 0), diag_text, font=small_font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(
            (
                padding + N * cell_px + label_space // 2 - tw // 2,
                padding // 2 - th // 2,
            ),
            diag_text,
            fill=(220, 50, 50),
            font=small_font,
        )

        return img

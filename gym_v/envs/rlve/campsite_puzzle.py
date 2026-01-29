"""Campsite puzzle environment for gym-v (self-contained)."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation
from gym_v.logger import get_logger

logger = get_logger()


class RLVECampsitePuzzleEnv(Env):
    """RLVE Campsite puzzle as a single-turn environment."""

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""You are given a {N} × {M} matrix. Each cell contains either '0', '1', or '*' ('*' means the cell is empty). Please fill all '*' cells with either '0' or '1' such that:
1. No two (horizontally or vertically) adjacent cells in a row or column can both contain `1`.
2. The number of `1`s in each row (from top to bottom) is: {row_counts}.
3. The number of `1`s in each column (from left to right) is: {col_counts}.

The matrix is given in **row-major order**, with each row represented as a string of '0', '1', and '*':
{matrix}

**Output Format:** Output {N} lines, each containing {M} characters, where each character is either '0' or '1'. The output should match the format of the input (i.e., one row per line, no separators)."""

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

        self._n: int | None = None
        self._m: int | None = None
        self._matrix: list[str] | None = None
        self._row_counts: list[int] | None = None
        self._col_counts: list[int] | None = None
        self._prompt: str | None = None
        self._oracle_answer: str | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        if self._n and self._m:
            size_hint = f"{self._n} x {self._m}"
        else:
            size_hint = "N x M"
        return dedent(
            f"""
            Campsite puzzle rules:
            1) Fill all empty cells ('*') with either '0' or '1'.
            2) No two horizontally or vertically adjacent cells can both be '1'.
            3) Each row must have a specific number of '1's (given in the prompt).
            4) Each column must have a specific number of '1's (given in the prompt).

            In the image:
            - Each cell shows '0', '1', or '?' for empty cells
            - The grid is {size_hint}
            - White cells represent '0', gray cells represent '1', light blue cells are empty

            Output format: N lines with M characters ('0' or '1'), no separators.
            Do not change the pre-filled cells.
            """
        ).strip()

    def _get_state_text(self) -> str:
        """Return text representation of the campsite grid."""
        if self._matrix is None:
            return ""
        return "\n".join(self._matrix)

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
            text=state_text,
            metadata={
                "text_prompt": f"{state_text}\n\n{self.description}",
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
            text=state_text,
            metadata={
                "text_prompt": f"{state_text}\n\n{self.description}",
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
        max_n_m = self._max_n_m
        if max_n_m < 2:
            raise ValueError("max_n_m must be >= 2")

        self._n = int(self.np_random.integers(2, max_n_m + 1))
        self._m = int(self.np_random.integers(2, max_n_m + 1))
        N, M = self._n, self._m

        def generate_matrix(N: int, M: int) -> list[list[str]]:
            # Initialize the grid with None
            grid: list[list[str | None]] = [[None] * M for _ in range(N)]
            all_cells = [(i, j) for i in range(N) for j in range(M)]
            self.np_random.shuffle(all_cells)

            def backtrack(idx: int) -> bool:
                # If we've filled past the last row, we're done
                if idx == len(all_cells):
                    return True
                i, j = all_cells[idx]

                # Try placing 0 or 1 in random order
                for v in self.np_random.permutation(["0", "1"]).tolist():
                    # Check adjacency constraints in row (no adjacent 1s)
                    if j >= 1 and grid[i][j - 1] == v == "1":
                        continue
                    if j + 1 < M and grid[i][j + 1] == v == "1":
                        continue

                    # Check adjacency constraints in column
                    if i >= 1 and grid[i - 1][j] == v == "1":
                        continue
                    if i + 1 < N and grid[i + 1][j] == v == "1":
                        continue

                    # Place v
                    grid[i][j] = v

                    # Recurse
                    if backtrack(idx + 1):
                        return True

                    # Undo placement
                    grid[i][j] = None

                # No valid value at (i, j): backtrack
                return False

            if not backtrack(0):
                raise RuntimeError("Failed to generate a valid matrix")
            return grid  # type: ignore

        matrix = generate_matrix(N, M)
        self._oracle_answer = "\n".join("".join(row) for row in matrix)

        self._row_counts = [sum(int(cell == "1") for cell in row) for row in matrix]
        self._col_counts = [
            sum(int(matrix[i][j] == "1") for i in range(N)) for j in range(M)
        ]

        sparsity = self._sparsity
        if not (0 < sparsity < 1):
            raise ValueError("sparsity must be between 0 and 1")
        empty_cells = self.np_random.choice(
            N * M, size=max(1, int(N * M * sparsity)), replace=False
        )
        for cell in empty_cells:
            row, column = divmod(int(cell), M)
            matrix[row][column] = "*"
        self._matrix = ["".join(row) for row in matrix]

    def _prompt_generate(self) -> str:
        if self._matrix is None or self._n is None or self._m is None:
            raise RuntimeError("No matrix generated")
        return self.prompt_template.format(
            N=self._n,
            M=self._m,
            matrix="\n".join(self._matrix),
            row_counts=", ".join(map(str, self._row_counts)),
            col_counts=", ".join(map(str, self._col_counts)),
        )

    def _process(self, answer: str | None) -> list[str] | None:
        if answer is None:
            return None
        answer = answer.strip()
        try:
            rows = []
            for line in answer.splitlines():
                line = line.strip()
                if line:
                    rows.append(line.strip())
            return rows
        except ValueError:
            return None

    def _score_answer(self, answer: str) -> float:
        processed_result = self._process(answer)
        if processed_result is None:
            return 0.0
        N = self._n
        M = self._m
        solution = processed_result

        # Check format
        if len(solution) != N or any(len(row) != M for row in solution):
            return 0.0
        for row in solution:
            if not all(c in "01" for c in row):
                return 0.0
        for row, original_row in zip(solution, self._matrix, strict=False):
            for cell, original_cell in zip(row, original_row, strict=False):
                if original_cell != "*" and cell != original_cell:
                    return 0.0
        delta = [
            (+1, 0),
            (-1, 0),
            (0, +1),
            (0, -1),
        ]
        for i in range(N):
            for j in range(M):
                for di, dj in delta:
                    ni, nj = i + di, j + dj
                    if (
                        0 <= ni < N
                        and 0 <= nj < M
                        and solution[i][j] == solution[ni][nj] == "1"
                    ):
                        return 0.0
        row_counts = [sum(int(cell == "1") for cell in row) for row in solution]
        col_counts = [
            sum(int(solution[i][j] == "1") for i in range(N)) for j in range(M)
        ]

        satisfied = sum(
            int(answer == gold)
            for answer, gold in zip(row_counts, self._row_counts, strict=False)
        ) + sum(
            int(answer == gold)
            for answer, gold in zip(col_counts, self._col_counts, strict=False)
        )

        return (satisfied / (N + M)) ** 10

    def render(self) -> Image.Image | list[Image.Image] | None:
        if self._matrix is None or self._n is None or self._m is None:
            raise RuntimeError("No matrix generated")
        rows, cols = self._n, self._m
        cell_px = self._cell_px
        padding = self._padding

        width = padding * 2 + cols * cell_px
        height = padding * 2 + rows * cell_px
        img = Image.new("RGB", (width, height), (250, 250, 250))
        draw = ImageDraw.Draw(img)

        font_path = None
        font_path_candidate = self.assets_dir / "DejaVuSans.ttf"
        if font_path_candidate.exists():
            font_path = str(font_path_candidate)

        if font_path:
            font = ImageFont.truetype(font_path, int(cell_px * 0.45))
        else:
            font = ImageFont.load_default()

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

        # Draw cells
        for r in range(rows):
            for c in range(cols):
                cell_value = self._matrix[r][c]
                x0 = padding + c * cell_px
                y0 = padding + r * cell_px
                x1 = x0 + cell_px
                y1 = y0 + cell_px

                # Fill cell with color
                if cell_value == "0":
                    # White for 0
                    draw.rectangle(
                        [x0 + 2, y0 + 2, x1 - 2, y1 - 2], fill=(255, 255, 255)
                    )
                elif cell_value == "1":
                    # Gray for 1
                    draw.rectangle(
                        [x0 + 2, y0 + 2, x1 - 2, y1 - 2], fill=(180, 180, 180)
                    )
                else:  # "*" - empty cell
                    # Light blue for empty
                    draw.rectangle(
                        [x0 + 2, y0 + 2, x1 - 2, y1 - 2], fill=(220, 240, 255)
                    )

                # Draw text
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

"""Hitori puzzle environment for gym-v (self-contained)."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation
from gym_v.logger import get_logger

logger = get_logger()


class RLVEHitoriPuzzleEnv(Env):
    """RLVE Hitori puzzle as a single-turn environment."""

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""You are given a {N} × {M} matrix. Each cell contains an integer. Please "black out" some cells such that:
1. In each row and each column, no number appears more than once **among the remaining (non-blacked-out) cells**.
2. No two blacked-out cells are **adjacent** (horizontally or vertically).
3. All remaining cells must form a **single connected region** — you must be able to reach any remaining cell from any other by moving up, down, left, or right.

The matrix is given in **row-major order**, with each row represented as a list of integers separated by spaces:
{matrix}

**Output Format:** Output {N} lines, each containing {M} characters with no separators (also in **row-major order**). Use `.` for a remaining cell and `*` for a blacked-out cell."""

    def __init__(
        self,
        max_n_m: int = 4,
        cell_px: int = 56,
        padding: int = 24,
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._max_n_m = max_n_m
        self._cell_px = cell_px
        self._padding = padding
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._matrix: list[list[int]] | None = None
        self._prompt: str | None = None
        self._reference_answer: str | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        if self._matrix:
            rows = len(self._matrix)
            cols = len(self._matrix[0]) if self._matrix[0] else 0
            size_hint = f"{rows} x {cols}"
        else:
            size_hint = "N x M"
        return dedent(
            f"""
            Hitori puzzle rules:
            1) Mark cells with '*' to black out and '.' to keep.
            2) In each row and column, no number appears more than once among kept cells.
            3) No two blacked-out cells are adjacent (up/down/left/right).
            4) All kept cells form one connected region (4-neighbor connectivity).

            In the image:
            - Each cell shows an integer
            - The grid is {size_hint}

            Output format: N lines with M characters ('.' or '*'), no separators.
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
        max_n_m = self._max_n_m
        if max_n_m < 2:
            raise ValueError("max_n_m must be >= 2")

        N = int(self.np_random.integers(2, max_n_m + 1))
        M = int(self.np_random.integers(2, max_n_m + 1))

        def check_connected(grid, N, M):
            visited = [[False] * M for _ in range(N)]

            def DFS(x, y):
                visited[x][y] = True
                for dx, dy in [(-1, 0), (+1, 0), (0, -1), (0, +1)]:
                    nx, ny = x + dx, y + dy
                    if (
                        0 <= nx < N
                        and 0 <= ny < M
                        and not visited[nx][ny]
                        and grid[nx][ny] == "."
                    ):
                        DFS(nx, ny)

            for i in range(N):
                for j in range(M):
                    if grid[i][j] == ".":
                        DFS(i, j)
                        return all(
                            visited[_i][_j]
                            for _i in range(N)
                            for _j in range(M)
                            if grid[_i][_j] == "."
                        )
            raise ValueError("No remaining cell")

        def generate(N, M):
            matrix = [[None] * M for _ in range(N)]
            reference_answer = [["."] * M for _ in range(N)]

            all_cells = [(i, j) for i in range(N) for j in range(M)]
            self.np_random.shuffle(all_cells)

            def backtrack(idx):
                if idx == len(all_cells):
                    return True
                i, j = all_cells[idx]

                remaining_numbers = set(
                    matrix[i][_j]
                    for _j in range(M)
                    if reference_answer[i][_j] == "." and matrix[i][_j] is not None
                ) | set(
                    matrix[_i][j]
                    for _i in range(N)
                    if reference_answer[_i][j] == "." and matrix[_i][j] is not None
                )

                colors = [".", "*"]
                self.np_random.shuffle(colors)
                for color in colors:
                    if color == ".":
                        num = 0
                        while num in remaining_numbers:
                            num += 1
                        matrix[i][j] = num
                    else:
                        if not remaining_numbers:
                            continue
                        ok = True
                        for di, dj in [(-1, 0), (+1, 0), (0, -1), (0, +1)]:
                            ni, nj = i + di, j + dj
                            if (
                                0 <= ni < N
                                and 0 <= nj < M
                                and reference_answer[ni][nj] == "*"
                            ):
                                ok = False
                                break
                        if not ok:
                            continue
                        reference_answer[i][j] = "*"
                        if not check_connected(reference_answer, N, M):
                            reference_answer[i][j] = "."
                            continue
                        matrix[i][j] = int(
                            self.np_random.choice(list(remaining_numbers))
                        )
                    assert backtrack(idx + 1)
                    return True

                return False

            if not backtrack(0):
                raise RuntimeError("Failed to generate a valid matrix")
            return matrix, reference_answer

        self._matrix, reference_answer = generate(N, M)
        self._reference_answer = "\n".join("".join(row) for row in reference_answer)

    def _prompt_generate(self) -> str:
        if self._matrix is None:
            raise RuntimeError("No matrix generated")
        N = len(self._matrix)
        M = len(self._matrix[0])
        return self.prompt_template.format(
            N=N,
            M=M,
            matrix="\n".join(" ".join(map(str, row)) for row in self._matrix),
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
                    rows.append(line)
            return rows
        except ValueError:
            return None

    def _score_answer(self, answer: str) -> float:
        processed_result = self._process(answer)
        if processed_result is None:
            return -1.0

        N = len(self._matrix)
        M = len(self._matrix[0])
        solution = processed_result

        if len(solution) != N or any(len(row) != M for row in solution):
            return -1.0
        if not all(c in ".*" for row in solution for c in row):
            return -1.0

        # adjacency
        for i in range(N):
            for j in range(M):
                if solution[i][j] == "*":
                    for di, dj in [(-1, 0), (+1, 0), (0, -1), (0, +1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < N and 0 <= nj < M and solution[ni][nj] == "*":
                            return -0.5

        # connected
        if not self._check_connected(solution, N, M):
            return -0.5

        satisfied = 0
        for i in range(N):
            row_numbers = [
                self._matrix[i][j] for j in range(M) if solution[i][j] == "."
            ]
            if len(row_numbers) == len(set(row_numbers)):
                satisfied += 1
        for j in range(M):
            col_numbers = [
                self._matrix[i][j] for i in range(N) if solution[i][j] == "."
            ]
            if len(col_numbers) == len(set(col_numbers)):
                satisfied += 1

        return (satisfied / (N + M)) ** 10

    def _check_connected(self, grid, N, M):
        visited = [[False] * M for _ in range(N)]

        def DFS(x, y):
            visited[x][y] = True
            for dx, dy in [(-1, 0), (+1, 0), (0, -1), (0, +1)]:
                nx, ny = x + dx, y + dy
                if (
                    0 <= nx < N
                    and 0 <= ny < M
                    and not visited[nx][ny]
                    and grid[nx][ny] == "."
                ):
                    DFS(nx, ny)

        for i in range(N):
            for j in range(M):
                if grid[i][j] == ".":
                    DFS(i, j)
                    return all(
                        visited[_i][_j]
                        for _i in range(N)
                        for _j in range(M)
                        if grid[_i][_j] == "."
                    )
        return False

    def render(self) -> Image.Image | list[Image.Image] | None:
        if self._matrix is None:
            raise RuntimeError("No matrix generated")
        rows, cols = len(self._matrix), len(self._matrix[0])
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

        for r in range(rows):
            for c in range(cols):
                v = str(self._matrix[r][c])
                bbox = draw.textbbox((0, 0), v, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                cx = padding + c * cell_px + cell_px // 2
                cy = padding + r * cell_px + cell_px // 2
                draw.text((cx - tw // 2, cy - th // 2), v, fill=(10, 10, 10), font=font)

        return img

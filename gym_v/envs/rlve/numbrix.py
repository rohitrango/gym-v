"""Numbrix puzzle environment for gym-v (self-contained)."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation
from gym_v.envs.rlve.parameter_controllers import RLVENumbrixController
from gym_v.logger import get_logger

logger = get_logger()


class RLVENumbrixEnv(Env):
    """RLVE Numbrix puzzle as a single-turn environment."""

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""You are given a {N} × {M} matrix with some cells filled with numbers from `0` to `{NM_minus_1}`, and some cells empty (represented by `-1`). Please fill the empty cells with numbers from `0` to `{NM_minus_1}` such that:
1. Each number from `0` to `{NM_minus_1}` appears **exactly once** in the matrix.
2. Each number is **horizontally or vertically adjacent** to the next number (i.e., every number `x` is adjacent to `x + 1`).

The matrix is given as follows:
{matrix}

**Output Format:** Your final answer should contain {N} lines, each with {M} numbers, separated by spaces. The numbers should represent the completed matrix in **row-major order**, matching the format of the given input."""

    def __init__(
        self,
        max_n_m: int | None = None,
        sparsity: float | None = None,
        cell_px: int = 56,
        padding: int = 24,
        num_players: int = 1,
        difficulty: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(difficulty=difficulty, **kwargs)
        self._cell_px = cell_px
        self._padding = padding
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        # Check if explicit parameters or difficulty is provided
        self._use_explicit_params = max_n_m is not None or sparsity is not None
        self._use_difficulty = difficulty is not None

        # Initialize parameter controller only if difficulty is used
        if self._use_difficulty:
            self._parameter_controller = RLVENumbrixController(self._difficulty)
        else:
            self._parameter_controller = None

        if self._use_explicit_params:
            # Use explicit parameters (backward compatibility)
            self._max_n_m = max_n_m if max_n_m is not None else 4
            self._sparsity = sparsity if sparsity is not None else 0.5
        elif self._use_difficulty:
            # Use difficulty-based parameters
            self._apply_difficulty_parameters()
        else:
            # Use original defaults (backward compatibility)
            self._max_n_m = 4
            self._sparsity = 0.5

        self._matrix: list[list[int]] | None = None
        self._prompt: str | None = None
        self._oracle_answer: str | None = None
        self._last_image: Image.Image | None = None

    def _apply_difficulty_parameters(self) -> None:
        """Apply parameters from the controller."""
        if self._use_difficulty and self._parameter_controller is not None:
            params = self._parameter_controller.get_parameters()
            self._max_n_m = params["max_n_m"]
            self._sparsity = params["sparsity"]

    @property
    def description(self) -> str:
        if self._matrix:
            rows = len(self._matrix)
            cols = len(self._matrix[0]) if self._matrix[0] else 0
            size_hint = f"{rows} x {cols}"
            nm_minus_1 = rows * cols - 1
        else:
            size_hint = "N x M"
            nm_minus_1 = "N*M-1"
        return dedent(
            f"""
            You are given a matrix with some cells filled with numbers and some cells \
            empty (represented by -1). Please fill the empty cells such that:
            1) Each number from 0 to {nm_minus_1} appears exactly once in the matrix.
            2) Each number is horizontally or vertically adjacent to the next number \
            (i.e., every number x is adjacent to x + 1).

            In the image:
            - Pre-filled cells show numbers
            - Empty cells show "?"
            - The grid is {size_hint}

            **Output Format:** Your final answer should contain N lines, each with M \
            numbers, separated by spaces. The numbers should represent the completed \
            matrix in row-major order, matching the format of the given input.
            """
        ).strip()

    def _get_state_text(self) -> str:
        """Return text representation of the numbrix grid."""
        if self._matrix is None:
            return ""
        return "\n".join(" ".join(str(cell) for cell in row) for row in self._matrix)

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
        """Generate a Numbrix puzzle with a valid Hamiltonian path."""
        max_n_m = self._max_n_m
        if max_n_m < 2:
            raise ValueError("max_n_m must be >= 2")

        N = int(self.np_random.integers(2, max_n_m + 1))
        M = int(self.np_random.integers(2, max_n_m + 1))

        dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        def is_inside(x: int, y: int) -> bool:
            return 0 <= x < N and 0 <= y < M

        def count_unvisited_degree(x: int, y: int, visited: list[list[bool]]) -> int:
            cnt = 0
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if is_inside(nx, ny) and not visited[nx][ny]:
                    cnt += 1
            return cnt

        def check_connectivity(visited: list[list[bool]], remain: int) -> bool:
            start = None
            for i in range(N):
                for j in range(M):
                    if not visited[i][j]:
                        start = (i, j)
                        break
                if start:
                    break
            if not start:
                return True
            stack = [start]
            seen = {start}
            count = 1
            while stack:
                x, y = stack.pop()
                for dx, dy in dirs:
                    xx, yy = x + dx, y + dy
                    if (
                        is_inside(xx, yy)
                        and not visited[xx][yy]
                        and (xx, yy) not in seen
                    ):
                        seen.add((xx, yy))
                        stack.append((xx, yy))
                        count += 1
            return count == remain

        def DFS(
            step: int,
            x: int,
            y: int,
            visited: list[list[bool]],
            order: list[list[int]],
            path: list[tuple[int, int]],
        ) -> bool:
            if step == N * M:
                return True
            cand = []
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if is_inside(nx, ny) and not visited[nx][ny]:
                    cand.append((nx, ny))
            if not cand:
                return False

            # Shuffle candidates for randomness
            indices = list(range(len(cand)))
            self.np_random.shuffle(indices)
            cand = [cand[i] for i in indices]

            cand_scores = []
            for nx, ny in cand:
                deg = count_unvisited_degree(nx, ny, visited)
                cand_scores.append((deg, nx, ny))
            cand_scores.sort(key=lambda t: t[0])

            for _, nx, ny in cand_scores:
                visited[nx][ny] = True
                order[nx][ny] = step
                path.append((nx, ny))
                remain = N * M - (step + 1)
                if check_connectivity(visited, remain):
                    if DFS(step + 1, nx, ny, visited, order, path):
                        return True
                visited[nx][ny] = False
                order[nx][ny] = -1
                path.pop()
            return False

        def generate_random_hamiltonian_path() -> list[list[int]]:
            while True:
                sx = int(self.np_random.integers(0, N))
                sy = int(self.np_random.integers(0, M))
                visited = [[False] * M for _ in range(N)]
                order = [[-1] * M for _ in range(N)]
                path: list[tuple[int, int]] = []
                visited[sx][sy] = True
                order[sx][sy] = 0
                path = [(sx, sy)]
                if DFS(1, sx, sy, visited, order, path):
                    return order

        matrix = generate_random_hamiltonian_path()
        self._oracle_answer = "\n".join(" ".join(map(str, row)) for row in matrix)

        # Make a copy for the puzzle (with some cells empty)
        sparsity = self._sparsity
        if not (0 < sparsity < 1):
            raise ValueError("sparsity must be between 0 and 1")

        empty_cells = self.np_random.choice(
            N * M, size=max(1, int(N * M * sparsity)), replace=False
        )

        puzzle_matrix = [row[:] for row in matrix]
        for cell in empty_cells:
            row, column = divmod(int(cell), M)
            puzzle_matrix[row][column] = -1

        self._matrix = puzzle_matrix

    def _prompt_generate(self) -> str:
        if self._matrix is None:
            raise RuntimeError("No matrix generated")
        N = len(self._matrix)
        M = len(self._matrix[0])
        return self.prompt_template.format(
            N=N,
            M=M,
            NM_minus_1=N * M - 1,
            matrix="\n".join(" ".join(map(str, row)) for row in self._matrix),
        )

    def _process(self, answer: str | None) -> list[list[int]] | None:
        if answer is None:
            return None
        answer = answer.strip()
        try:
            matrix = []
            for line in answer.splitlines():
                line = line.strip()
                if line:
                    matrix.append(list(map(int, line.split())))
            return matrix
        except (ValueError, AttributeError):
            return None

    def _score_answer(self, answer: str) -> float:
        """Score the answer following RLVE's exact logic."""
        processed_result = self._process(answer)
        if processed_result is None:
            return 0.0
        N = len(self._matrix)
        M = len(self._matrix[0])
        solution = processed_result

        # Check format
        if len(solution) != N or any(len(row) != M for row in solution):
            return 0.0
        location: list[tuple[int, int] | None] = [None] * (N * M)
        i = 0
        for original_row, solution_row in zip(self._matrix, solution, strict=False):
            j = 0
            for original_value, solution_value in zip(
                original_row, solution_row, strict=False
            ):
                # Check that pre-filled cells match
                if original_value != -1 and original_value != solution_value:
                    return 0.0
                if not (0 <= solution_value < N * M):
                    return 0.0
                if location[solution_value] is not None:
                    return 0.0
                location[solution_value] = (i, j)
                j += 1
            i += 1

        # Count path breaks (consecutive numbers not adjacent)
        path = 1
        for value in range(N * M - 1):
            if location[value] is None or location[value + 1] is None:
                return 0.0
            x1, y1 = location[value]
            x2, y2 = location[value + 1]
            # If not adjacent (Manhattan distance != 1), it's a break
            if abs(x1 - x2) + abs(y1 - y2) != 1:
                path += 1

        # Reward based on path quality
        # (1/path)^3 - perfect solution has path=1, giving reward=1.0
        return (1.0 / path) ** 3.0

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

        # Draw numbers (or ? for empty cells)
        for r in range(rows):
            for c in range(cols):
                val = self._matrix[r][c]
                if val == -1:
                    v = "?"
                    text_color = (150, 150, 150)
                else:
                    v = str(val)
                    text_color = (10, 10, 10)

                bbox = draw.textbbox((0, 0), v, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                cx = padding + c * cell_px + cell_px // 2
                cy = padding + r * cell_px + cell_px // 2
                draw.text((cx - tw // 2, cy - th // 2), v, fill=text_color, font=font)

        return img

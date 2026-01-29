"""Grid BFS environment for gym-v (self-contained)."""

from __future__ import annotations

from collections import deque
from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation


class RLVEGridBFSEnv(Env):
    """RLVE Grid BFS as a single-turn environment."""

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""You are given a {N} × {M} grid. Each cell contains `0`, `1`, or `X`. For each cell, compute its **shortest distance** to any cell containing `1`, where distance is defined as the minimum number of steps required to move from one cell to another under the following rules:
1. You may move **up**, **down**, **left**, or **right** to an adjacent cell.
2. You **cannot** move through cells containing `X`.
3. If a cell **cannot reach** any `1`, its distance should be -1.
4. Obviously, the distance for a `1` cell is 0; the distance for an `X` cell is also -1.

The grid is given as follows:
{grid}

**Output Format:** Output {N} lines, each containing {M} integers (separated by spaces), representing the distance of each cell to the nearest `1` cell."""

    def __init__(
        self,
        max_n_m: int = 8,
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

        self._grid: list[list[str]] | None = None
        self._distances: list[list[int]] | None = None
        self._N: int = 0
        self._M: int = 0
        self._prompt: str | None = None
        self._oracle_answer: str | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        if self._grid:
            size_hint = f"{self._N} x {self._M}"
        else:
            size_hint = "N x M"
        return dedent(
            f"""
            Grid BFS (Breadth-First Search) puzzle:

            Given a grid where each cell contains:
            - '0': Empty cell (passable)
            - '1': Target cell (distance = 0)
            - 'X': Obstacle (impassable, distance = -1)

            Task: Calculate the shortest distance from each cell to the nearest target cell.

            Rules:
            1) Movement is 4-directional (up, down, left, right).
            2) Cannot move through obstacles ('X').
            3) Target cells ('1') have distance 0.
            4) Unreachable cells have distance -1.

            In the image:
            - Green cells: Targets ('1')
            - Gray cells: Obstacles ('X')
            - Blue gradient: Empty cells colored by distance (darker = farther)
            - The grid is {size_hint}

            Output format: N lines with M space-separated integers representing distances.
            """
        ).strip()

    def _get_state_text(self) -> str:
        """Return text representation of the input grid."""
        if self._grid is None:
            return ""
        return "\n".join("".join(row) for row in self._grid)

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
            text=state_text,
            metadata={
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
        max_n_m = self._max_n_m
        if max_n_m < 2:
            raise ValueError("max_n_m must be >= 2")

        N = int(self.np_random.integers(2, max_n_m + 1))
        M = int(self.np_random.integers(2, max_n_m + 1))
        self._N = N
        self._M = M

        cell_distribution = [
            int(self.np_random.integers(1, N * M + 1)) for _ in range(3)
        ]
        cell_distribution = [x / sum(cell_distribution) for x in cell_distribution]
        grid = [
            [
                self.np_random.choice(["0", "1", "X"], p=cell_distribution)
                for _ in range(M)
            ]
            for _ in range(N)
        ]

        distances = [[-1] * M for _ in range(N)]
        queue = deque()
        for i in range(N):
            for j in range(M):
                if grid[i][j] == "1":
                    distances[i][j] = 0
                    queue.append((i, j))
        while queue:
            x, y = queue.popleft()
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if (
                    0 <= nx < N
                    and 0 <= ny < M
                    and grid[nx][ny] != "X"
                    and distances[nx][ny] == -1
                ):
                    distances[nx][ny] = distances[x][y] + 1
                    queue.append((nx, ny))

        self._grid = grid
        self._distances = distances
        self._oracle_answer = "\n".join(" ".join(map(str, row)) for row in distances)

    def _prompt_generate(self) -> str:
        if self._grid is None:
            raise RuntimeError("No grid generated")
        return self.prompt_template.format(
            N=self._N,
            M=self._M,
            grid="\n".join("".join(row) for row in self._grid),
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
        except ValueError:
            return None

    def _score_answer(self, answer: str) -> float:
        processed_result = self._process(answer)
        if processed_result is None:
            return 0.0
        distance = processed_result
        if len(distance) != self._N:
            return 0.0
        if not all(len(row) == self._M for row in distance):
            return 0.0
        correct_cells = sum(
            sum(ans == gold for ans, gold in zip(answer_row, gold_row, strict=False))
            for answer_row, gold_row in zip(distance, self._distances, strict=False)
        )
        total_cells = self._N * self._M
        accuracy = correct_cells / total_cells
        return float(accuracy**10)

    def render(self) -> Image.Image | list[Image.Image] | None:
        if self._grid is None:
            raise RuntimeError("No grid generated")
        rows, cols = self._N, self._M
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
            font = ImageFont.truetype(font_path, int(cell_px * 0.40))
            font_small = ImageFont.truetype(font_path, int(cell_px * 0.25))
        else:
            font = ImageFont.load_default()
            font_small = ImageFont.load_default()

        # Find max distance for color scaling (excluding -1)
        max_dist = max(
            (
                self._distances[r][c]
                for r in range(rows)
                for c in range(cols)
                if self._distances[r][c] != -1
            ),
            default=0,
        )

        # Draw cells
        for r in range(rows):
            for c in range(cols):
                cell = self._grid[r][c]
                dist = self._distances[r][c]
                x0 = padding + c * cell_px
                y0 = padding + r * cell_px
                x1 = x0 + cell_px
                y1 = y0 + cell_px

                # Color based on cell type
                if cell == "X":
                    # Obstacle - dark gray
                    fill = (80, 80, 80)
                    text_color = (200, 200, 200)
                elif cell == "1":
                    # Target - green
                    fill = (50, 180, 50)
                    text_color = (255, 255, 255)
                else:
                    # Empty cell - blue gradient based on distance
                    if dist == -1:
                        # Unreachable - red tint
                        fill = (200, 80, 80)
                    elif max_dist > 0:
                        # Distance gradient from light blue (close) to dark blue (far)
                        intensity = 1.0 - (dist / max_dist) * 0.7
                        fill = (
                            int(200 + intensity * 55),
                            int(220 + intensity * 35),
                            255,
                        )
                    else:
                        fill = (220, 235, 255)
                    text_color = (20, 20, 20)

                draw.rectangle(
                    (x0, y0, x1, y1), fill=fill, outline=(30, 30, 30), width=2
                )

                # Draw cell content (grid symbol)
                if cell in ["0", "1", "X"]:
                    bbox = draw.textbbox((0, 0), cell, font=font)
                    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    cx = x0 + cell_px // 2
                    cy = y0 + cell_px // 4
                    if cell == "1":
                        draw.text(
                            (cx - tw // 2, cy - th // 2),
                            cell,
                            fill=text_color,
                            font=font,
                        )
                    else:
                        draw.text(
                            (cx - tw // 2, cy - th // 2),
                            cell,
                            fill=text_color,
                            font=font,
                        )

                # Draw distance below (small text)
                dist_str = str(dist)
                bbox = draw.textbbox((0, 0), dist_str, font=font_small)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                cx = x0 + cell_px // 2
                cy = y0 + cell_px * 3 // 4
                draw.text(
                    (cx - tw // 2, cy - th // 2),
                    dist_str,
                    fill=(100, 100, 100),
                    font=font_small,
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

        return img

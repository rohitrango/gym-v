"""Light Up puzzle environment for gym-v (self-contained)."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation


class RLVELightUpPuzzleEnv(Env):
    """RLVE Light Up puzzle as a single-turn environment."""

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""You are given a {N} × {M} grid. Each cell contains either a number from `0` to `4`, or a character `B` or `W`.
- All `W` cells are considered **white cells** (including those that may be replaced with `L` later).
- All other cells (`0`–`4` or `B`) are considered **black cells**.

You may replace some `W` cells with `L`, indicating the placement of a **light bulb**. A light bulb illuminates its own cell and extends light in all **four directions** (up, down, left, right), stopping when it hits a black cell or the edge of the grid. Please place light bulbs such that:
1. **Each white cell** is illuminated by **at least one** light bulb.
2. No light bulb is illuminated by another light bulb, i.e., no two light bulbs can be placed in the same row or column without a black cell in between.
3. **Each black cell** with a number from `0` to `4` must have **exactly that many** light bulbs in its 4 neighboring cells (up, down, left, right).

The grid is given in **row-major order**:
{grid}

**Output Format:** Output {N} lines, each containing {M} characters with no separators. Some `W` cells should be replaced with `L` to indicate light bulbs; all other cells remain unchanged."""

    def __init__(
        self,
        max_n_m: int = 4,
        density: float | None = None,
        density_list: list[float] | None = None,
        black_cell_density_range: tuple[float, float] = (0.6, 0.95),
        cell_px: int = 48,
        padding: int = 24,
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._max_n_m = max_n_m
        self._density = density
        if density_list is None:
            density_list = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        self._density_list = density_list
        self._black_cell_density_range = black_cell_density_range
        self._cell_px = cell_px
        self._padding = padding
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._grid: list[str] | None = None
        self._prompt: str | None = None
        self._reference_answer: str | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        if self._grid:
            rows = len(self._grid)
            cols = len(self._grid[0]) if self._grid[0] else 0
            size_hint = f"{rows} x {cols}"
        else:
            size_hint = "N x M"
        return dedent(
            f"""
            Light Up puzzle rules:
            1) Place bulbs (L) on white cells (W).
            2) A bulb lights its own cell and extends in four directions until a black cell or edge.
            3) All white cells must be lit.
            4) No two bulbs can see each other in the same row/column without a black cell between.
            5) Numbered black cells (0-4) must have exactly that many adjacent bulbs.

            In the image:
            - Light squares are white cells
            - Dark squares are black cells
            - Digits on black cells are the required adjacent bulb counts
            - The grid is {size_hint}

            Output format: N lines of M characters, replacing some W with L; black cells unchanged.
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
        grid = [["W"] * M for _ in range(N)]

        min_density, max_density = self._black_cell_density_range
        if not 0 < min_density < max_density < 1:
            raise ValueError(
                "black_cell_density_range must be in (0, 1) and increasing"
            )
        black_cell_density = float(self.np_random.uniform(min_density, max_density))
        black_cells = self.np_random.choice(
            range(N * M),
            size=max(1, min(int(N * M * black_cell_density), N * M - 1)),
            replace=False,
        )
        for cell in black_cells:
            row, column = divmod(cell, M)
            grid[row][column] = "B"

        white_cells = [(i, j) for i in range(N) for j in range(M) if grid[i][j] == "W"]
        self.np_random.shuffle(white_cells)
        illuminated = [[False] * M for _ in range(N)]
        for i, j in white_cells:
            if illuminated[i][j]:
                continue
            grid[i][j] = "L"
            illuminated[i][j] = True

            for di, dj in ((-1, 0), (+1, 0), (0, -1), (0, +1)):
                ni, nj = i + di, j + dj
                while 0 <= ni < N and 0 <= nj < M:
                    if grid[ni][nj] == "B":
                        break
                    illuminated[ni][nj] = True
                    ni += di
                    nj += dj

        if self._density is None:
            if not self._density_list:
                raise ValueError("density_list must not be empty when density is None")
            density = float(self.np_random.choice(self._density_list))
        else:
            density = self._density
        if not 0 < density < 1:
            raise ValueError("density must be between 0 and 1")
        black_cells = [(i, j) for i in range(N) for j in range(M) if grid[i][j] == "B"]
        black_cells = list(
            self.np_random.choice(
                black_cells,
                size=max(1, int(len(black_cells) * density)),
                replace=False,
            )
        )
        if not black_cells:
            raise RuntimeError("No numbered black cell")
        for i, j in black_cells:
            counting = 0
            for di, dj in ((-1, 0), (+1, 0), (0, -1), (0, +1)):
                ni, nj = i + di, j + dj
                if 0 <= ni < N and 0 <= nj < M and grid[ni][nj] == "L":
                    counting += 1
            grid[i][j] = str(counting)

        self._reference_answer = "\n".join("".join(row) for row in grid)
        self._grid = [
            "".join(cell if cell != "L" else "W" for cell in row) for row in grid
        ]

    def _prompt_generate(self) -> str:
        N = len(self._grid)
        M = len(self._grid[0])
        return self.prompt_template.format(
            N=N,
            M=M,
            grid="\n".join("".join(row) for row in self._grid),
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

        N = len(self._grid)
        M = len(self._grid[0])
        solution = processed_result

        if len(solution) != N or any(len(row) != M for row in solution):
            return -1.0

        for solution_row, original_row in zip(solution, self._grid, strict=False):
            for solution_cell, original_cell in zip(
                solution_row, original_row, strict=False
            ):
                if original_cell == "W":
                    if solution_cell not in "WL":
                        return -0.5
                elif original_cell in "B01234":
                    if solution_cell != original_cell:
                        return -0.5

        illuminated = [[False] * M for _ in range(N)]
        for i in range(N):
            for j in range(M):
                if solution[i][j] == "L":
                    illuminated[i][j] = True
                    for di, dj in ((-1, 0), (+1, 0), (0, -1), (0, +1)):
                        ni, nj = i + di, j + dj
                        while 0 <= ni < N and 0 <= nj < M:
                            if solution[ni][nj] != "W":
                                if solution[ni][nj] == "L":
                                    return -0.5
                                if solution[ni][nj] in "B01234":
                                    break
                            illuminated[ni][nj] = True
                            ni += di
                            nj += dj
        if any(
            not illuminated[i][j]
            for i in range(N)
            for j in range(M)
            if self._grid[i][j] == "W"
        ):
            return -0.5

        satisfied, total = 0, 0
        for i in range(N):
            for j in range(M):
                if self._grid[i][j] in "01234":
                    total += 1
                    counting = 0
                    for di, dj in ((-1, 0), (+1, 0), (0, -1), (0, +1)):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < N and 0 <= nj < M and solution[ni][nj] == "L":
                            counting += 1
                    if counting == int(self._grid[i][j]):
                        satisfied += 1

        return (satisfied / total) ** 10

    def render(self) -> Image.Image | list[Image.Image] | None:
        if self._grid is None:
            raise RuntimeError("No grid generated")
        rows, cols = len(self._grid), len(self._grid[0])
        cell_px = self._cell_px
        padding = self._padding

        width = padding * 2 + cols * cell_px
        height = padding * 2 + rows * cell_px
        img = Image.new("RGB", (width, height), (248, 248, 248))
        draw = ImageDraw.Draw(img)

        font_path = None
        font_path_candidate = self.assets_dir / "DejaVuSans.ttf"
        if font_path_candidate.exists():
            font_path = str(font_path_candidate)

        if font_path:
            font = ImageFont.truetype(font_path, int(cell_px * 0.45))
        else:
            font = ImageFont.load_default()

        for r in range(rows):
            for c in range(cols):
                cell = self._grid[r][c]
                x0 = padding + c * cell_px
                y0 = padding + r * cell_px
                x1 = x0 + cell_px
                y1 = y0 + cell_px

                if cell == "W":
                    fill = (245, 245, 245)
                elif cell == "B":
                    fill = (40, 40, 40)
                else:
                    fill = (60, 60, 60)
                draw.rectangle(
                    (x0, y0, x1, y1), fill=fill, outline=(20, 20, 20), width=2
                )

                if cell in "01234":
                    bbox = draw.textbbox((0, 0), cell, font=font)
                    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    cx = x0 + cell_px // 2
                    cy = y0 + cell_px // 2
                    draw.text(
                        (cx - tw // 2, cy - th // 2),
                        cell,
                        fill=(230, 230, 230),
                        font=font,
                    )

        return img

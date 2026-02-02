"""Nine puzzle environment for gym-v (self-contained)."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation
from gym_v.logger import get_logger

logger = get_logger()


class RLVENinePuzzleEnv(Env):
    """RLVE Nine puzzle as a single-turn environment."""

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""You are given a {N} × {M} grid, where each cell contains a digit from `0` to `{NM_minus_1}`.

At any time, you may perform one of the following actions:
- Pick a row i (0 ≤ i < {N}) and shift it left or right by **at most** {row_K} cells.
- Pick a column j (0 ≤ j < {M}) and shift it up or down by **at most** {col_K} cells.

You start with the following grid:
{start_grid}

Your goal is to transform it into the following grid:
{destination_grid}

**Output Format:** Each action should be written on its own line in the following format: `[row_or_column] [index] [shifts]`
Where:
- `row_or_column` is either `row` or `column`
- `index` is the 0-based index of the row or column
- `shifts` is a signed integer: positive for right/down, negative for left/up
- Example: `row 0 2` or `column 1 -3`
Do **NOT** include backticks or quotes in your output. Output one action per line in the order they should be performed."""

    def __init__(
        self,
        max_n_m: int = 3,
        steps: int = 5,
        cell_px: int = 64,
        padding: int = 32,
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._max_n_m = max_n_m
        self._steps = steps
        self._cell_px = cell_px
        self._padding = padding
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._n: int | None = None
        self._m: int | None = None
        self._row_k: int | None = None
        self._col_k: int | None = None
        self._start_grid: list[list[int]] | None = None
        self._destination_grid: list[list[int]] | None = None
        self._prompt: str | None = None
        self._oracle_answer: str | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        if self._start_grid:
            rows = len(self._start_grid)
            cols = len(self._start_grid[0]) if self._start_grid[0] else 0
            size_hint = f"{rows} x {cols}"
        else:
            size_hint = "N x M"
        return dedent(
            f"""
            Nine Puzzle rules:
            1) You have a grid with digits from 0 to N*M-1.
            2) You can shift rows left/right or columns up/down (cyclic shifts).
            3) Each shift has a maximum magnitude (row_K for rows, col_K for columns).
            4) Transform the start grid into the destination grid using these shifts.

            In the image:
            - Top grid: Starting configuration ({size_hint})
            - Bottom grid: Destination configuration to achieve
            - Each cell shows a digit from the permutation

            Output Format: Each action should be written on its own line in the \
following format: [row_or_column] [index] [shifts]
            Where:
            - row_or_column is either "row" or "column"
            - index is the 0-based index of the row or column
            - shifts is a signed integer: positive for right/down, negative for left/up
            - Example: row 0 2 or column 1 -3
            Do NOT include backticks or quotes in your output. Output one action \
per line in the order they should be performed.
            """
        ).strip()

    def _get_state_text(self) -> str:
        """Return the text representation of the current state."""
        return self._prompt or ""

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
            metadata={"state_text": state_text, "text_prompt": self._prompt},
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
            metadata={"state_text": state_text, "text_prompt": self._prompt},
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
        """Generate puzzle instance (ported from RLVE)."""
        max_n_m = self._max_n_m
        if max_n_m < 2:
            raise ValueError("max_n_m must be >= 2")

        N = int(self.np_random.integers(2, max_n_m + 1))
        M = int(self.np_random.integers(2, max_n_m + 1))
        self._n = N
        self._m = M

        row_K = int(self.np_random.integers(1, M))
        col_K = int(self.np_random.integers(1, N))
        self._row_k = row_K
        self._col_k = col_K

        # Generate random permutation for start grid
        start_permutation = list(range(N * M))
        self.np_random.shuffle(start_permutation)
        start_grid = [
            [start_permutation[i * M + j] for j in range(M)] for i in range(N)
        ]
        self._start_grid = start_grid

        # Apply random shifts to create destination grid
        steps = self._steps
        if steps < 1:
            raise ValueError("steps must be >= 1")

        destination_grid = [row.copy() for row in start_grid]
        reference_answer = ""

        for step in range(steps):
            row_or_column = self.np_random.choice(["row", "column"])
            if row_or_column == "row":
                index = int(self.np_random.integers(0, N))
            else:
                index = int(self.np_random.integers(0, M))

            # Generate non-zero shift
            while True:
                if row_or_column == "row":
                    shifts = int(self.np_random.integers(-row_K, row_K + 1))
                else:
                    shifts = int(self.np_random.integers(-col_K, col_K + 1))
                if shifts != 0:
                    break

            reference_answer += f"{row_or_column} {index} {shifts}\n"

            # Apply shift
            new_grid = [row.copy() for row in destination_grid]
            if row_or_column == "row":
                assert abs(shifts) <= M - 1
                assert abs(shifts) <= row_K
                for j in range(M):
                    new_grid[index][j] = destination_grid[index][
                        ((j - shifts) % M + M) % M
                    ]
            else:
                assert row_or_column == "column"
                assert abs(shifts) <= N - 1
                assert abs(shifts) <= col_K
                for i in range(N):
                    new_grid[i][index] = destination_grid[((i - shifts) % N + N) % N][
                        index
                    ]
            destination_grid = new_grid

        self._destination_grid = destination_grid
        self._oracle_answer = reference_answer.strip()

    def _prompt_generate(self) -> str:
        """Generate prompt text."""
        if self._start_grid is None or self._destination_grid is None:
            raise RuntimeError("No grid generated")
        N = self._n
        M = self._m
        return self.prompt_template.format(
            N=N,
            M=M,
            NM_minus_1=N * M - 1,
            row_K=self._row_k,
            col_K=self._col_k,
            start_grid="\n".join(" ".join(map(str, row)) for row in self._start_grid),
            destination_grid="\n".join(
                " ".join(map(str, row)) for row in self._destination_grid
            ),
        )

    def _process(self, answer: str | None) -> list[list[str | int]] | None:
        """Process answer string into list of actions."""
        if answer is None:
            return None
        answer = answer.strip()
        if not answer:
            return None
        actions = []
        for line in answer.splitlines():
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) != 3:
                    return None
                if parts[0] not in ("row", "column"):
                    return None
                try:
                    action = [parts[0], int(parts[1]), int(parts[2])]
                    actions.append(action)
                except ValueError:
                    return None
        return actions

    def _score_answer(self, answer: str) -> float:
        """Score the answer (ported from RLVE)."""
        processed_result = self._process(answer)
        if processed_result is None:
            return 0.0
        destination_grid = [row.copy() for row in self._start_grid]

        for action in processed_result:
            new_grid = [row.copy() for row in destination_grid]
            if action[0] == "row":
                index = action[1]
                if not (0 <= index < self._n):
                    return 0.0
                shifts = action[2]
                if not (-self._row_k <= shifts <= self._row_k):
                    return 0.0
                for j in range(self._m):
                    new_grid[index][j] = destination_grid[index][
                        ((j - shifts) % self._m + self._m) % self._m
                    ]
            else:
                assert action[0] == "column"
                index = action[1]
                if not (0 <= index < self._m):
                    return 0.0
                shifts = action[2]
                if not (-self._col_k <= shifts <= self._col_k):
                    return 0.0
                for i in range(self._n):
                    new_grid[i][index] = destination_grid[
                        ((i - shifts) % self._n + self._n) % self._n
                    ][index]
            destination_grid = new_grid

        # Calculate reward using mean([gold=answer])^beta strategy
        matching_cells = sum(
            sum(int(a == b) for a, b in zip(gold_row, answer_row, strict=False))
            for gold_row, answer_row in zip(
                self._destination_grid, destination_grid, strict=False
            )
        )
        total_cells = self._n * self._m
        return (matching_cells / total_cells) ** 10

    def render(self) -> Image.Image | list[Image.Image] | None:
        """Render the puzzle showing both start and destination grids."""
        if self._start_grid is None or self._destination_grid is None:
            raise RuntimeError("No grid generated")

        rows, cols = self._n, self._m
        cell_px = self._cell_px
        padding = self._padding
        grid_spacing = padding  # Space between two grids

        # Calculate dimensions for two grids stacked vertically
        grid_width = cols * cell_px
        grid_height = rows * cell_px
        width = padding * 2 + grid_width
        height = padding * 3 + grid_height * 2 + grid_spacing

        img = Image.new("RGB", (width, height), (250, 250, 250))
        draw = ImageDraw.Draw(img)

        # Load font
        font_path = None
        font_path_candidate = self.assets_dir / "DejaVuSans.ttf"
        if font_path_candidate.exists():
            font_path = str(font_path_candidate)

        if font_path:
            number_font = ImageFont.truetype(font_path, int(cell_px * 0.4))
            label_font = ImageFont.truetype(font_path, int(cell_px * 0.3))
        else:
            number_font = ImageFont.load_default()
            label_font = ImageFont.load_default()

        # Helper function to draw a grid
        def draw_grid(grid: list[list[int]], offset_y: int, label: str) -> None:
            # Draw label
            label_y = offset_y - int(cell_px * 0.4)
            draw.text((padding, label_y), label, fill=(30, 30, 30), font=label_font)

            # Draw grid lines
            for r in range(rows + 1):
                y = offset_y + r * cell_px
                draw.line(
                    (padding, y, padding + grid_width, y),
                    fill=(60, 60, 60),
                    width=2,
                )
            for c in range(cols + 1):
                x = padding + c * cell_px
                draw.line(
                    (x, offset_y, x, offset_y + grid_height),
                    fill=(60, 60, 60),
                    width=2,
                )

            # Draw numbers
            for r in range(rows):
                for c in range(cols):
                    v = str(grid[r][c])
                    bbox = draw.textbbox((0, 0), v, font=number_font)
                    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    cx = padding + c * cell_px + cell_px // 2
                    cy = offset_y + r * cell_px + cell_px // 2
                    draw.text(
                        (cx - tw // 2, cy - th // 2),
                        v,
                        fill=(10, 10, 10),
                        font=number_font,
                    )

        # Draw start grid
        draw_grid(self._start_grid, padding, "Start Grid:")

        # Draw destination grid
        destination_offset = padding * 2 + grid_height + grid_spacing
        draw_grid(self._destination_grid, destination_offset, "Destination Grid:")

        return img

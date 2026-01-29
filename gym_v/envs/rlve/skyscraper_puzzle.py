"""Skyscraper puzzle environment for gym-v (self-contained)."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation


class RLVESkyscraperPuzzleEnv(Env):
    """RLVE Skyscraper puzzle as a single-turn environment."""

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""You are given a {N} × {N} grid. Your task is to place a building of height in the range [0, {N_minus_1}] in each cell such that:
- Each **row** and each **column** contains all integer heights from `0` to `{N_minus_1}` **exactly once**.
- A building is **visible from a direction** if there are no taller buildings before it in that direction.

The number of visible buildings is specified as follows:
- From the **left** of each row: {left}
- From the **right** of each row: {right}
- From the **top** of each column: {top}
- From the **bottom** of each column: {bottom}

**Output Format:** Your final answer should contain {N} lines, each with {N} integers (heights), separated by spaces. Each line represents a row of the grid."""

    def __init__(
        self,
        n: int = 3,
        cell_px: int = 52,
        padding: int = 28,
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._n = n
        self._cell_px = cell_px
        self._padding = padding
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._prompt: str | None = None
        self._oracle_answer: str | None = None
        self._last_image: Image.Image | None = None
        self._left: list[int] | None = None
        self._right: list[int] | None = None
        self._top: list[int] | None = None
        self._bottom: list[int] | None = None

    @property
    def description(self) -> str:
        n = int(self._n)
        return dedent(
            f"""
            Skyscraper puzzle rules:
            1) Fill each row/column with heights 0..{n - 1} exactly once.
            2) A building is visible from a direction if no taller buildings are before it.
            3) Edge clues show the number of visible buildings from left/right/top/bottom.

            In the image:
            - The grid is {n} x {n}
            - Edge numbers are visibility clues

            Output format: {n} lines, each with {n} integers separated by spaces.
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
        N = int(self._n)
        if N < 3:
            raise ValueError("n must be >= 3")

        permutation_row, permutation_col = list(range(N)), list(range(N))
        self.np_random.shuffle(permutation_row)
        self.np_random.shuffle(permutation_col)

        grid = [
            [(permutation_row[i] + permutation_col[j]) % N for j in range(N)]
            for i in range(N)
        ]
        self._left = [
            sum(int(grid[i][j] == max(grid[i][: j + 1])) for j in range(N))
            for i in range(N)
        ]
        self._right = [
            sum(int(grid[i][j] == max(grid[i][j:])) for j in range(N)) for i in range(N)
        ]

        transposed_grid = [[grid[j][i] for j in range(N)] for i in range(N)]
        self._top = [
            sum(
                int(transposed_grid[i][j] == max(transposed_grid[i][: j + 1]))
                for j in range(N)
            )
            for i in range(N)
        ]
        self._bottom = [
            sum(
                int(transposed_grid[i][j] == max(transposed_grid[i][j:]))
                for j in range(N)
            )
            for i in range(N)
        ]

        self._oracle_answer = "\n".join(" ".join(map(str, row)) for row in grid)

    def _prompt_generate(self) -> str:
        N = int(self._n)
        return self.prompt_template.format(
            N=N,
            N_minus_1=N - 1,
            left=" ".join(map(str, self._left)),
            right=" ".join(map(str, self._right)),
            top=" ".join(map(str, self._top)),
            bottom=" ".join(map(str, self._bottom)),
        )

    def _process(self, answer: str | None) -> list[list[int]] | None:
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
        processed_result = self._process(answer)
        if processed_result is None:
            return 0.0
        N = int(self._n)
        solution = processed_result

        if len(solution) != N or not all(len(row) == N for row in solution):
            return 0.0
        if not all(set(row) == set(range(N)) for row in solution):
            return 0.0
        if not all(
            set(solution[i][j] for i in range(N)) == set(range(N)) for j in range(N)
        ):
            return 0.0
        left = [
            sum(int(solution[i][j] == max(solution[i][: j + 1])) for j in range(N))
            for i in range(N)
        ]
        right = [
            sum(int(solution[i][j] == max(solution[i][j:])) for j in range(N))
            for i in range(N)
        ]

        transposed_solution = [[solution[j][i] for j in range(N)] for i in range(N)]
        top = [
            sum(
                int(transposed_solution[i][j] == max(transposed_solution[i][: j + 1]))
                for j in range(N)
            )
            for i in range(N)
        ]
        bottom = [
            sum(
                int(transposed_solution[i][j] == max(transposed_solution[i][j:]))
                for j in range(N)
            )
            for i in range(N)
        ]

        satisfied = sum(int(a == b) for a, b in zip(left, self._left, strict=False))
        satisfied += sum(int(a == b) for a, b in zip(right, self._right, strict=False))
        satisfied += sum(int(a == b) for a, b in zip(top, self._top, strict=False))
        satisfied += sum(
            int(a == b) for a, b in zip(bottom, self._bottom, strict=False)
        )

        return (satisfied / (4 * N)) ** 10

    def render(self) -> Image.Image | list[Image.Image] | None:
        N = int(self._n)
        cell_px = self._cell_px
        padding = self._padding
        margin = cell_px

        width = padding * 2 + N * cell_px + margin * 2
        height = padding * 2 + N * cell_px + margin * 2
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

        x0 = padding + margin
        y0 = padding + margin
        for r in range(N + 1):
            y = y0 + r * cell_px
            draw.line((x0, y, x0 + N * cell_px, y), fill=(30, 30, 30), width=2)
        for c in range(N + 1):
            x = x0 + c * cell_px
            draw.line((x, y0, x, y0 + N * cell_px), fill=(30, 30, 30), width=2)

        def draw_centered(text, cx, cy):
            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text((cx - tw // 2, cy - th // 2), text, fill=(10, 10, 10), font=font)

        for i, v in enumerate(self._top):
            cx = x0 + i * cell_px + cell_px // 2
            cy = padding + margin // 2
            draw_centered(str(v), cx, cy)
        for i, v in enumerate(self._bottom):
            cx = x0 + i * cell_px + cell_px // 2
            cy = y0 + N * cell_px + margin // 2
            draw_centered(str(v), cx, cy)
        for i, v in enumerate(self._left):
            cx = padding + margin // 2
            cy = y0 + i * cell_px + cell_px // 2
            draw_centered(str(v), cx, cy)
        for i, v in enumerate(self._right):
            cx = x0 + N * cell_px + margin // 2
            cy = y0 + i * cell_px + cell_px // 2
            draw_centered(str(v), cx, cy)

        return img

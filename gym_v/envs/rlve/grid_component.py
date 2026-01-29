"""Grid component environment for gym-v (self-contained)."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation
from gym_v.logger import get_logger

logger = get_logger()


class RLVEGridComponentEnv(Env):
    """RLVE Grid Component as a single-turn environment."""

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""You are given a {N} × {M} grid. Each cell contains either `0` or `1`. Please compute the **largest connected component** of `1`s in the grid, where a connected component is defined as a group of `1` cells that are reachable from each other by moving **up**, **down**, **left**, or **right** to an adjacent `1` cell.

The grid is given as follows:
{grid}

**Output Format:** Output a single integer — the size of the largest connected component (i.e., the number of `1`s in it). If there are no `1`s in the grid, output `0`."""

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

        self._grid: list[str] | None = None
        self._n: int | None = None
        self._m: int | None = None
        self._prompt: str | None = None
        self._oracle_answer: int | None = None
        self._component_labels: list[list[int]] | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        if self._grid:
            size_hint = f"{self._n} x {self._m}"
        else:
            size_hint = "N x M"
        return dedent(
            f"""
            Grid Component Identification rules:
            1) Find the largest connected component of 1s in a binary grid.
            2) Two cells are connected if they are adjacent (up/down/left/right) and both contain 1.
            3) A connected component is a maximal set of 1s where any two can reach each other through adjacent 1s.
            4) Return the size (number of cells) of the largest connected component.

            In the image:
            - Each cell contains 0 or 1
            - Different colored regions represent different connected components
            - The grid is {size_hint}

            Output format: A single integer representing the size of the largest component.
            If there are no 1s, output 0.
            """
        ).strip()

    def _get_state_text(self) -> str:
        """Return text representation of the grid."""
        if self._grid is None:
            return ""
        return "\n".join(self._grid)

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
            "oracle_answer": str(self._oracle_answer),
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
            "oracle_answer": str(self._oracle_answer),
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
        """Generate a random grid with connected components."""
        max_n_m = self._max_n_m
        if max_n_m < 2:
            raise ValueError("max_n_m must be >= 2")

        N = int(self.np_random.integers(2, max_n_m + 1))
        M = int(self.np_random.integers(2, max_n_m + 1))
        self._n = N
        self._m = M

        one_probability = float(self.np_random.uniform(0.1, 0.9))
        grid = [
            "".join("01"[self.np_random.random() < one_probability] for _ in range(M))
            for _ in range(N)
        ]
        self._grid = grid

        # Label connected components
        labels = [[0] * M for _ in range(N)]

        def DFS(x: int, y: int, label: int) -> None:
            """Depth-first search to label connected components."""
            stack = [(x, y)]
            while stack:
                x, y = stack.pop()
                for dx, dy in [(-1, 0), (+1, 0), (0, -1), (0, +1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < N and 0 <= ny < M and grid[nx][ny] == "1":
                        if labels[nx][ny] == 0:
                            labels[nx][ny] = label
                            stack.append((nx, ny))

        total = 0
        counting = [0]
        for x in range(N):
            for y in range(M):
                if grid[x][y] == "1":
                    if labels[x][y] == 0:
                        total += 1
                        counting.append(0)
                        labels[x][y] = total
                        DFS(x, y, total)
                    counting[labels[x][y]] += 1

        self._component_labels = labels
        self._oracle_answer = max(counting)

    def _prompt_generate(self) -> str:
        if self._grid is None:
            raise RuntimeError("No grid generated")
        return self.prompt_template.format(
            N=self._n,
            M=self._m,
            grid="\n".join(self._grid),
        )

    def _process(self, answer: str | None) -> int | None:
        """Process the answer string into an integer."""
        if answer is not None:
            answer = answer.strip()
            try:
                int_answer = int(answer)
                return int_answer
            except ValueError:
                return None
        else:
            return None

    def _score_answer(self, answer: str) -> float:
        """Score the answer."""
        processed_result = self._process(answer)
        if processed_result is not None:
            if processed_result == self._oracle_answer:
                return 1.0
            else:
                return 0.0
        else:
            return 0.0

    def render(self) -> Image.Image | list[Image.Image] | None:
        """Render the grid with colored components."""
        if self._grid is None or self._component_labels is None:
            raise RuntimeError("No grid generated")
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

        # Define distinct colors for components
        component_colors = [
            (255, 220, 220),  # Light red
            (220, 240, 255),  # Light blue
            (220, 255, 220),  # Light green
            (255, 240, 200),  # Light orange
            (240, 220, 255),  # Light purple
            (255, 255, 200),  # Light yellow
            (200, 255, 255),  # Light cyan
            (255, 220, 240),  # Light pink
            (230, 255, 220),  # Light lime
            (255, 230, 200),  # Light peach
        ]

        # Fill cells with component colors
        for r in range(rows):
            for c in range(cols):
                x = padding + c * cell_px
                y = padding + r * cell_px

                if self._grid[r][c] == "1":
                    label = self._component_labels[r][c]
                    color = component_colors[(label - 1) % len(component_colors)]
                    draw.rectangle(
                        [x + 2, y + 2, x + cell_px - 2, y + cell_px - 2],
                        fill=color,
                        outline=None,
                    )
                else:
                    # 0 cells are white
                    draw.rectangle(
                        [x + 2, y + 2, x + cell_px - 2, y + cell_px - 2],
                        fill=(255, 255, 255),
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
                v = self._grid[r][c]
                bbox = draw.textbbox((0, 0), v, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                cx = padding + c * cell_px + cell_px // 2
                cy = padding + r * cell_px + cell_px // 2
                draw.text((cx - tw // 2, cy - th // 2), v, fill=(10, 10, 10), font=font)

        return img

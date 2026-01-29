"""Grid parity construction environment for gym-v (self-contained)."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation
from gym_v.logger import get_logger

logger = get_logger()


class RLVEGridParityConstructionEnv(Env):
    """RLVE Grid Parity Construction as a single-turn environment.

    The goal is to construct a binary matrix (0s and 1s) such that when
    computing the parity at each cell (XOR of cell value with all 4-neighbors),
    it matches a given target parity matrix.
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""Please construct a {N} × {M} binary matrix (i.e., a matrix where each cell is either 0 or 1) such that its **parity matrix** is:
{parity}

**Definition (Parity Matrix):** For each cell (i, j), its parity is the XOR of the cell's value and the values of its four neighbors (up, down, left, right). A neighbor outside the grid is treated as 0.

**Output Format:** Output {N} lines, each with {M} characters (each '0' or '1'), without separators. The format must match the input: one line per row."""

    def __init__(
        self,
        max_n_m: int = 4,
        cell_px: int = 56,
        padding: int = 24,
        num_players: int = 1,
        wrong_format: float = -1.0,
        rewarding_strategy: str = "(satisfied/all)^beta",
        rewarding_weight: float = 1.0,
        rewarding_beta: float = 5.0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._max_n_m = max_n_m
        self._cell_px = cell_px
        self._padding = padding
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._rewards = {
            "wrong_format": wrong_format,
            "rewarding_strategy": rewarding_strategy,
            "rewarding_weight": rewarding_weight,
            "rewarding_beta": rewarding_beta,
        }

        self._n: int | None = None
        self._m: int | None = None
        self._parity: list[list[int]] | None = None
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
            Grid Parity Construction puzzle:
            Construct a binary matrix (0s and 1s) such that its parity matrix matches
            the given target parity matrix.

            Parity Definition: For each cell (i,j), the parity is the XOR of:
            - The cell's own value
            - Its up neighbor (or 0 if out of bounds)
            - Its down neighbor (or 0 if out of bounds)
            - Its left neighbor (or 0 if out of bounds)
            - Its right neighbor (or 0 if out of bounds)

            In the image:
            - Light blue cells represent parity value 0 (even)
            - Light coral cells represent parity value 1 (odd)
            - Grid lines clearly separate cells
            - The grid is {size_hint}

            Output format: N lines with M characters ('0' or '1'), no separators.
            Each line represents one row of the solution matrix.
            """
        ).strip()

    def _get_state_text(self) -> str:
        """Return text representation of the parity matrix."""
        if self._parity is None:
            return ""
        return "\n".join("".join(str(cell) for cell in row) for row in self._parity)

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
        """Generate a random binary grid and compute its parity matrix."""
        max_n_m = self._max_n_m
        if max_n_m < 2:
            raise ValueError("max_n_m must be >= 2")

        N = int(self.np_random.integers(2, max_n_m + 1))
        M = int(self.np_random.integers(2, max_n_m + 1))

        self._n = N
        self._m = M

        # Generate random binary grid with random density
        one_probability = float(self.np_random.random())
        grid = [
            [1 if self.np_random.random() < one_probability else 0 for _ in range(M)]
            for _ in range(N)
        ]

        self._oracle_answer = "\n".join(
            "".join(str(cell) for cell in row) for row in grid
        )

        # Compute parity matrix
        parity = [[0] * M for _ in range(N)]
        for i in range(N):
            for j in range(M):
                parity[i][j] ^= grid[i][j]
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < N and 0 <= nj < M:
                        parity[i][j] ^= grid[ni][nj]

        self._parity = parity

    def _prompt_generate(self) -> str:
        """Generate the prompt with the parity matrix."""
        if self._parity is None or self._n is None or self._m is None:
            raise RuntimeError("No parity matrix generated")
        return self.prompt_template.format(
            N=self._n,
            M=self._m,
            parity="\n".join(
                "".join(str(cell) for cell in row) for row in self._parity
            ),
        )

    def _process(self, answer: str | None) -> list[str] | None:
        """Process the answer string into a list of row strings."""
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
        """Score the answer based on how well it matches the target parity."""
        processed_result = self._process(answer)
        if processed_result is None:
            return 0.0

        N = self._n
        M = self._m
        grid = processed_result

        # Check format
        if len(grid) != N or any(len(row) != M for row in grid):
            return 0.0
        for row in grid:
            if not all(c in "01" for c in row):
                return 0.0

        # Compute parity of the submitted grid
        parity = [[0] * M for _ in range(N)]
        for i in range(N):
            for j in range(M):
                parity[i][j] ^= int(grid[i][j])
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < N and 0 <= nj < M:
                        parity[i][j] ^= int(grid[ni][nj])

        # Count satisfied cells
        satisfied = sum(
            int(parity[i][j] == self._parity[i][j]) for i in range(N) for j in range(M)
        )

        # Compute reward based on strategy
        if self._rewards["rewarding_strategy"] == "(satisfied/all)^beta":
            return self._rewards["rewarding_weight"] * (
                (satisfied / (N * M)) ** self._rewards["rewarding_beta"]
            )
        elif self._rewards["rewarding_strategy"] == "satisfied=all":
            return self._rewards["rewarding_weight"] * (satisfied == (N * M))
        else:
            raise NotImplementedError(
                f"Unknown rewarding strategy: {self._rewards['rewarding_strategy']}"
            )

    def render(self) -> Image.Image | list[Image.Image] | None:
        """Render the parity matrix with beautiful colors for even/odd cells."""
        if self._parity is None or self._n is None or self._m is None:
            raise RuntimeError("No parity matrix generated")

        rows, cols = self._n, self._m
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
            font = ImageFont.truetype(font_path, int(cell_px * 0.5))
        else:
            font = ImageFont.load_default()

        # Define colors for parity values
        # 0 (even) = light blue, 1 (odd) = light coral
        color_map = {
            0: (173, 216, 230),  # Light blue for even parity
            1: (240, 128, 128),  # Light coral for odd parity
        }

        # Draw cells with background colors
        for r in range(rows):
            for c in range(cols):
                parity_val = self._parity[r][c]
                bg_color = color_map[parity_val]

                x0 = padding + c * cell_px
                y0 = padding + r * cell_px
                x1 = x0 + cell_px
                y1 = y0 + cell_px

                draw.rectangle([x0, y0, x1, y1], fill=bg_color)

        # Draw grid lines
        for r in range(rows + 1):
            y = padding + r * cell_px
            draw.line(
                (padding, y, padding + cols * cell_px, y),
                fill=(40, 40, 40),
                width=2,
            )
        for c in range(cols + 1):
            x = padding + c * cell_px
            draw.line(
                (x, padding, x, padding + rows * cell_px),
                fill=(40, 40, 40),
                width=2,
            )

        # Draw parity values
        for r in range(rows):
            for c in range(cols):
                v = str(self._parity[r][c])
                bbox = draw.textbbox((0, 0), v, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                cx = padding + c * cell_px + cell_px // 2
                cy = padding + r * cell_px + cell_px // 2
                draw.text((cx - tw // 2, cy - th // 2), v, fill=(20, 20, 20), font=font)

        return img

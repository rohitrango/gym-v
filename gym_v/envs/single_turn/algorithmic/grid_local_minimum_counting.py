"""Grid local minimum counting environment for gym-v (self-contained)."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation
from gym_v.logger import get_logger

logger = get_logger()


class GridLocalMinimumCountingEnv(Env):
    # Meta: source=RLVE, category=algorithmic, turn=single
    """RLVE grid local minimum counting as a single-turn environment."""

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""Consider a grid of size {N} × {M}, where the numbers from 1 to {N} × {M} are placed in the cells such that **each number appears exactly once**.
A cell is considered a local minimum if its value is strictly less than all of its 8 neighbors (adjacent vertically, horizontally, or diagonally); if a neighbor does not exist, it is considered to be infinitely large. You are given a grid of size {N} × {M} where some cells are marked with `X` and others with `.`. Please count how many valid numberings exist such that the local minima are **exactly** those marked with `X`. The grid is given as follows:
{grid}

**Output Format:** Output a single integer — the number of valid labelings."""

    def __init__(
        self,
        max_n_m: int = 4,
        cell_px: int = 64,
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
        self._n: int = 0
        self._m: int = 0
        self._prompt: str | None = None
        self._oracle_answer: int | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        if self._grid:
            rows = len(self._grid)
            cols = len(self._grid[0]) if self._grid else 0
            size_hint = f"{rows} x {cols}"
        else:
            size_hint = "N x M"
        return dedent(
            f"""
            Consider a grid where numbers from 1 to N*M are placed in the cells such that each number appears exactly once. A cell is considered a local minimum if its value is strictly less than all of its 8 neighbors (adjacent vertically, horizontally, or diagonally); if a neighbor does not exist, it is considered to be infinitely large. Count how many valid numberings exist such that the local minima are exactly those marked with 'X'.

            Grid Local Minimum Counting rules:
            1) A grid of {size_hint} contains numbers 1 to N*M, each appearing exactly once.
            2) A cell is a local minimum if its value is strictly less than all 8 neighbors (diagonal and orthogonal).
            3) Cells without neighbors in a direction treat that direction as infinitely large.
            4) The grid shows 'X' for required local minima and '.' for other positions.
            5) Count how many valid numberings exist where local minima are EXACTLY the 'X' positions.

            In the image:
            - 'X' marks cells that MUST be local minima (required local minimum positions)
            - '.' marks cells that must NOT be local minima (non-local-minimum positions)
            - Color coding:
              - Blue/purple (cool colors): cells marked with 'X' (local minima)
              - Yellow/beige (warm colors): cells marked with '.' (non-local-minima)

            **Output Format:** Output a single integer — the number of valid labelings.
            """
        ).strip()

    def _get_state_text(self) -> str:
        """Return the text representation of the grid."""
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
            text=None,
            metadata={"state_text": state_text, "text_prompt": self._prompt},
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
            text=None,
            metadata={"state_text": state_text, "text_prompt": self._prompt},
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
        """Generate a random grid with local minima marked."""
        max_n_m = self._max_n_m
        if max_n_m < 2:
            raise ValueError("max_n_m must be >= 2")

        N = int(self.np_random.integers(2, max_n_m + 1))
        M = int(self.np_random.integers(2, max_n_m + 1))
        self._n = N
        self._m = M

        # Generate random permutation of numbers 1 to N*M
        permutation = list(range(1, N * M + 1))
        self.np_random.shuffle(permutation)

        def get_num(i: int, j: int) -> int:
            return permutation[i * M + j]

        # Mark local minima
        grid = [["."] * M for _ in range(N)]
        for i in range(N):
            for j in range(M):
                local_minimum = True
                for dx, dy in [
                    (-1, -1),
                    (-1, 0),
                    (-1, +1),
                    (0, -1),
                    (0, +1),
                    (+1, -1),
                    (+1, 0),
                    (+1, +1),
                ]:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < N and 0 <= nj < M and get_num(ni, nj) <= get_num(i, j):
                        local_minimum = False
                        break
                if local_minimum:
                    grid[i][j] = "X"

        self._grid = grid
        self._oracle_answer = self._compute_answer(grid, N, M)

    def _compute_answer(self, raw: list[list[str]], N: int, M: int) -> int:
        """Compute the number of valid labelings using inclusion-exclusion DP."""
        # Build boolean map of required local minima
        grid = [[(raw[i][j] == "X") for j in range(M)] for i in range(N)]

        # Quick invalid check: no two required 'X's may be adjacent (including diagonals)
        for i in range(N):
            for j in range(M):
                if grid[i][j]:
                    for di in (-1, 0, 1):
                        for dj in (-1, 0, 1):
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if 0 <= ni < N and 0 <= nj < M and grid[ni][nj]:
                                # Invalid grid
                                return 0

        ans = 0

        def inrange(x: int, y: int) -> bool:
            return 0 <= x < N and 0 <= y < M

        def calc() -> int:
            """Use inclusion-exclusion DP to count valid labelings."""
            pos = [(i, j) for i in range(N) for j in range(M) if grid[i][j]]
            cntX = len(pos)
            total = N * M

            # dp[used_cells][subset_mask]
            dp = [[0] * (1 << cntX) for _ in range(total + 2)]
            dp[0][0] = 1

            for s in range(1 << cntX):
                # mark all cells "blocked" by the minima NOT in subset s
                blocked = [[False] * M for _ in range(N)]
                free_cells = total
                for k in range(cntX):
                    if not (s & (1 << k)):
                        x, y = pos[k]
                        for di in (-1, 0, 1):
                            for dj in (-1, 0, 1):
                                ni, nj = x + di, y + dj
                                if inrange(ni, nj) and not blocked[ni][nj]:
                                    blocked[ni][nj] = True
                                    free_cells -= 1

                for used in range(free_cells + 1):
                    v = dp[used][s]
                    if not v:
                        continue
                    # place a non-min in one of the remaining free cells
                    dp[used + 1][s] += v * (free_cells - used)
                    # or turn one of the excluded minima into an actual minima
                    for k in range(cntX):
                        if not (s & (1 << k)):
                            dp[used + 1][s | (1 << k)] += v

            # We want all total cells assigned, and all minima chosen
            return dp[total][(1 << cntX) - 1]

        def dfs(i: int, j: int, sign: int) -> None:
            """Inclusion-exclusion DFS over possible additional local minima."""
            nonlocal ans
            if i == N:
                ans += sign * calc()
                return

            # move to next cell
            ni, nj = (i, j + 1) if j + 1 < M else (i + 1, 0)

            # option 1: don't add a minima here
            dfs(ni, nj, sign)

            # option 2: if this cell is not already a minima, and none of its neighbors is one, we can add it
            if not grid[i][j]:
                ok = True
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        if di == 0 and dj == 0:
                            continue
                        ai, aj = i + di, j + dj
                        if inrange(ai, aj) and grid[ai][aj]:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    grid[i][j] = True
                    dfs(ni, nj, -sign)
                    grid[i][j] = False

        dfs(0, 0, 1)
        if ans <= 0:
            ans = 1  # ensure positive answer
        return ans

    def _prompt_generate(self) -> str:
        if self._grid is None:
            raise RuntimeError("No grid generated")
        return self.prompt_template.format(
            N=self._n,
            M=self._m,
            grid="\n".join("".join(row) for row in self._grid),
        )

    def _process(self, answer: str | None) -> int | None:
        """Process the answer string into an integer."""
        if answer is None:
            return None
        answer = answer.strip()
        try:
            int_answer = int(answer)
            return int_answer
        except ValueError:
            return None

    def _score_answer(self, answer: str) -> float:
        """Score the answer using (min/max)^beta strategy."""
        processed_result = self._process(answer)
        if processed_result is None:
            return 0.0
        if processed_result < 0:
            return 0.0
        a, b = self._oracle_answer, processed_result
        beta = 10.0
        return (min(a, b) / max(a, b)) ** beta

    def render(self) -> Image.Image | list[Image.Image] | None:
        """Render the grid with gradient colors showing local minima."""
        if self._grid is None:
            raise RuntimeError("No grid generated")

        rows, cols = len(self._grid), len(self._grid[0])
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
            font = ImageFont.truetype(font_path, int(cell_px * 0.35))
        else:
            font = ImageFont.load_default()

        # Draw grid cells with gradient colors
        for r in range(rows):
            for c in range(cols):
                x0 = padding + c * cell_px
                y0 = padding + r * cell_px
                x1 = x0 + cell_px
                y1 = y0 + cell_px

                # Create gradient effect based on position
                # Local minima (X) get cooler colors (blue/purple)
                # Non-minima (.) get warmer colors (yellow/orange)
                if self._grid[r][c] == "X":
                    # Cool colors for local minima
                    base_hue = 0.6  # Blue
                    variation = (r * cols + c) / (rows * cols) * 0.2
                    base_hue + variation - 0.1

                    # Convert HSV to RGB
                    r_val = int(120 + (r / rows) * 50)
                    g_val = int(150 + (c / cols) * 50)
                    b_val = int(220 - (r / rows) * 30)
                    fill_color = (r_val, g_val, b_val)
                else:
                    # Warm/neutral colors for non-minima
                    r_val = int(240 - (r / rows) * 30)
                    g_val = int(235 - (c / cols) * 30)
                    b_val = int(200 - (r + c) / (rows + cols) * 40)
                    fill_color = (r_val, g_val, b_val)

                draw.rectangle([x0, y0, x1, y1], fill=fill_color)

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

        # Draw symbols (X or .)
        for r in range(rows):
            for c in range(cols):
                v = self._grid[r][c]
                bbox = draw.textbbox((0, 0), v, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                cx = padding + c * cell_px + cell_px // 2
                cy = padding + r * cell_px + cell_px // 2

                # Draw text with better visibility
                if self._grid[r][c] == "X":
                    # Bold X for local minima
                    text_color = (255, 255, 255)
                    # Draw shadow for better visibility
                    draw.text(
                        (cx - tw // 2 + 1, cy - th // 2 + 1),
                        v,
                        fill=(0, 0, 0),
                        font=font,
                    )
                    draw.text(
                        (cx - tw // 2, cy - th // 2), v, fill=text_color, font=font
                    )
                else:
                    # Lighter dot for non-minima
                    text_color = (100, 100, 100)
                    draw.text(
                        (cx - tw // 2, cy - th // 2), v, fill=text_color, font=font
                    )

        return img

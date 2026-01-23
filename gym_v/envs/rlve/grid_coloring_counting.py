"""Grid coloring counting environment for gym-v (self-contained)."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation
from gym_v.logger import get_logger

logger = get_logger()


class RLVEGridColoringCountingEnv(Env):
    """RLVE Grid Coloring Counting as a single-turn environment.

    Count the number of valid colorings of an N×M grid where:
    1. Some cells are colored (others left uncolored) using C colors (0 to C-1)
    2. No two different colors appear in the same row or column
    3. Color i is used exactly X[i] times

    The answer is computed modulo MOD.
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""You are given a grid of size {N} × {M}. You may color some cells (and leave others uncolored) using {C} colors labeled from 0 to {C_minus_1}, such that:
1. No two different colors appear in the same row or the same column.
2. Color `i` is used exactly X[i] times. The array X is given as: {Xs}

Please compute the number of valid colorings modulo {MOD}."""

    def __init__(
        self,
        max_n_m: int = 6,
        max_mod: int = 10000,
        cell_px: int = 56,
        padding: int = 24,
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._max_n_m = max_n_m
        self._max_mod = max_mod
        self._cell_px = cell_px
        self._padding = padding
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._n: int | None = None
        self._m: int | None = None
        self._c: int | None = None
        self._xs: list[int] | None = None
        self._mod: int | None = None
        self._prompt: str | None = None
        self._reference_answer: int | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        if self._n and self._m and self._c and self._xs and self._mod:
            size_hint = f"{self._n} x {self._m}"
            colors_hint = f"{self._c} colors"
            usage_hint = ", ".join(f"X[{i}]={x}" for i, x in enumerate(self._xs))
            mod_hint = f"mod {self._mod}"
        else:
            size_hint = "N x M"
            colors_hint = "C colors"
            usage_hint = "X[i] times for color i"
            mod_hint = "mod MOD"

        return dedent(
            f"""
            Grid Coloring Counting Problem:

            Given a {size_hint} grid, count valid colorings using {colors_hint}
            where:
            1) You may color some cells (leave others uncolored)
            2) No two different colors appear in the same row or column
            3) Each color i must be used exactly {usage_hint}

            Compute the count {mod_hint}.

            In the visualization:
            - Grid shows N x M dimensions
            - Color usage requirements are displayed
            - Each color is shown with its required count

            Output format: A single integer (the count modulo MOD).
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
                "rlve_reference_answer": str(self._reference_answer),
            },
        )
        info = {
            "reference_answer": str(self._reference_answer),
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
                "rlve_reference_answer": str(self._reference_answer),
            },
        )
        info = {
            "reference_answer": str(self._reference_answer),
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
        """Generate a grid coloring counting problem instance.

        Uses dynamic programming with inclusion-exclusion to compute the
        number of valid colorings modulo MOD.
        """
        max_n_m = self._max_n_m
        if max_n_m < 2:
            raise ValueError("max_n_m must be >= 2")

        while True:
            N = int(self.np_random.integers(2, max_n_m + 1))
            M = int(self.np_random.integers(2, max_n_m + 1))
            sum_X = int(self.np_random.integers(1, N * M + 1))
            C = int(self.np_random.integers(1, min(N, M, sum_X) + 1))

            # Generate partition of sum_X into C positive parts
            deltas = sorted(self.np_random.choice(range(1, sum_X), C - 1, replace=False).tolist())
            deltas = [0] + deltas + [sum_X]
            Xs = [deltas[i + 1] - deltas[i] for i in range(C)]

            if not (len(Xs) == C and all(x > 0 for x in Xs)):
                continue

            MOD = int(self.np_random.integers(2, self._max_mod + 1))

            # Precompute binomial coefficients up to N*M
            total_cells = N * M
            comb = [[0] * (total_cells + 1) for _ in range(total_cells + 1)]
            for i in range(total_cells + 1):
                comb[i][0] = 1
                for j in range(1, i + 1):
                    comb[i][j] = (comb[i - 1][j] + comb[i - 1][j - 1]) % MOD

            # f[i][j][k]: number of ways to place first k colors into an i×j subboard
            f = [[[0] * (C + 1) for _ in range(M + 1)] for __ in range(N + 1)]
            f[0][0][0] = 1

            # Process each color one by one
            for k in range(1, C + 1):
                x = Xs[k - 1]
                # g[a][b]: number of ways to place x pieces of this color into an a×b rectangle
                # so that every row and column used by it has at least one piece,
                # by inclusion–exclusion
                g = [[0] * (M + 1) for _ in range(N + 1)]
                for a in range(1, N + 1):
                    for b in range(1, M + 1):
                        if a * b < x:
                            continue
                        # total ways to choose x squares out of a*b
                        val = comb[a * b][x]
                        # subtract configurations that leave an unused border row or column
                        for la in range(1, a + 1):
                            for lb in range(1, b + 1):
                                if la < a or lb < b:
                                    val -= g[la][lb] * comb[a][la] * comb[b][lb]
                        g[a][b] = val % MOD

                # Transition: add this color's placements to all previous subboards
                for i in range(1, N + 1):
                    for j in range(1, M + 1):
                        # split the i×j board into an l×r part (already filled with k−1 colors)
                        # and a (i−l)×(j−r) part filled with k-th color
                        for l in range(i):
                            for r in range(j):
                                ti, tj = i - l, j - r
                                if ti * tj < x:
                                    continue
                                ways = (
                                    f[l][r][k - 1]
                                    * g[ti][tj]
                                    * comb[N - l][ti]
                                    * comb[M - r][tj]
                                ) % MOD
                                f[i][j][k] = (f[i][j][k] + ways) % MOD

            # Sum over all non-empty subboards
            answer = 0
            for i in range(1, N + 1):
                for j in range(1, M + 1):
                    answer = (answer + f[i][j][C]) % MOD

            if answer > 0:
                self._n = N
                self._m = M
                self._c = C
                self._xs = Xs
                self._mod = MOD
                self._reference_answer = answer
                break

    def _prompt_generate(self) -> str:
        """Generate the prompt text for the problem."""
        if self._n is None:
            raise RuntimeError("No problem generated")
        return self.prompt_template.format(
            N=self._n,
            M=self._m,
            C=self._c,
            C_minus_1=self._c - 1,
            Xs=" ".join(f"X[{i}]={x}" for i, x in enumerate(self._xs)),
            MOD=self._mod,
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
        """Score the answer based on correctness.

        Returns:
            -1.0: wrong format (not an integer)
            -0.5: wrong range (not in [0, MOD))
             0.0: wrong answer
            +1.0: correct answer
        """
        processed_result = self._process(answer)
        if processed_result is not None:
            if not (0 <= processed_result < self._mod):
                return -0.5
            if processed_result == self._reference_answer:
                return 1.0
            else:
                return 0.0
        else:
            return -1.0

    def render(self) -> Image.Image | list[Image.Image] | None:
        """Render the grid coloring counting problem as an image.

        Shows:
        - Grid dimensions (N x M)
        - Color constraints with visual representation
        - Usage requirements for each color
        """
        if self._n is None:
            raise RuntimeError("No problem generated")

        cell_px = self._cell_px
        padding = self._padding

        # Define color palette (distinct colors for visualization)
        color_palette = [
            (255, 99, 71),    # Red
            (54, 162, 235),   # Blue
            (75, 192, 192),   # Teal
            (255, 206, 86),   # Yellow
            (153, 102, 255),  # Purple
            (255, 159, 64),   # Orange
            (199, 199, 199),  # Gray
            (83, 102, 255),   # Indigo
            (255, 99, 255),   # Pink
            (99, 255, 132),   # Green
        ]

        # Calculate dimensions
        grid_width = self._m * cell_px
        grid_height = self._n * cell_px

        # Space for color legend
        legend_height = max(120, 30 + len(self._xs) * 35)

        width = padding * 2 + grid_width
        height = padding * 3 + grid_height + legend_height

        img = Image.new("RGB", (width, height), (250, 250, 250))
        draw = ImageDraw.Draw(img)

        # Load font
        font_path = None
        font_path_candidate = self.assets_dir / "DejaVuSans.ttf"
        if font_path_candidate.exists():
            font_path = str(font_path_candidate)

        if font_path:
            font_large = ImageFont.truetype(font_path, int(cell_px * 0.35))
            font_medium = ImageFont.truetype(font_path, 18)
            font_small = ImageFont.truetype(font_path, 14)
        else:
            font_large = ImageFont.load_default()
            font_medium = ImageFont.load_default()
            font_small = ImageFont.load_default()

        # Draw grid
        grid_x = padding
        grid_y = padding

        # Draw grid cells
        for r in range(self._n):
            for c in range(self._m):
                x = grid_x + c * cell_px
                y = grid_y + r * cell_px

                # Draw cell with light gray fill
                draw.rectangle(
                    [x, y, x + cell_px, y + cell_px],
                    outline=(100, 100, 100),
                    fill=(240, 240, 240),
                    width=1
                )

        # Draw grid borders (thicker)
        for r in range(self._n + 1):
            y = grid_y + r * cell_px
            draw.line(
                (grid_x, y, grid_x + grid_width, y),
                fill=(30, 30, 30),
                width=2 if r == 0 or r == self._n else 1
            )
        for c in range(self._m + 1):
            x = grid_x + c * cell_px
            draw.line(
                (x, grid_y, x, grid_y + grid_height),
                fill=(30, 30, 30),
                width=2 if c == 0 or c == self._m else 1
            )

        # Draw grid dimensions labels
        dim_text = f"{self._n} × {self._m} grid"
        bbox = draw.textbbox((0, 0), dim_text, font=font_medium)
        tw = bbox[2] - bbox[0]
        draw.text(
            (grid_x + grid_width // 2 - tw // 2, grid_y + grid_height // 2 - 10),
            dim_text,
            fill=(80, 80, 80),
            font=font_medium
        )

        # Draw color legend
        legend_y = grid_y + grid_height + padding * 2

        # Title
        title = "Color Usage Requirements:"
        draw.text((padding, legend_y), title, fill=(30, 30, 30), font=font_medium)
        legend_y += 30

        # Draw each color with its requirement
        for i, x in enumerate(self._xs):
            color = color_palette[i % len(color_palette)]

            # Color box
            box_size = 24
            box_x = padding + 10
            box_y = legend_y
            draw.rectangle(
                [box_x, box_y, box_x + box_size, box_y + box_size],
                fill=color,
                outline=(30, 30, 30),
                width=2
            )

            # Color label
            text = f"Color {i}: use exactly {x} time{'s' if x != 1 else ''}"
            draw.text(
                (box_x + box_size + 10, box_y + 4),
                text,
                fill=(30, 30, 30),
                font=font_small
            )

            legend_y += 35

        # Draw constraint reminder
        legend_y += 5
        constraint_text = f"Constraint: No two different colors in same row/column"
        draw.text((padding, legend_y), constraint_text, fill=(100, 100, 100), font=font_small)
        legend_y += 20
        answer_text = f"Compute count modulo {self._mod}"
        draw.text((padding, legend_y), answer_text, fill=(100, 100, 100), font=font_small)

        return img

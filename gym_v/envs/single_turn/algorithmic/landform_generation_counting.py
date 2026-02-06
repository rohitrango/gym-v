"""Landform generation counting environment for gym-v (self-contained)."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation
from gym_v.logger import get_logger

logger = get_logger()


class LandformGenerationCountingEnv(Env):
    # Meta: source=RLVE, category=algorithmic, turn=single
    """RLVE Landform Generation Counting as a single-turn environment.

    Count the number of distinct height sequences that can be formed by valid
    permutations of terrain points. A permutation is valid if each point has
    fewer preceding points with greater height than its capacity value.

    Source: https://www.luogu.com.cn/problem/P3255
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""You are given two arrays `H` and `C`, each of length {N}:
H: {H}
C: {C}

A permutation `p` of the indices `0` to `{N_minus_1}` (i.e., `p[0], p[1], ..., p[{N_minus_1}]`) is considered **valid** if and only if the following condition holds for every index `i` from `0` to `{N_minus_1}`: there are **fewer than** C[p[i]] indices `j` (j < i) such that H[p[j]] > H[p[i]].
Please count the number of **distinct sequences** `H[p[0]], H[p[1]], ..., H[p[{N_minus_1}]]` that can be obtained by a valid permutation `p`. (Two permutations producing the same `H`-sequence count as one.) Output the result modulo {MOD}."""

    def __init__(
        self,
        max_n: int = 10,
        max_mod: int = 1000000000,
        cell_px: int = 56,
        padding: int = 24,
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._max_n = max_n
        self._max_mod = max_mod
        self._cell_px = cell_px
        self._padding = padding
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._n: int | None = None
        self._h_array: list[int] | None = None
        self._c_array: list[int] | None = None
        self._mod: int | None = None
        self._prompt: str | None = None
        self._oracle_answer: int | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        if self._n and self._h_array and self._c_array and self._mod:
            size_hint = f"N={self._n}"
            h_hint = ", ".join(f"H[{i}]={h}" for i, h in enumerate(self._h_array))
            c_hint = ", ".join(f"C[{i}]={c}" for i, c in enumerate(self._c_array))
            mod_value = self._mod
        else:
            size_hint = "N points"
            h_hint = "H[i] = height of point i"
            c_hint = "C[i] = capacity of point i"
            mod_value = "MOD"

        return dedent(
            f"""
            Count the number of distinct height sequences that can be formed by valid
            permutations of terrain points.

            **Understanding Capacity:**
            C[i] (Capacity of point i) indicates how many taller points can appear before
            point i in a valid permutation. In other words, when placing point i in a
            sequence, at most C[i]-1 points with greater height can precede it.

            Example: If C[2]=3, then point 2 can be placed at a position where 0, 1, or 2
            taller points have already been placed before it.

            Given {size_hint} terrain points with heights and capacities:
            - Heights: {h_hint}
            - Capacities: {c_hint}

            A permutation p of indices 0 to N-1 is **valid** if:
            For each position i in the permutation, the number of points j < i
            where H[p[j]] > H[p[i]] is strictly less than C[p[i]].

            Count the number of **distinct height sequences** H[p[0]], H[p[1]], ..., H[p[N-1]]
            that can be formed by valid permutations. Two permutations producing the same
            height sequence count as one.

            In the image:
            - Each point as a vertical bar with height proportional to H[i]
            - Color intensity represents the capacity C[i] (darker = higher capacity)
            - Points are arranged to show the terrain profile

            Output the result modulo {mod_value}.
            """
        ).strip()

    def _get_state_text(self) -> str:
        """Return the text representation of the current state."""
        return self._prompt

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
        """Generate a landform generation counting problem instance.

        Creates height array H and capacity array C, then computes the number of
        distinct height sequences using dynamic programming on contour blocks.
        """
        if self._max_n < 3:
            raise ValueError("max_n must be >= 3")

        N = int(self.np_random.integers(5, self._max_n + 1))

        # Generate example heights
        example_H = [int(self.np_random.integers(1, N + 1)) for _ in range(N)]

        # Generate (height, capacity) pairs
        A = [None] * N
        for i, Hi in enumerate(example_H):
            # Count how many heights before position i are greater than Hi
            min_capacity = sum(int(Hj > Hi) for Hj in example_H[:i]) + 1
            # Count total number of heights greater than Hi
            max_capacity = sum(int(Hj > Hi) for Hj in example_H) + 1
            # Capacity must be in [min_capacity, max_capacity]
            A[i] = (Hi, int(self.np_random.integers(min_capacity, max_capacity + 1)))

        # Shuffle the pairs
        self.np_random.shuffle(A)

        MOD = int(self.np_random.integers(2, self._max_mod + 1))

        # Extract H and C arrays for storage
        self._n = N
        self._h_array = [a[0] for a in A]
        self._c_array = [a[1] for a in A]
        self._mod = MOD

        # ---------- Compute answer using dynamic programming ----------
        # Sort by height descending, capacity ascending
        A_sorted = sorted(A, key=lambda x: (-x[0], x[1]))

        # Process same-height blocks
        ans_heights = 1
        start = 0
        while start < N:
            end = start
            h_cur = A_sorted[start][0]
            # Find end of same-height block
            while end + 1 < N and A_sorted[end + 1][0] == h_cur:
                end += 1

            processed = start + 1  # 1-based count of processed positions

            # Dynamic programming for this block
            dp = [0] * (processed + 2)  # dp[0 ... processed]

            # Initialize with first element in block
            first_key = A_sorted[start][1]
            for j in range(1, min(processed, first_key) + 1):
                dp[j] = 1

            # Process remaining elements in block
            for i in range(start + 1, end + 1):
                key = A_sorted[i][1]
                limit = min(processed, key)
                # Update dp with prefix sums
                for j in range(1, limit + 1):
                    dp[j] = (dp[j] + dp[j - 1]) % MOD

            # Sum valid configurations for last element
            last_key = A_sorted[end][1]
            res = sum(dp[1 : min(processed, last_key) + 1]) % MOD
            ans_heights = (ans_heights * res) % MOD

            start = end + 1  # Move to next block

        self._oracle_answer = ans_heights

    def _prompt_generate(self) -> str:
        """Generate the prompt text for the problem."""
        if self._n is None:
            raise RuntimeError("No problem generated")
        return self.prompt_template.format(
            N=self._n,
            N_minus_1=self._n - 1,
            H=" ".join(f"H[{i}]={h}" for i, h in enumerate(self._h_array)),
            C=" ".join(f"C[{i}]={c}" for i, c in enumerate(self._c_array)),
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
            -0.25: wrong answer
            +1.0: correct answer
        """
        processed_result = self._process(answer)
        if processed_result is not None:
            if not (0 <= processed_result < self._mod):
                return 0.0
            if processed_result == self._oracle_answer:
                return 1.0
            else:
                return 0.0
        else:
            return 0.0

    def render(self) -> Image.Image | list[Image.Image] | None:
        """Render the landform generation counting problem as an image.

        Shows:
        - Terrain profile with height bars
        - Color gradient based on capacity (elevation constraints)
        - Height and capacity values for each point
        - Professional geographic visualization style
        """
        if self._n is None:
            raise RuntimeError("No problem generated")

        padding = self._padding
        cell_width = self._cell_px

        # Calculate dimensions
        n_points = self._n
        max_height = max(self._h_array)
        max_capacity = max(self._c_array)

        # Scale heights for visualization
        height_scale = 200 / max_height if max_height > 0 else 1
        terrain_width = n_points * cell_width
        terrain_height = 250

        # Overall image dimensions
        legend_height = 220
        width = padding * 2 + terrain_width
        height = padding * 3 + terrain_height + legend_height

        img = Image.new("RGB", (width, height), (250, 250, 250))
        draw = ImageDraw.Draw(img)

        # Load font
        font_path = None
        font_path_candidate = self.assets_dir / "DejaVuSans.ttf"
        if font_path_candidate.exists():
            font_path = str(font_path_candidate)

        if font_path:
            font_large = ImageFont.truetype(font_path, 20)
            font_medium = ImageFont.truetype(font_path, 16)
            font_small = ImageFont.truetype(font_path, 12)
        else:
            font_large = ImageFont.load_default()
            font_medium = ImageFont.load_default()
            font_small = ImageFont.load_default()

        # Draw terrain profile
        terrain_x = padding
        terrain_y = padding
        baseline_y = terrain_y + terrain_height - 20

        # Define color gradient for capacity (elevation constraints)
        def get_terrain_color(capacity: int) -> tuple[int, int, int]:
            """Map capacity to color using a gradient.

            Low capacity (blue) -> Mid capacity (green) -> High capacity (brown).
            """
            ratio = capacity / max_capacity if max_capacity > 0 else 0
            if ratio < 0.33:
                # Blue to Cyan
                t = ratio / 0.33
                r = int(65 + (70 - 65) * t)
                g = int(105 + (180 - 105) * t)
                b = int(225 + (200 - 225) * t)
            elif ratio < 0.67:
                # Cyan to Green
                t = (ratio - 0.33) / 0.34
                r = int(70 + (85 - 70) * t)
                g = int(180 + (170 - 180) * t)
                b = int(200 + (85 - 200) * t)
            else:
                # Green to Brown
                t = (ratio - 0.67) / 0.33
                r = int(85 + (139 - 85) * t)
                g = int(170 + (90 - 170) * t)
                b = int(85 + (65 - 85) * t)
            return (r, g, b)

        # Draw each terrain point as a bar
        for i in range(n_points):
            h = self._h_array[i]
            c = self._c_array[i]

            bar_height = h * height_scale
            x_left = terrain_x + i * cell_width
            x_right = x_left + cell_width - 4
            y_top = baseline_y - bar_height

            # Get color based on capacity
            color = get_terrain_color(c)

            # Draw bar with gradient effect
            draw.rectangle(
                [x_left, y_top, x_right, baseline_y],
                fill=color,
                outline=(50, 50, 50),
                width=2,
            )

            # Add subtle highlight
            highlight_color = tuple(min(255, c + 40) for c in color)
            highlight_width = (x_right - x_left) // 4
            draw.rectangle(
                [x_left + 2, y_top + 2, x_left + highlight_width, baseline_y - 2],
                fill=highlight_color,
                outline=None,
            )

            # Draw height label on top of bar
            h_text = str(h)
            bbox = draw.textbbox((0, 0), h_text, font=font_small)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            text_x = x_left + (x_right - x_left) // 2 - tw // 2
            text_y = y_top - th - 4
            draw.text((text_x, text_y), h_text, fill=(30, 30, 30), font=font_small)

            # Draw index below baseline
            idx_text = str(i)
            bbox = draw.textbbox((0, 0), idx_text, font=font_small)
            tw = bbox[2] - bbox[0]
            text_x = x_left + (x_right - x_left) // 2 - tw // 2
            draw.text(
                (text_x, baseline_y + 4), idx_text, fill=(80, 80, 80), font=font_small
            )

        # Draw baseline
        draw.line(
            (terrain_x, baseline_y, terrain_x + terrain_width, baseline_y),
            fill=(60, 60, 60),
            width=2,
        )

        # Draw title
        title = "Landform Terrain Profile"
        bbox = draw.textbbox((0, 0), title, font=font_large)
        tw = bbox[2] - bbox[0]
        draw.text(
            (width // 2 - tw // 2, terrain_y - 8),
            title,
            fill=(30, 30, 30),
            font=font_large,
        )

        # Draw legend section
        legend_y = terrain_y + terrain_height + padding * 2

        # Title
        legend_title = "Configuration Details:"
        draw.text(
            (padding, legend_y), legend_title, fill=(30, 30, 30), font=font_medium
        )
        legend_y += 28

        # Heights
        h_text = "Heights (H): " + ", ".join(
            f"[{i}]={h}" for i, h in enumerate(self._h_array)
        )
        draw.text((padding + 10, legend_y), h_text, fill=(60, 60, 60), font=font_small)
        legend_y += 22

        # Capacities
        c_text = "Capacities (C): " + ", ".join(
            f"[{i}]={c}" for i, c in enumerate(self._c_array)
        )
        draw.text((padding + 10, legend_y), c_text, fill=(60, 60, 60), font=font_small)
        legend_y += 28

        # Draw color scale legend
        scale_title = "Capacity Color Scale:"
        draw.text((padding, legend_y), scale_title, fill=(30, 30, 30), font=font_medium)
        legend_y += 24

        # Color gradient bar
        gradient_width = terrain_width // 2
        gradient_height = 20
        gradient_x = padding + 10
        gradient_y = legend_y

        # Draw gradient in segments
        segments = 50
        for seg in range(segments):
            ratio = seg / segments
            capacity_val = int(ratio * max_capacity)
            color = get_terrain_color(capacity_val)
            seg_width = gradient_width // segments
            draw.rectangle(
                [
                    gradient_x + seg * seg_width,
                    gradient_y,
                    gradient_x + (seg + 1) * seg_width,
                    gradient_y + gradient_height,
                ],
                fill=color,
                outline=None,
            )

        # Gradient border
        draw.rectangle(
            [
                gradient_x,
                gradient_y,
                gradient_x + gradient_width,
                gradient_y + gradient_height,
            ],
            outline=(60, 60, 60),
            width=2,
        )

        # Labels for gradient
        draw.text(
            (gradient_x, gradient_y + gradient_height + 4),
            "Low",
            fill=(80, 80, 80),
            font=font_small,
        )
        bbox = draw.textbbox((0, 0), "High", font=font_small)
        tw = bbox[2] - bbox[0]
        draw.text(
            (gradient_x + gradient_width - tw, gradient_y + gradient_height + 4),
            "High",
            fill=(80, 80, 80),
            font=font_small,
        )

        # Add constraint info
        legend_y += gradient_height + 26
        constraint_text = f"Task: Count distinct height sequences from valid permutations (mod {self._mod})"
        draw.text(
            (padding, legend_y), constraint_text, fill=(100, 100, 100), font=font_small
        )

        return img

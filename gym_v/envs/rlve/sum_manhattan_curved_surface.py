"""Sum Manhattan curved surface environment for gym-v (self-contained)."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation
from gym_v.logger import get_logger

logger = get_logger()


class RLVESumManhattanCurvedSurfaceEnv(Env):
    """RLVE Sum Manhattan Curved Surface as a single-turn environment.

    This environment computes P(k) as the sum of (|x| + |y| + |z|)^2 over all
    integer triples (x, y, z) such that x × y × z = k. The task is to compute
    the sum of P(k) for all integers k in the range [A, B] (inclusive).
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""Define P(k) as the sum of (|x| + |y| + |z|)^2 over all integer triples (x, y, z) such that x × y × z = k. Compute the sum of P(k) for all integers k in the range [{A}, {B}] (inclusive)."""

    def __init__(
        self,
        max_a_b: int = 100,
        cell_px: int = 80,
        padding: int = 32,
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._max_a_b = max_a_b
        self._cell_px = cell_px
        self._padding = padding
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._a: int | None = None
        self._b: int | None = None
        self._prompt: str | None = None
        self._reference_answer: int | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        range_hint = (
            f"[{self._a}, {self._b}]"
            if self._a is not None and self._b is not None
            else "[A, B]"
        )
        return dedent(
            f"""
            Sum Manhattan Curved Surface problem:

            For each integer k, define P(k) as the sum of (|x| + |y| + |z|)^2 over all
            integer triples (x, y, z) such that x × y × z = k.

            The Manhattan distance |x| + |y| + |z| from the origin to point (x, y, z)
            creates a "curved surface" when squared and summed over all divisor triples.

            Given a range {range_hint}, compute the sum of P(k) for all k in this range.

            Mathematical insight:
            - Each k has divisors that form triples (x, y, z) where xyz = k
            - The squared Manhattan distance (|x| + |y| + |z|)^2 includes terms like:
              x^2 + y^2 + z^2 + 2(|x||y| + |x||z| + |y||z|)
            - Summing over a range creates a complex curved surface pattern

            In the visualization:
            - The heatmap shows P(k) values across the range [A, B]
            - Warmer colors (red/yellow) indicate higher P(k) values
            - Cooler colors (blue/purple) indicate lower P(k) values
            - The pattern reveals the mathematical structure of divisor sums

            Output format: A single integer — the sum of P(k) for k ∈ [A, B].
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
        """Generate random A and B values and compute the reference answer."""
        max_a_b = self._max_a_b
        if max_a_b < 1:
            raise ValueError("max_a_b must be >= 1")

        self._a = int(self.np_random.integers(1, max_a_b + 1))
        self._b = int(self.np_random.integers(self._a, max_a_b + 1))

        def funa(left: int, right: int) -> int:
            """Sum of i for i in [left..right]."""
            cnt = right - left + 1
            return (left + right) * cnt // 2

        def ready(x: int) -> int:
            """Sum of i^2 for i in [1..x]."""
            return x * (x + 1) * (2 * x + 1) // 6

        def funb(left: int, right: int) -> int:
            """Sum of i^2 for i in [left..right]."""
            return ready(right) - ready(left - 1)

        def work2(n: int) -> tuple[int, int, int]:
            """Compute the three helper sums for a given n.

            ans1 = sum_{i=1..n} floor(n/i)
            ans2 = sum_{i=1..n} [ sum_{j=1..i} j + i * sum_{j=1..floor(n/i)} j ]
            ans3 = sum_{i=1..n} [ sum_{j=1..i} j^2 + i * sum_{j=1..floor(n/i)} j^2
                                  + 2 * (sum_{j=1..i} j) * (sum_{k=1..floor(n/i)} k) ]

            Uses divisor grouping to run in ~O(sqrt(n)).
            """
            ans1 = ans2 = ans3 = 0
            left_idx = 1
            while left_idx <= n:
                d = n // left_idx
                right_idx = n // d
                cnt = right_idx - left_idx + 1

                # Accumulate contributions
                ans1 += cnt * d
                ans2 += funa(left_idx, right_idx) * d + cnt * funa(1, d)
                ans3 += (
                    funb(left_idx, right_idx) * d
                    + cnt * funb(1, d)
                    + 2 * funa(left_idx, right_idx) * funa(1, d)
                )

                left_idx = right_idx + 1

            return ans1, ans2, ans3

        def work(n: int) -> int:
            """Compute the cumulative beauty sum S(n) = sum_{k=1..n} P(k)/4.

            The final answer is 4*(S(b) - S(a-1)).
            """
            ans = 0
            left_idx = 1
            while left_idx <= n:
                d = n // left_idx
                right_idx = n // d
                cnt = right_idx - left_idx + 1

                a1, a2, a3 = work2(d)
                ans += funb(left_idx, right_idx) * a1 + funa(left_idx, right_idx) * 2 * a2 + cnt * a3

                left_idx = right_idx + 1

            return ans

        result = work(self._b) - work(self._a - 1)
        result = result * 4
        if result <= 0:
            result = 1  # Ensure positive result
        self._reference_answer = result

    def _prompt_generate(self) -> str:
        """Generate the prompt string."""
        if self._a is None or self._b is None:
            raise RuntimeError("No A and B values generated")
        return self.prompt_template.format(A=self._a, B=self._b)

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
            return -1.0

        if processed_result < 0:
            return -1.0

        # Use (min/max)^beta strategy
        a, b = self._reference_answer, processed_result
        beta = 10.0
        return (min(a, b) / max(a, b)) ** beta

    def render(self) -> Image.Image | list[Image.Image] | None:
        """Render a beautiful curved surface visualization."""
        if self._a is None or self._b is None:
            raise RuntimeError("No A and B values generated")

        # Compute P(k) for visualization (only for a small sample if range is large)
        range_size = self._b - self._a + 1
        max_display = 20  # Maximum number of values to display
        sample_size = min(range_size, max_display)

        # Sample k values evenly across the range
        if range_size <= max_display:
            k_values = list(range(self._a, self._b + 1))
        else:
            indices = np.linspace(0, range_size - 1, sample_size, dtype=int)
            k_values = [self._a + i for i in indices]

        # Compute P(k) for sampled values (simplified approximation for visualization)
        p_values = []
        for k in k_values:
            # Simplified P(k) approximation for visualization
            # Actual computation is complex, so we use a heuristic
            # P(k) grows roughly with divisor function and k
            divisor_count = sum(1 for i in range(1, min(k + 1, 100)) if k % i == 0)
            p_k_approx = (divisor_count * k) ** 1.5
            p_values.append(p_k_approx)

        # Normalize P values for color mapping
        if p_values:
            min_p = min(p_values)
            max_p = max(p_values)
            if max_p > min_p:
                normalized_p = [
                    (p - min_p) / (max_p - min_p) for p in p_values
                ]
            else:
                normalized_p = [0.5] * len(p_values)
        else:
            normalized_p = []

        # Create visualization
        cell_px = self._cell_px
        padding = self._padding

        # Layout: grid of cells
        cols = min(10, len(k_values))
        rows = (len(k_values) + cols - 1) // cols

        width = padding * 2 + cols * cell_px + padding
        height = padding * 3 + rows * cell_px + padding * 2
        img = Image.new("RGB", (width, height), (245, 245, 250))
        draw = ImageDraw.Draw(img)

        # Load font
        font_path = None
        font_path_candidate = self.assets_dir / "DejaVuSans.ttf"
        if font_path_candidate.exists():
            font_path = str(font_path_candidate)

        if font_path:
            font_small = ImageFont.truetype(font_path, int(cell_px * 0.15))
            font_title = ImageFont.truetype(font_path, int(cell_px * 0.3))
        else:
            font_small = ImageFont.load_default()
            font_title = ImageFont.load_default()

        # Draw title
        title = f"P(k) Heatmap: Range [{self._a}, {self._b}]"
        title_bbox = draw.textbbox((0, 0), title, font=font_title)
        title_w = title_bbox[2] - title_bbox[0]
        title_x = (width - title_w) // 2
        draw.text((title_x, padding // 2), title, fill=(40, 40, 60), font=font_title)

        # Draw heatmap cells
        for idx, (k, p_norm) in enumerate(zip(k_values, normalized_p, strict=True)):
            row = idx // cols
            col = idx % cols

            x0 = padding + col * cell_px
            y0 = padding * 2 + row * cell_px
            x1 = x0 + cell_px - 4
            y1 = y0 + cell_px - 4

            # Color mapping: blue (low) -> cyan -> green -> yellow -> red (high)
            # Using a smooth gradient
            if p_norm < 0.25:
                # Blue to Cyan
                t = p_norm / 0.25
                r = int(50 + 30 * t)
                g = int(100 + 155 * t)
                b = int(220 - 20 * t)
            elif p_norm < 0.5:
                # Cyan to Green
                t = (p_norm - 0.25) / 0.25
                r = int(80 + 20 * t)
                g = int(255 - 55 * t)
                b = int(200 - 100 * t)
            elif p_norm < 0.75:
                # Green to Yellow
                t = (p_norm - 0.5) / 0.25
                r = int(100 + 155 * t)
                g = int(200 + 55 * t)
                b = int(100 - 100 * t)
            else:
                # Yellow to Red
                t = (p_norm - 0.75) / 0.25
                r = 255
                g = int(255 - 105 * t)
                b = 0

            fill_color = (r, g, b)
            draw.rounded_rectangle([x0, y0, x1, y1], radius=8, fill=fill_color)

            # Draw border
            draw.rounded_rectangle(
                [x0, y0, x1, y1], radius=8, outline=(30, 30, 30), width=2
            )

            # Draw k value
            k_str = f"k={k}"
            k_bbox = draw.textbbox((0, 0), k_str, font=font_small)
            k_w = k_bbox[2] - k_bbox[0]
            k_h = k_bbox[3] - k_bbox[1]
            cx = x0 + (cell_px - 4) // 2
            cy = y0 + (cell_px - 4) // 2

            # Draw shadow for better visibility
            shadow_color = (0, 0, 0) if p_norm > 0.5 else (255, 255, 255)
            text_color = (255, 255, 255) if p_norm > 0.5 else (0, 0, 0)

            draw.text(
                (cx - k_w // 2 + 1, cy - k_h // 2 + 1),
                k_str,
                fill=shadow_color,
                font=font_small,
            )
            draw.text((cx - k_w // 2, cy - k_h // 2), k_str, fill=text_color, font=font_small)

        # Draw color scale legend
        legend_y = padding * 2 + rows * cell_px + padding
        legend_x0 = padding
        legend_width = width - padding * 2
        legend_height = 30

        # Draw gradient bar
        for i in range(legend_width):
            t = i / legend_width
            if t < 0.25:
                tt = t / 0.25
                r = int(50 + 30 * tt)
                g = int(100 + 155 * tt)
                b = int(220 - 20 * tt)
            elif t < 0.5:
                tt = (t - 0.25) / 0.25
                r = int(80 + 20 * tt)
                g = int(255 - 55 * tt)
                b = int(200 - 100 * tt)
            elif t < 0.75:
                tt = (t - 0.5) / 0.25
                r = int(100 + 155 * tt)
                g = int(200 + 55 * tt)
                b = int(100 - 100 * tt)
            else:
                tt = (t - 0.75) / 0.25
                r = 255
                g = int(255 - 105 * tt)
                b = 0

            draw.line(
                [(legend_x0 + i, legend_y), (legend_x0 + i, legend_y + legend_height)],
                fill=(r, g, b),
                width=1,
            )

        # Draw legend labels
        draw.text(
            (legend_x0, legend_y + legend_height + 5),
            "Low P(k)",
            fill=(40, 40, 60),
            font=font_small,
        )
        legend_high_text = "High P(k)"
        high_bbox = draw.textbbox((0, 0), legend_high_text, font=font_small)
        high_w = high_bbox[2] - high_bbox[0]
        draw.text(
            (legend_x0 + legend_width - high_w, legend_y + legend_height + 5),
            legend_high_text,
            fill=(40, 40, 60),
            font=font_small,
        )

        return img

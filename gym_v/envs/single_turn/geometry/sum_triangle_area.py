"""Sum of triangle areas environment for gym-v (self-contained)."""

from __future__ import annotations

import functools
from textwrap import dedent
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from gym_v import Env, Observation
from gym_v.logger import get_logger

matplotlib.use("Agg")
logger = get_logger()


class SumTriangleAreaEnv(Env):
    # Meta: source=RLVE, category=geometry, turn=single
    """RLVE Sum of triangle areas as a single-turn environment.

    Source: https://www.luogu.com.cn/problem/P3476
    """

    prompt_template = r"""There are {N} points in a 2D plane, each represented by its coordinates (x, y). The points are given as follows:
{points}

Please compute the **sum of the areas of all triangles** that can be formed by any three distinct points in this set. If a triangle is degenerate (i.e., the three points are collinear), its area is considered 0. **Output the total area multiplied by 2** (i.e., twice the sum of all triangle areas), which will always be an integer (think about why this is the case)."""

    def __init__(
        self,
        max_n: int = 10,
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._max_n = max_n
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._n: int | None = None
        self._points: list[tuple[int, int]] | None = None
        self._oracle_answer: int | None = None
        self._prompt: str | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        n_hint = str(self._n) if self._n else "N"
        return dedent(
            f"""
            Sum of Triangle Areas:

            Compute the sum of the areas of all triangles that can be formed by
            any three distinct points in a set. If a triangle is degenerate
            (i.e., the three points are collinear), its area is considered 0.

            Rules:
            - For three points (x1, y1), (x2, y2), (x3, y3), the triangle area is:
              Area = |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)| / 2
            - If three points are collinear, the area is 0.
            - The problem uses an efficient O(n^2 log n) algorithm that sorts
              points and uses polar angle sorting to avoid checking all O(n^3)
              triangles explicitly.

            In the image:
            - {n_hint} points are shown as blue dots in a 2D plane.
            - A few example triangles are highlighted with different colors.

            Output Format:
            Output the total area multiplied by 2 (i.e., twice the sum of all
            triangle areas), which will always be an integer.
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
            metadata={
                "state_text": state_text,
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
            text=None,
            metadata={
                "state_text": state_text,
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
        """Generate random points and compute the reference answer.

        Uses an efficient O(n^2 log n) algorithm based on polar angle sorting
        around each point to compute the sum of all triangle areas.
        """
        if self._max_n < 3:
            raise ValueError("max_n must be >= 3")

        self._n = int(self.np_random.integers(3, self._max_n + 1))
        N = self._n

        # Sample N distinct points from a grid
        all_points = [(x, y) for x in range(0, N + 1) for y in range(0, N + 1)]
        indices = self.np_random.choice(len(all_points), size=N, replace=False)
        self._points = [all_points[i] for i in indices]

        # Compute reference answer using efficient algorithm
        A = sorted(self._points, key=lambda p: (p[0], p[1]))

        ans = 0
        for i in range(N):
            xi, yi = A[i]
            # Build vectors from A[i] to all later points
            s = [(x - xi, y - yi) for x, y in A[i + 1 :]]
            # Sort by polar angle around the origin using cross-product comparator
            s.sort(
                key=functools.cmp_to_key(
                    lambda a, b: (
                        -1
                        if a[1] * b[0] < a[0] * b[1]
                        else (1 if a[1] * b[0] > a[0] * b[1] else 0)
                    )
                )
            )

            m = len(s)
            # Build suffix sums of x- and y-components
            sx = [0] * (m + 1)
            sy = [0] * (m + 1)
            for j in range(m - 1, -1, -1):
                sx[j] = sx[j + 1] + s[j][0]
                sy[j] = sy[j + 1] + s[j][1]
                # Accumulate cross-products to sum triangle areas (twice the area)
                ans += s[j][0] * sy[j + 1] - s[j][1] * sx[j + 1]

        self._oracle_answer = ans

    def _prompt_generate(self) -> str:
        """Generate the prompt from the current points."""
        if self._points is None:
            raise RuntimeError("No points generated")
        return self.prompt_template.format(
            N=self._n,
            points="\n".join(f"({x}, {y})" for x, y in self._points),
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
        """Score the answer using the (min/max)^beta strategy."""
        processed_result = self._process(answer)
        if processed_result is None:
            return 0.0
        if processed_result < 0:
            return 0.0
        if processed_result == 0:
            return 1.0 if self._oracle_answer == 0 else 0.0

        a, b = self._oracle_answer, processed_result
        return (min(a, b) / max(a, b)) ** 10

    def render(self) -> Image.Image | list[Image.Image] | None:
        """Render the 2D points with example triangles.

        Creates a clean mathematical plot showing:
        - All points as blue dots
        - A few example triangles with different colors
        - Coordinate axes
        - Grid for reference
        """
        if self._points is None:
            raise RuntimeError("No points generated")

        # Create figure with clean mathematical style
        fig, ax = plt.subplots(figsize=(8, 8), dpi=100)

        # Extract x and y coordinates
        xs = [p[0] for p in self._points]
        ys = [p[1] for p in self._points]

        # Set up the plot limits with some padding
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        x_range = x_max - x_min
        y_range = y_max - y_min
        padding = max(x_range, y_range) * 0.15 + 0.5

        ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_ylim(y_min - padding, y_max + padding)

        # Add grid
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

        # Draw coordinate axes
        ax.axhline(y=0, color="k", linewidth=0.8, alpha=0.3)
        ax.axvline(x=0, color="k", linewidth=0.8, alpha=0.3)

        # Draw a few example triangles (up to 3) with different colors
        n_triangles = min(3, len(self._points) // 3)
        triangle_colors = ["#FF6B6B", "#4ECDC4", "#FFD93D"]

        rng_state = np.random.RandomState(42)
        for i in range(n_triangles):
            # Select 3 random distinct points
            indices = rng_state.choice(len(self._points), size=3, replace=False)
            triangle_points = [self._points[idx] for idx in indices]

            # Draw the triangle
            triangle_xs = [p[0] for p in triangle_points] + [triangle_points[0][0]]
            triangle_ys = [p[1] for p in triangle_points] + [triangle_points[0][1]]

            ax.fill(
                triangle_xs,
                triangle_ys,
                color=triangle_colors[i % len(triangle_colors)],
                alpha=0.2,
                edgecolor=triangle_colors[i % len(triangle_colors)],
                linewidth=2,
            )

        # Plot all points as blue dots on top
        ax.scatter(
            xs,
            ys,
            c="darkblue",
            s=120,
            zorder=5,
            edgecolors="white",
            linewidths=2,
        )

        # Add point labels
        for i, (x, y) in enumerate(self._points):
            ax.annotate(
                f"P{i}",
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
                color="darkblue",
                weight="bold",
            )

        # Set labels and title
        ax.set_xlabel("x", fontsize=12, weight="bold")
        ax.set_ylabel("y", fontsize=12, weight="bold")
        ax.set_title(
            f"{self._n} Points in 2D Plane\n(showing {n_triangles} example triangles)",
            fontsize=13,
            weight="bold",
            pad=15,
        )

        # Equal aspect ratio for proper geometric representation
        ax.set_aspect("equal", adjustable="box")

        # Clean up the plot
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Convert matplotlib figure to PIL Image
        fig.tight_layout()
        fig.canvas.draw()

        # Get the RGBA buffer from the figure
        img_buffer = fig.canvas.buffer_rgba()
        width, height = fig.canvas.get_width_height()
        img = Image.frombytes("RGBA", (width, height), img_buffer).convert("RGB")

        plt.close(fig)

        return img

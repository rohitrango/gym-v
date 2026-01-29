"""Largest rectangle among points environment for gym-v (self-contained)."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

from matplotlib.patches import Polygon as MplPolygon
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from gym_v import Env, Observation
from gym_v.logger import get_logger

logger = get_logger()


class RLVELargestRectangleAmongPointsEnv(Env):
    """RLVE largest rectangle among points as a single-turn environment.

    Task: Given N points in 2D plane, find 4 distinct points that form a
    rectangle (not necessarily axis-aligned) with maximum area.
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""You are given a set of {N} points in a 2D plane, each represented by its coordinates `(x, y)`:
{points}

Your task is to find four **distinct** points such that they form a rectangle (NOT necessarily axis-aligned). Among all such rectangles, choose one with the **maximum possible area**.

**Output Format:** Output one line containing the indices (0-based) of the four selected points, separated by spaces."""

    def __init__(
        self,
        max_n: int = 15,
        rewarding_strategy: str = "(answer/gold)^beta",
        rewarding_weight: float = 1.0,
        rewarding_beta: float = 5.0,
        fig_size: tuple[int, int] = (8, 8),
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._max_n = max_n
        self._rewarding_strategy = rewarding_strategy
        self._rewarding_weight = rewarding_weight
        self._rewarding_beta = rewarding_beta
        self._fig_size = fig_size
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._n: int | None = None
        self._points: list[tuple[int, int]] | None = None
        self._gold_answer: int | None = None
        self._oracle_answer: str | None = None
        self._prompt: str | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        if self._n:
            size_hint = f"{self._n} points"
        else:
            size_hint = "N points"
        return dedent(
            f"""
            Largest Rectangle Among Points:

            Given {size_hint} in a 2D plane, find 4 distinct points that form a
            rectangle (not necessarily axis-aligned) with maximum possible area.

            Rectangle definition:
            - Four points A, B, C, D form a rectangle if:
              1) Opposite sides are equal and parallel
              2) All angles are 90 degrees
              3) Diagonals have equal length and bisect each other

            The image shows:
            - Scatter plot of all points with their indices labeled
            - Points are distributed across a 2D coordinate plane

            Output format: Four space-separated indices (0-based) of the points
            forming the maximum-area rectangle.
            Example: "0 3 5 8"
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
            text=state_text,
            metadata={
                "text_prompt": self._prompt,
                "rlve_gold_area": self._gold_answer,
            },
        )
        info = {
            "oracle_answer": self._oracle_answer,
            "gold_area": self._gold_answer,
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
                "rlve_gold_area": self._gold_answer,
            },
        )
        info = {
            "oracle_answer": self._oracle_answer,
            "gold_area": self._gold_answer,
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
        """Generate N points where first 4 form a rectangle."""
        N = int(self.np_random.integers(5, self._max_n + 1))
        self._n = N

        # Generate first 4 points that form a rectangle
        points = []
        x0 = int(self.np_random.integers(-N // 2, N // 2 + 1))
        y0 = int(self.np_random.integers(-N // 2, N // 2 + 1))
        points.append((x0, y0))

        # Generate a rectangle by constructing from one corner
        while True:
            dx = int(self.np_random.integers(-N // 2, N // 2 + 1))
            dy = int(self.np_random.integers(-N // 2, N // 2 + 1))
            if dx == 0 and dy == 0:
                continue
            # A is (x0, y0), B is (x0+dx, y0+dy)
            # D is (x0-dy, y0+dx), C is (x0+dx-dy, y0+dy+dx)
            # This forms a rectangle with sides AB perpendicular to AD
            points.append((x0 + dx, y0 + dy))
            points.append((x0 - dy, y0 + dx))
            points.append((x0 + dx - dy, y0 + dy + dx))
            break

        # Generate remaining random points
        for _ in range(4, N):
            x = int(self.np_random.integers(-N, N + 1))
            y = int(self.np_random.integers(-N, N + 1))
            points.append((x, y))

        # Shuffle to hide the answer
        self.np_random.shuffle(points)
        self._points = points

        # Compute gold answer (maximum rectangle area)
        # Build list of all point-pairs (diagonals), storing:
        # (squared_length, sum_x, sum_y, idx1, idx2)
        lines = []
        for i in range(N):
            xi, yi = points[i]
            for j in range(i + 1, N):
                xj, yj = points[j]
                dx = xi - xj
                dy = yi - yj
                s = dx * dx + dy * dy
                # midpoint * 2 is (xi+xj, yi+yj)
                sx = xi + xj
                sy = yi + yj
                lines.append((s, sx, sy, i, j))

        # Sort by (length, midpoint_x, midpoint_y)
        lines.sort(key=lambda t: (t[0], t[1], t[2]))

        ans = 0
        best_indices = None
        M = len(lines)
        # Scan through sorted diagonals, grouping by equal (s, sx, sy)
        i = 0
        while i < M:
            s0, sx0, sy0, idx1, idx2 = lines[i]
            j = i + 1
            # For each other diagonal with same length and midpoint...
            while (
                j < M
                and lines[j][0] == s0
                and lines[j][1] == sx0
                and lines[j][2] == sy0
            ):
                _, _, _, idx3, idx4 = lines[j]
                # Compute the rectangle area via the cross-product trick:
                # area = |(C-A) × (B-A)|, with A=points[idx1], C=points[idx2], B=points[idx3]
                x1, y1 = points[idx1]  # A
                x2, y2 = points[idx2]  # C (opposite of A)
                x3, y3 = points[idx3]  # B (one endpoint of other diagonal)
                # Determinant = x1*y2 + x2*y3 + x3*y1 - x2*y1 - x3*y2 - x1*y3
                tmp = abs(x1 * y2 + x2 * y3 + x3 * y1 - x2 * y1 - x3 * y2 - x1 * y3)
                if tmp > ans:
                    ans = tmp
                    best_indices = [idx1, idx2, idx3, idx4]
                j += 1
            i += 1

        assert ans > 0, "The maximum area should be greater than 0"
        self._gold_answer = ans
        self._oracle_answer = " ".join(map(str, best_indices))

    def _prompt_generate(self) -> str:
        if self._points is None:
            raise RuntimeError("No points generated")
        return self.prompt_template.format(
            N=self._n,
            points="\n".join(
                f"Point {i}: ({x}, {y})" for i, (x, y) in enumerate(self._points)
            ),
        )

    def _process(self, answer: str | None) -> tuple[int, int, int, int] | None:
        """Process answer string into tuple of 4 indices."""
        if answer is None:
            return None
        answer = answer.strip()
        try:
            indices = list(map(int, answer.split()))
            if len(indices) != 4:
                return None
            return (indices[0], indices[1], indices[2], indices[3])
        except ValueError:
            return None

    def _score_answer(self, answer: str) -> float:
        """Score the answer based on rectangle area."""
        processed_result = self._process(answer)
        if processed_result is None:
            return 0.0
        if not all(0 <= idx < self._n for idx in processed_result):
            return 0.0
        if len(set(processed_result)) != 4:
            return 0.0
        rect_area = self._rectangle_area(
            [self._points[idx] for idx in processed_result]
        )
        if rect_area is None:
            return 0.0
        gold = self._gold_answer
        assert rect_area <= gold, "Answer area should be <= gold area"

        if self._rewarding_strategy == "(answer/gold)^beta":
            return self._rewarding_weight * ((rect_area / gold) ** self._rewarding_beta)
        elif self._rewarding_strategy == "gold=answer":
            return self._rewarding_weight * (rect_area == gold)
        else:
            raise NotImplementedError(
                f"Unknown rewarding strategy: {self._rewarding_strategy}"
            )

    def _rectangle_area(self, P: list[tuple[int, int]]) -> int | None:
        """Check if 4 points form a rectangle and return area."""
        A = P[0]
        others = P[1:]

        # Compute squared distances from A to other points
        d2 = []
        for X in others:
            dx, dy = X[0] - A[0], X[1] - A[1]
            d2.append((dx * dx + dy * dy, X, dx, dy))
        d2.sort(key=lambda t: t[0])

        d1, B, dx1, dy1 = d2[0]
        d2_val, D, dx2, dy2 = d2[1]
        C = d2[2][1]

        # Check for zero-length sides (duplicate points)
        if d1 == 0 or d2_val == 0:
            return None

        # Check perpendicularity: AB ⊥ AD
        if dx1 * dx2 + dy1 * dy2 != 0:
            return None

        # Check parallelogram property: C = B + D - A
        expected_C = (B[0] + D[0] - A[0], B[1] + D[1] - A[1])
        if expected_C != tuple(C):
            return None

        # Compute area
        area = abs(dx1 * dy2 - dy1 * dx2)
        return area

    def render(self) -> Image.Image | list[Image.Image] | None:
        """Render the points as a scatter plot."""
        if self._points is None:
            raise RuntimeError("No points generated")

        fig, ax = plt.subplots(figsize=self._fig_size)

        # Extract coordinates
        xs = [p[0] for p in self._points]
        ys = [p[1] for p in self._points]

        # Plot points
        ax.scatter(
            xs, ys, c="steelblue", s=150, alpha=0.7, zorder=3, edgecolors="black"
        )

        # Label points with indices
        for i, (x, y) in enumerate(self._points):
            ax.annotate(
                str(i),
                (x, y),
                fontsize=10,
                fontweight="bold",
                ha="center",
                va="center",
                color="white",
                zorder=4,
            )

        # Find and draw the maximum rectangle
        self._draw_max_rectangle(ax)

        # Set axis properties
        ax.set_xlabel("X", fontsize=12, fontweight="bold")
        ax.set_ylabel("Y", fontsize=12, fontweight="bold")
        ax.set_title(
            "Find Maximum Rectangle Among Points",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_aspect("equal", adjustable="box")

        # Add some padding to limits
        margin = max(2, (max(xs) - min(xs)) * 0.15)
        ax.set_xlim(min(xs) - margin, max(xs) + margin)
        ax.set_ylim(min(ys) - margin, max(ys) + margin)

        # Set background color
        ax.set_facecolor("#f8f9fa")
        fig.patch.set_facecolor("white")

        # Convert to PIL Image
        fig.tight_layout()
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape((h, w, 4))
        img = Image.fromarray(buf[:, :, :3])  # Drop alpha channel
        plt.close(fig)

        return img

    def _draw_max_rectangle(self, ax) -> None:
        """Find and draw the maximum rectangle."""
        if self._points is None:
            return

        N = len(self._points)
        points = self._points

        # Build list of all point-pairs (diagonals)
        lines = []
        for i in range(N):
            xi, yi = points[i]
            for j in range(i + 1, N):
                xj, yj = points[j]
                dx = xi - xj
                dy = yi - yj
                s = dx * dx + dy * dy
                sx = xi + xj
                sy = yi + yj
                lines.append((s, sx, sy, i, j))

        # Sort by (length, midpoint_x, midpoint_y)
        lines.sort(key=lambda t: (t[0], t[1], t[2]))

        # Find the maximum rectangle
        max_area = 0
        max_rect_indices = None

        M = len(lines)
        i = 0
        while i < M:
            s0, sx0, sy0, idx1, idx2 = lines[i]
            j = i + 1
            while (
                j < M
                and lines[j][0] == s0
                and lines[j][1] == sx0
                and lines[j][2] == sy0
            ):
                _, _, _, idx3, idx4 = lines[j]
                x1, y1 = points[idx1]
                x2, y2 = points[idx2]
                x3, y3 = points[idx3]
                tmp = abs(x1 * y2 + x2 * y3 + x3 * y1 - x2 * y1 - x3 * y2 - x1 * y3)
                if tmp > max_area:
                    max_area = tmp
                    # The four corners are idx1, idx2, idx3, idx4
                    max_rect_indices = [idx1, idx2, idx3, idx4]
                j += 1
            i += 1

        # Draw the rectangle if found
        if max_rect_indices and max_area > 0:
            rect_points = [points[idx] for idx in max_rect_indices]
            # Order points to form a proper polygon (counterclockwise)
            rect_points = self._order_rectangle_points(rect_points)
            if rect_points:
                polygon = MplPolygon(
                    rect_points,
                    fill=False,
                    edgecolor="crimson",
                    linewidth=4,
                    linestyle="-",
                    alpha=0.9,
                    zorder=2,
                )
                ax.add_patch(polygon)
                # Add area label
                cx = sum(p[0] for p in rect_points) / 4
                cy = sum(p[1] for p in rect_points) / 4
                ax.text(
                    cx,
                    cy,
                    f"Area={max_area}",
                    fontsize=11,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    bbox=dict(
                        boxstyle="round,pad=0.5",
                        facecolor="crimson",
                        alpha=0.8,
                        edgecolor="darkred",
                    ),
                    color="white",
                    zorder=5,
                )

    def _order_rectangle_points(
        self, points: list[tuple[int, int]]
    ) -> list[tuple[int, int]] | None:
        """Order 4 points to form a proper rectangle polygon."""
        if len(points) != 4:
            return None

        # Find centroid
        cx = sum(p[0] for p in points) / 4
        cy = sum(p[1] for p in points) / 4

        # Sort by angle from centroid
        def angle(p):
            return np.arctan2(p[1] - cy, p[0] - cx)

        sorted_points = sorted(points, key=angle)
        return sorted_points

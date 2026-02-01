"""Largest convex polygon environment for gym-v (self-contained)."""

from __future__ import annotations

from functools import cmp_to_key
from importlib import resources
from textwrap import dedent
from typing import Any

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon as MPLPolygon
import numpy as np
from PIL import Image

from gym_v import Env, Observation
from gym_v.logger import get_logger

matplotlib.use("Agg")
logger = get_logger()


class RLVELargestConvexPolygonEnv(Env):
    """RLVE Largest Convex Polygon as a single-turn environment.

    Find the maximum number of points from a 2D point set that can form
    a convex polygon.
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""You are given {N} points in the 2D plane, labeled from 1 to {N}. No two points share the same coordinates, and no three points are collinear:
{points}

Find a subset of distinct points that forms the vertices of a **convex polygon**, and maximize the number of points in this subset; please output the labels of the selected points in one line, separated by spaces (in any order); if multiple answers exist, output any one."""

    def __init__(
        self,
        n_points: int = 10,
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._n_points = n_points
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._points: list[tuple[int, int]] | None = None
        self._gold_answer: int | None = None
        self._oracle_answer: str | None = None
        self._prompt: str | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        n_hint = self._n_points if self._n_points else "N"
        return dedent(
            f"""
            Largest Convex Polygon Problem:
            Given {n_hint} points in the 2D plane with unique coordinates and no three collinear points,
            find the maximum number of points that can form vertices of a convex polygon.

            Rules:
            1) Select a subset of distinct point labels (numbered 1 to {n_hint})
            2) The selected points must form a convex polygon
            3) Maximize the number of selected points

            A polygon is convex if all interior angles are less than 180 degrees,
            or equivalently, if all points lie on the convex hull of the selected subset.

            Output format: Space-separated list of point labels (integers from 1 to {n_hint})
            Example: "1 3 5 7" (order doesn't matter)
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
                "rlve_gold_answer": self._gold_answer,
            },
        )
        info = {
            "gold_answer": self._gold_answer,
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
            metadata={
                "state_text": state_text,
                "text_prompt": self._prompt,
                "rlve_gold_answer": self._gold_answer,
            },
        )
        info = {
            "gold_answer": self._gold_answer,
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
        """Generate a random point set ensuring no three are collinear."""
        N = self._n_points
        if N < 3:
            raise ValueError("n_points must be >= 3")

        points = set()
        lines = set()

        for i in range(N):
            while True:
                x = int(self.np_random.integers(0, N + 1))
                y = int(self.np_random.integers(0, N + 1))

                if (x, y) in points:
                    continue

                # Check collinearity with existing points
                coline = False
                new_lines = set()

                for px, py in points:
                    # Compute normalized line equation ax + by + c = 0
                    if px == x:
                        a, b, c = 1, 0, -x
                    else:
                        a, b = py - y, x - px
                        c = -(a * x + b * y)

                    # Normalize by GCD
                    def gcd(a, b):
                        while b:
                            a, b = b, a % b
                        return a

                    g = gcd(abs(a), gcd(abs(b), abs(c)))
                    if g > 0:
                        a, b, c = a // g, b // g, c // g

                    # Canonical form: ensure a >= 0, or a == 0 and b >= 0
                    if a < 0:
                        a, b, c = -a, -b, -c
                    elif a == 0 and b < 0:
                        b, c = -b, -c

                    if (a, b, c) in lines:
                        coline = True
                        break

                    new_lines.add((a, b, c))

                if coline:
                    continue

                points.add((x, y))
                lines.update(new_lines)
                break

        self._points = list(points)

        # Compute gold answer using dynamic programming
        P = self._points

        def octant(dx, dy):
            """Determine octant for directional sorting."""
            if dx == 0 and dy > 0:
                return 1
            elif dx > 0 and dy > 0:
                return 2
            elif dx > 0 and dy == 0:
                return 3
            elif dx > 0 and dy < 0:
                return 4
            elif dx == 0 and dy < 0:
                return 5
            elif dx < 0 and dy < 0:
                return 6
            elif dx < 0 and dy == 0:
                return 7
            else:  # dx < 0 and dy > 0
                return 8

        # Build all directed edges with precomputed direction info
        edges = []
        for u in range(N):
            xu, yu = P[u]
            for v in range(N):
                if u == v:
                    continue
                xv, yv = P[v]
                dx = xv - xu
                dy = yv - yu
                edges.append((u, v, dx, dy, octant(dx, dy)))

        def cmp_edges(e1, e2):
            """Compare edges by octant then by slope using cross product."""
            if e1[4] != e2[4]:
                return -1 if e1[4] < e2[4] else 1
            cross = e1[3] * e2[2] - e2[3] * e1[2]  # dy1*dx2 - dy2*dx1
            if cross > 0:
                return -1
            elif cross < 0:
                return 1
            else:
                return 0

        edges.sort(key=cmp_to_key(cmp_edges))

        # Extract edge pairs for DP
        EV = [(u, v) for (u, v, _, _, _) in edges]

        # DP to find longest path in directed graph
        ans = 0
        for i in range(N):
            mx = [None] * N
            mx[i] = 0
            for u, v in EV:
                val = mx[u]
                if val is not None:
                    cand = val + 1
                    if mx[v] is None or cand > mx[v]:
                        mx[v] = cand
            if mx[i] is not None and mx[i] > ans:
                ans = mx[i]

        if ans < 3:
            raise RuntimeError("Generated points cannot form a polygon >= 3 vertices")

        self._gold_answer = ans

        # Compute a reference answer by finding one maximal convex polygon
        # Simple approach: use the convex hull which is always a valid convex polygon
        hull = self._compute_convex_hull(self._points)
        # Map hull points back to indices (1-based)
        hull_indices = []
        for hp in hull:
            for idx, p in enumerate(self._points, start=1):
                if p == hp:
                    hull_indices.append(idx)
                    break
        self._oracle_answer = " ".join(map(str, hull_indices))

    def _prompt_generate(self) -> str:
        """Generate problem prompt from generated points."""
        if self._points is None:
            raise RuntimeError("No points generated")

        N = len(self._points)
        points_str = "\n".join(
            f"Point {i}: ({x}, {y})" for i, (x, y) in enumerate(self._points, start=1)
        )

        return self.prompt_template.format(N=N, points=points_str)

    def _process(self, answer: str | None) -> list[int] | None:
        """Parse answer string into list of point indices."""
        if answer is None:
            return None
        answer = answer.strip()
        try:
            answer_array = list(map(int, answer.split()))
            return answer_array
        except ValueError:
            return None

    def _score_answer(self, answer: str) -> float:
        """Score the answer based on correctness and quality."""
        processed_result = self._process(answer)

        if processed_result is None:
            return 0.0

        N = len(self._points)

        # Check valid indices
        if not all(1 <= i <= N for i in processed_result):
            return 0.0

        # Check distinct indices
        if len(processed_result) != len(set(processed_result)):
            return 0.0

        # Check if forms convex polygon
        if not self._can_form_convex_polygon(
            [self._points[i - 1] for i in processed_result]
        ):
            return 0.0

        # Reward based on quality
        answer_count = len(processed_result)
        gold = self._gold_answer

        if answer_count > gold:
            raise AssertionError(f"Answer {answer_count} should be <= gold {gold}")

        # Reward strategy: (answer/gold)^5
        return (answer_count / gold) ** 5

    def _can_form_convex_polygon(self, points: list[tuple[int, int]]) -> bool:
        """Check if points form a convex polygon using convex hull."""
        if len(points) < 3:
            return False

        def cross(o: tuple[int, int], a: tuple[int, int], b: tuple[int, int]) -> int:
            """Compute cross product of vectors OA and OB."""
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        # Sort points and remove duplicates
        pts = sorted(set(points))
        n = len(pts)
        if n < 3:
            return False

        # Build lower hull
        lower = []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)

        # Build upper hull
        upper = []
        for p in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)

        # Combine hulls (remove duplicate endpoints)
        hull = lower[:-1] + upper[:-1]

        # Points form convex polygon iff all are on the hull
        return len(hull) == n

    def render(self) -> Image.Image | list[Image.Image] | None:
        """Render the point set as a 2D scatter plot."""
        if self._points is None:
            raise RuntimeError("No points generated")

        # Extract coordinates
        xs = [p[0] for p in self._points]
        ys = [p[1] for p in self._points]

        # Compute convex hull for visualization
        hull_points = self._compute_convex_hull(self._points)

        # Create figure with clean mathematical style
        fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
        ax.set_aspect("equal", adjustable="box")

        # Set up axes with margins
        if xs and ys:
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            x_range = max(x_max - x_min, 1)
            y_range = max(y_max - y_min, 1)
            margin = max(x_range, y_range) * 0.1

            ax.set_xlim(x_min - margin, x_max + margin)
            ax.set_ylim(y_min - margin, y_max + margin)

        # Draw convex hull polygon with filled area
        if len(hull_points) >= 3:
            hull_coords = np.array(hull_points)
            polygon = MPLPolygon(
                hull_coords,
                facecolor="lightblue",
                edgecolor="steelblue",
                linewidth=2.5,
                alpha=0.4,
            )
            ax.add_patch(polygon)

        # Draw all points
        ax.scatter(
            xs,
            ys,
            c="darkred",
            s=120,
            zorder=3,
            alpha=0.8,
            edgecolors="black",
            linewidths=1.5,
        )

        # Label points with their indices
        for i, (x, y) in enumerate(self._points, start=1):
            ax.annotate(
                str(i),
                (x, y),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=10,
                color="black",
                fontweight="bold",
            )

        # Style the plot
        ax.grid(True, linestyle="--", alpha=0.3, color="gray")
        ax.set_xlabel("X", fontsize=12, fontweight="bold")
        ax.set_ylabel("Y", fontsize=12, fontweight="bold")
        ax.set_title(
            "Point Set - Find Largest Convex Polygon",
            fontsize=14,
            fontweight="bold",
            pad=15,
        )

        # Convert to PIL Image
        fig.tight_layout()
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape((h, w, 4))
        img = Image.fromarray(buf[:, :, :3])  # Drop alpha channel
        plt.close(fig)

        return img

    def _compute_convex_hull(
        self, points: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        """Compute convex hull of points using Andrew's monotone chain algorithm."""
        if len(points) < 3:
            return points

        def cross(o: tuple[int, int], a: tuple[int, int], b: tuple[int, int]) -> int:
            """Compute cross product of vectors OA and OB."""
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        # Sort points
        pts = sorted(set(points))

        # Build lower hull
        lower = []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)

        # Build upper hull
        upper = []
        for p in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)

        # Combine hulls
        hull = lower[:-1] + upper[:-1]

        return hull

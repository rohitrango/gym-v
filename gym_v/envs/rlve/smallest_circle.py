"""RLVE Smallest Circle environment for gym-v (self-contained)."""

from __future__ import annotations

from math import sqrt
from textwrap import dedent
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle as MplCircle

from gym_v import Env, Observation
from gym_v.logger import get_logger

matplotlib.use("Agg")
logger = get_logger()


def distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points."""
    return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def circle_from_two_points(
    p1: tuple[float, float], p2: tuple[float, float]
) -> tuple[tuple[float, float], float]:
    """Find the smallest circle containing two points."""
    center_x = (p1[0] + p2[0]) / 2
    center_y = (p1[1] + p2[1]) / 2
    radius = distance(p1, p2) / 2
    return (center_x, center_y), radius


def circle_from_three_points(
    p1: tuple[float, float], p2: tuple[float, float], p3: tuple[float, float]
) -> tuple[tuple[float, float], float]:
    """Find the circle passing through three points."""
    a1 = p2[0] - p1[0]
    b1 = p2[1] - p1[1]
    c1 = (a1 * a1 + b1 * b1) / 2
    a2 = p3[0] - p1[0]
    b2 = p3[1] - p1[1]
    c2 = (a2 * a2 + b2 * b2) / 2
    d = a1 * b2 - a2 * b1
    center_x = p1[0] + (c1 * b2 - c2 * b1) / d
    center_y = p1[1] + (a1 * c2 - a2 * c1) / d
    radius = distance((center_x, center_y), p1)
    return (center_x, center_y), radius


class RLVESmallestCircleEnv(Env):
    """RLVE Smallest Circle problem as a single-turn environment.

    Task: Given a set of points on a 2D plane, find the smallest circle that
    covers all points. The circle is characterized by its center (x, y) and
    radius r.
    """

    prompt_template = r"""You are given a set of {N} points on a 2D plane.
It is guaranteed that:
(1) all the coordinates are integers;
(2) no two points have the same coordinates;
(3) no three points are on the same line.
Below is the set of points:
{points}

Your task is to find the **smallest circle** covering these points, measured by the radius of the circle.
Your score will be based on the feasibility of your output and the optimality of the radius.
The precision tolerance is 0.001.

**Output Format:** Your output should be three **floats** in a single line, $x$, $y$, and $r$, separated by spaces.
$x$ and $y$ represent the center of the circle, and $r$ represents the radius of the circle."""

    epsilon = 1e-3

    def __init__(
        self,
        n_points: int = 10,
        coord_range: int | None = None,
        wrong_format: float = -1.0,
        invalid_solution: float = 0.0,
        rewarding_strategy: str = "(gold/answer)^beta",
        rewarding_weight: float = 1.0,
        rewarding_beta: float = 10.0,
        num_players: int = 1,
        **kwargs: Any,
    ):
        """Initialize the Smallest Circle environment.

        Args:
            n_points: Number of points to generate.
            coord_range: Maximum coordinate value. If None, uses 2 * n_points.
            wrong_format: Reward for invalid format output.
            invalid_solution: Reward for solution that doesn't cover all points.
            rewarding_strategy: Reward calculation method.
            rewarding_weight: Multiplier for reward calculation.
            rewarding_beta: Power factor for reward calculation.
            num_players: Number of players.
            **kwargs: Additional arguments for Env base class.
        """
        super().__init__(**kwargs)
        self._n_points = n_points
        self._coord_range = coord_range if coord_range is not None else 2 * n_points
        self._wrong_format = wrong_format
        self._invalid_solution = invalid_solution
        self._rewarding_strategy = rewarding_strategy
        self._rewarding_weight = rewarding_weight
        self._rewarding_beta = rewarding_beta
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._points: list[tuple[int, int]] | None = None
        self._optimal_center: tuple[float, float] | None = None
        self._optimal_radius: float | None = None
        self._reference_answer: str | None = None
        self._prompt: str | None = None
        self._last_image: Any = None

    @property
    def description(self) -> str:
        n = self._n_points if self._n_points else "N"
        return dedent(
            f"""
            Smallest Circle problem:
            Given {n} points on a 2D plane, find the smallest circle (minimum
            enclosing circle) that covers all points.

            Constraints:
            1) All coordinates are integers
            2) No two points share the same coordinates
            3) No three points are collinear

            Your task:
            Find the center (x, y) and radius r of the smallest circle that
            contains all given points. The circle is considered to contain a
            point if the distance from the center to the point is at most r.

            Scoring:
            - Score depends on both feasibility (all points covered) and
              optimality (how close to minimum radius)
            - Precision tolerance: 0.001

            Output format:
            Three space-separated floats: x y r
            where (x, y) is the center and r is the radius.
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
                "rlve_reference_answer": self._reference_answer,
            },
        )
        info = {
            "reference_answer": self._reference_answer,
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
                "rlve_reference_answer": self._reference_answer,
            },
        )
        info = {
            "reference_answer": self._reference_answer,
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
        """Generate a set of points and compute the smallest enclosing circle."""
        N = self._n_points
        if N < 2:
            raise ValueError("n_points must be >= 2")

        points_set: set[tuple[int, int]] = set()
        lines: set[tuple[int, int, int]] = set()

        for _ in range(N):
            while True:
                x = int(self.np_random.integers(0, self._coord_range + 1))
                y = int(self.np_random.integers(0, self._coord_range + 1))
                if (x, y) in points_set:
                    continue

                coline = False
                new_lines: set[tuple[int, int, int]] = set()
                for px, py in points_set:
                    if px == x:
                        a, b, c = 1, 0, -x
                    else:
                        a, b = py - y, x - px
                        c = -(a * x + b * y)

                    def gcd(a: int, b: int) -> int:
                        while b:
                            a, b = b, a % b
                        return a

                    g = gcd(abs(a), gcd(abs(b), abs(c)))
                    if g > 0:
                        a, b, c = a // g, b // g, c // g

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

                points_set.add((x, y))
                lines.update(new_lines)
                break

        self._points = list(points_set)

        # Use Welzl's randomized algorithm to find the smallest circle
        points_copy = self._points.copy()
        self.np_random.shuffle(points_copy)

        c = points_copy[0]
        r = 0.0
        for i in range(1, N):
            if distance(points_copy[i], c) < r + self.epsilon:
                continue

            c = points_copy[i]
            r = 0.0
            for j in range(i):
                if distance(points_copy[j], c) < r + self.epsilon:
                    continue

                c, r = circle_from_two_points(points_copy[i], points_copy[j])
                for k in range(j):
                    if distance(points_copy[k], c) < r + self.epsilon:
                        continue

                    c, r = circle_from_three_points(
                        points_copy[i], points_copy[j], points_copy[k]
                    )

        self._optimal_center = c
        self._optimal_radius = r
        self._reference_answer = f"{c[0]} {c[1]} {r}"

    def _prompt_generate(self) -> str:
        """Generate the prompt text."""
        if self._points is None:
            raise RuntimeError("No points generated")
        return self.prompt_template.format(
            N=len(self._points),
            points="\n".join(f"({x}, {y})" for x, y in self._points),
        )

    def _process(self, answer: str | None) -> tuple[float, float, float] | None:
        """Parse the answer string into (x, y, r) tuple."""
        if answer is None:
            return None
        answer = answer.strip()
        try:
            parts = answer.split()
            if len(parts) != 3:
                return None
            x, y, r = map(float, parts)
            return (x, y, r)
        except ValueError:
            return None

    def _score_answer(self, answer: str) -> float:
        """Score the answer based on feasibility and optimality."""
        processed_result = self._process(answer)
        if processed_result is None:
            return self._wrong_format

        x, y, r = processed_result
        if r <= 0:
            return self._wrong_format

        # Check if all points are covered
        if any(distance((x, y), p) > r + self.epsilon for p in self._points):
            return self._invalid_solution

        opt_r = self._optimal_radius
        # The radius should be at least as large as optimal (within tolerance)
        if r < opt_r - 2 * self.epsilon:
            return self._invalid_solution

        if self._rewarding_strategy == "(gold/answer)^beta":
            return self._rewarding_weight * min(
                ((opt_r / r) ** self._rewarding_beta), 1.0
            )
        elif self._rewarding_strategy == "gold=answer":
            return self._rewarding_weight * (abs(r - opt_r) < self.epsilon)
        else:
            raise NotImplementedError(
                f"Unknown rewarding strategy: {self._rewarding_strategy}"
            )

    def render(self) -> Any:
        """Render the current state as a matplotlib figure showing points and circle.

        Returns:
            A matplotlib figure converted to PIL Image showing:
            - All points as a scatter plot
            - The optimal enclosing circle
            - The center of the circle
            - Coordinate axes with grid
        """
        if self._points is None or self._optimal_center is None:
            raise RuntimeError("No points or circle generated")

        fig, ax = plt.subplots(figsize=(8, 8), dpi=100)

        # Extract coordinates
        xs = [p[0] for p in self._points]
        ys = [p[1] for p in self._points]

        # Plot points
        ax.scatter(
            xs, ys, color="#2E86DE", s=100, zorder=3, alpha=0.8, edgecolors="black"
        )

        # Draw the optimal circle
        circle = MplCircle(
            self._optimal_center,
            self._optimal_radius,
            color="#EE5A6F",
            fill=False,
            linewidth=2.5,
            linestyle="-",
            label="Minimum Enclosing Circle",
            zorder=2,
        )
        ax.add_patch(circle)

        # Mark the center
        ax.scatter(
            [self._optimal_center[0]],
            [self._optimal_center[1]],
            color="#EE5A6F",
            s=150,
            marker="x",
            linewidths=3,
            zorder=4,
            label="Center",
        )

        # Set up axes
        margin = max(self._optimal_radius * 0.2, 1.0)
        x_min = self._optimal_center[0] - self._optimal_radius - margin
        x_max = self._optimal_center[0] + self._optimal_radius + margin
        y_min = self._optimal_center[1] - self._optimal_radius - margin
        y_max = self._optimal_center[1] + self._optimal_radius + margin

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")

        # Grid and styling
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
        ax.axhline(y=0, color="k", linewidth=0.8, alpha=0.3)
        ax.axvline(x=0, color="k", linewidth=0.8, alpha=0.3)
        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("y", fontsize=12)
        ax.set_title(
            f"Smallest Circle Problem ({len(self._points)} points)", fontsize=14
        )
        ax.legend(loc="upper right", fontsize=10)

        # Convert to PIL Image
        fig.canvas.draw()
        img = fig.canvas.buffer_rgba()
        width, height = fig.canvas.get_width_height()
        from PIL import Image

        pil_img = Image.frombytes("RGBA", (width, height), img).convert("RGB")
        plt.close(fig)

        return pil_img

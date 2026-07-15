"""Convex hull environment for gym-v (self-contained)."""

from __future__ import annotations

from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw

from gym_v import Env, Observation
from gym_v.logger import get_logger

logger = get_logger()


class ConvexHullEnv(Env):
    # Meta: source=RLVE, category=geometry, turn=single
    """RLVE convex hull as a single-turn environment."""

    prompt_template = r"""You are given a set of {N} points on a 2D plane labeled from 0 to {N_minus_1}.
It is guaranteed that:
(1) all the coordinates are integers;
(2) no two points have the same coordinates;
(3) no three points are on the same line.

The set of points is given in the image.

Your task is to find the **convex hull** of these points, which is the smallest convex polygon that contains all the points.

**Output Format:** Your output should be one single **integer**, representing the value of 2 times the area of the convex hull (which can be proven to be an integer)."""

    def __init__(
        self,
        N: int = 10,
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._N = N
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._points: list[tuple[int, int]] | None = None
        self._prompt: str | None = None
        self._oracle_answer: int | None = None
        self._convex_hull_indices: list[int] | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        if self._points:
            num_points = len(self._points)
            point_hint = f"{num_points} points"
        else:
            point_hint = "N points"
        return dedent(
            f"""
            Your task is to find the convex hull of a set of points, which is the smallest convex polygon that contains all the points.

            Given {point_hint} on a 2D plane.

            Guarantees:
            - All coordinates are integers
            - No two points have the same coordinates
            - No three points are collinear

            In the image:
            - Red points indicate convex hull vertices
            - Gray points indicate interior points
            - Light blue filled area with blue outline shows the convex hull region
            - Gray lines show the coordinate axes (if visible within the plot range)

            Output Format: Your output should be one single integer, representing the value of 2 times the area of the convex hull (which can be proven to be an integer).
            """
        ).strip()

    def _get_state_text(self) -> str:
        """Return text representation of the points."""
        if self._points is None:
            return ""
        return "\n".join(f"({x}, {y})" for x, y in self._points)

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
        """Generate N points with no collinear triples."""
        N = self._N
        assert N >= 3, "N should be greater than or equal to 3"

        points = set()
        lines = set()

        for _i in range(N):
            while True:
                x = int(self.np_random.integers(0, N + 1))
                y = int(self.np_random.integers(0, N + 1))
                if (x, y) in points:
                    continue

                coline = False
                new_lines = set()
                for px, py in points:
                    if px == x:
                        a, b, c = 1, 0, -x
                    else:
                        a, b = py - y, x - px
                        c = -(a * x + b * y)

                    def gcd(a, b):
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

                points.add((x, y))
                lines.update(new_lines)
                break

        self._points = list(points)

        # Calculate convex hull using Andrew's algorithm
        sorted_point_indices = sorted(
            range(len(self._points)),
            key=lambda i: (self._points[i][0], self._points[i][1]),
        )

        def cross_product(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lower = []
        for i in sorted_point_indices:
            while (
                len(lower) >= 2
                and cross_product(
                    self._points[lower[-2]], self._points[lower[-1]], self._points[i]
                )
                <= 0
            ):
                lower.pop()
            lower.append(i)

        upper = []
        for i in reversed(sorted_point_indices):
            while (
                len(upper) >= 2
                and cross_product(
                    self._points[upper[-2]], self._points[upper[-1]], self._points[i]
                )
                <= 0
            ):
                upper.pop()
            upper.append(i)

        self._convex_hull_indices = lower[:-1] + upper[:-1]

        # Calculate 2 * area using shoelace formula
        area = 0
        for i in range(len(self._convex_hull_indices)):
            j = (i + 1) % len(self._convex_hull_indices)
            x1, y1 = self._points[self._convex_hull_indices[i]]
            x2, y2 = self._points[self._convex_hull_indices[j]]
            area += x1 * y2 - x2 * y1

        self._oracle_answer = abs(area)

    def _prompt_generate(self) -> str:
        if self._points is None:
            raise RuntimeError("No points generated")
        N = len(self._points)
        return self.prompt_template.format(
            N=N,
            N_minus_1=N - 1,
            points="\n".join(f"({x}, {y})" for x, y in self._points),
        )

    def _process(self, answer: str | None) -> int | None:
        if answer is None:
            return None
        answer = answer.strip()
        try:
            int_answer = int(answer)
            return int_answer
        except ValueError:
            return None

    def _score_answer(self, answer: str) -> float:
        """Score the answer using (min/max)^5 strategy."""
        processed_result = self._process(answer)
        if processed_result is None:
            return 0.0
        if processed_result <= 0:
            return 0.0
        a, b = self._oracle_answer, processed_result
        return (min(a, b) / max(a, b)) ** 5

    def render(self) -> Image.Image | list[Image.Image] | None:
        """Render the convex hull visualization."""
        if self._points is None or self._convex_hull_indices is None:
            raise RuntimeError("No points or convex hull generated")

        # Calculate bounds
        min_x = min(x for x, y in self._points)
        max_x = max(x for x, y in self._points)
        min_y = min(y for x, y in self._points)
        max_y = max(y for x, y in self._points)

        # Add padding
        padding = 0.1 * max(max_x - min_x, max_y - min_y, 1)
        min_x -= padding
        max_x += padding
        min_y -= padding
        max_y += padding

        # Image dimensions
        img_width = 800
        img_height = 800
        margin = 60
        plot_width = img_width - 2 * margin
        plot_height = img_height - 2 * margin

        img = Image.new("RGB", (img_width, img_height), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Coordinate transformation
        def to_pixel(x, y):
            px = margin + (x - min_x) / (max_x - min_x) * plot_width
            py = img_height - margin - (y - min_y) / (max_y - min_y) * plot_height
            return px, py

        # Draw axes
        origin_x, origin_y = to_pixel(0, 0)

        # X-axis
        if min_y <= 0 <= max_y:
            draw.line(
                [(margin, origin_y), (img_width - margin, origin_y)],
                fill=(200, 200, 200),
                width=1,
            )

        # Y-axis
        if min_x <= 0 <= max_x:
            draw.line(
                [(origin_x, margin), (origin_x, img_height - margin)],
                fill=(200, 200, 200),
                width=1,
            )

        # Draw convex hull polygon
        hull_points = [to_pixel(*self._points[i]) for i in self._convex_hull_indices]
        if len(hull_points) >= 3:
            # Fill polygon with light blue
            draw.polygon(hull_points, fill=(220, 235, 255), outline=None)
            # Draw outline
            for i in range(len(hull_points)):
                j = (i + 1) % len(hull_points)
                draw.line(
                    [hull_points[i], hull_points[j]], fill=(70, 130, 220), width=3
                )

        # Draw all points
        point_radius = 6
        for i, (x, y) in enumerate(self._points):
            px, py = to_pixel(x, y)

            if i in self._convex_hull_indices:
                # Hull vertices - larger and highlighted
                draw.ellipse(
                    [
                        px - point_radius - 2,
                        py - point_radius - 2,
                        px + point_radius + 2,
                        py + point_radius + 2,
                    ],
                    fill=(220, 50, 50),
                    outline=(150, 30, 30),
                    width=2,
                )
            else:
                # Interior points - smaller and gray
                draw.ellipse(
                    [
                        px - point_radius,
                        py - point_radius,
                        px + point_radius,
                        py + point_radius,
                    ],
                    fill=(100, 100, 100),
                    outline=(50, 50, 50),
                    width=1,
                )

        # Draw border
        draw.rectangle(
            [margin, margin, img_width - margin, img_height - margin],
            outline=(100, 100, 100),
            width=2,
        )

        return img

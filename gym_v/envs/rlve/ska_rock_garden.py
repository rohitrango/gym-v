"""Ska Rock Garden environment for gym-v (self-contained)."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation
from gym_v.logger import get_logger

logger = get_logger()


class RLVESkaRockGardenEnv(Env):
    """RLVE Ska Rock Garden as a single-turn environment.

    Given N points in a 2D plane, each with coordinates (X[i], Y[i]) and a swap
    cost M[i], determine which points to swap to minimize the perimeter of the
    smallest axis-aligned rectangle enclosing all points. If multiple solutions
    have the same minimum perimeter, choose the one with minimum total swap cost.
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""There are {N} points in a 2D plane, where the i-th point is (X[i], Y[i]) for 0 ≤ i < {N}. Each point has a cost M[i] to swap its coordinates (i.e., swapping (x, y) becomes (y, x)). Your goal is as follows:
- First, minimize the total perimeter of the smallest axis-aligned rectangle that can enclose all points after some of them are optionally swapped. The perimeter is obviously 2 × ((max_x - min_x) + (max_y - min_y)), where max_x and min_x are the maximum and minimum x-coordinates after your swaps (similarly for y).
- If multiple swap strategies result in the same minimum perimeter, choose the one with the smallest total swap cost (i.e., sum of M[i] for all swapped points).

X, Y, and M are given as follows:
{X_Y_M}

**Output Format:** Output a single line of {N} characters (no spaces or any other kinds of separators). The i-th character should be:
- `'0'` if you do **NOT** swap point i,
- `'1'` if you **do** swap point i."""

    def __init__(
        self,
        max_n: int = 10,
        cell_px: int = 60,
        padding: int = 40,
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._max_n = max_n
        self._cell_px = cell_px
        self._padding = padding
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        if max_n < 3:
            raise ValueError("max_n should be >= 3")

        self._n: int | None = None
        self._x: list[int] | None = None
        self._y: list[int] | None = None
        self._m: list[int] | None = None
        self._prompt: str | None = None
        self._oracle_answer: str | None = None
        self._gold_answer_perimeter: int | None = None
        self._gold_answer_cost: int | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        if self._n:
            size_hint = f"{self._n} points"
        else:
            size_hint = "N points"

        return dedent(
            f"""
            Ska Rock Garden Problem:

            Given {size_hint} in a 2D plane, each with coordinates (X[i], Y[i]) and swap cost M[i],
            determine which points to coordinate-swap to minimize the perimeter of the smallest
            axis-aligned rectangle that encloses all points.

            Rules:
            1) Each point i has coordinates (X[i], Y[i]) and swap cost M[i]
            2) Swapping point i changes it from (X[i], Y[i]) to (Y[i], X[i])
            3) Goal: minimize perimeter = 2 × ((max_x - min_x) + (max_y - min_y))
            4) If tied on perimeter, minimize total swap cost

            In the visualization:
            - Blue circles show points that should NOT be swapped (output '0')
            - Red circles show points that SHOULD be swapped (output '1')
            - Point labels show the coordinates (X[i], Y[i]) and swap cost M[i]
            - The green dashed rectangle shows the optimal bounding box after swaps
            - Grid lines help visualize spatial relationships

            Output format: A string of {self._n if self._n else "N"} characters, each '0' or '1',
            indicating whether to swap each point (e.g., "01001").
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
        """Generate a ska rock garden problem instance.

        Ports generation logic from RLVE using self.np_random.
        """
        N = int(self.np_random.integers(3, self._max_n + 1))

        X = [int(self.np_random.integers(0, 2 * N + 1)) for _ in range(N)]
        Y = [int(self.np_random.integers(0, 2 * N + 1)) for _ in range(N)]
        M = [int(self.np_random.integers(1, N + 1)) for _ in range(N)]

        INF = (max(max(X), max(Y)) + 1) * 2
        lx = INF
        rx = -INF
        ly = INF
        ry = -INF

        # Determine the minimal enclosing rectangle assuming no more swaps
        for i in range(N):
            x, y = X[i], Y[i]
            if x <= y:
                if x < lx:
                    lx = x
                if x > rx:
                    rx = x
                if y < ly:
                    ly = y
                if y > ry:
                    ry = y
            else:
                # these points are effectively swapped
                if y < lx:
                    lx = y
                if y > rx:
                    rx = y
                if x < ly:
                    ly = x
                if x > ry:
                    ry = x

        # The minimal fence length (perimeter of axis-aligned rectangle)
        fence_length = 2 * ((rx - lx) + (ry - ly))

        best_weight = sum(M)  # Start with the worst case: swap all points
        best_assign = None

        def try_bounds(
            lx0: int, rx0: int, ly0: int, ry0: int
        ) -> tuple[int | None, list[int] | None]:
            """Try using bounds [lx0,rx0] × [ly0,ry0], returning (weight, assignment)
            or (None, None) if impossible."""
            total = 0
            assign = [0] * N
            for i in range(N):
                x, y = X[i], Y[i]
                if lx0 <= x <= rx0 and ly0 <= y <= ry0:
                    # no swap needed
                    assign[i] = 0
                elif lx0 <= y <= rx0 and ly0 <= x <= ry0:
                    # swap needed
                    assign[i] = 1
                    total += M[i]
                else:
                    # this point can't fit even if swapped
                    return None, None
            return total, assign

        # Try the 4 possible ways of interpreting the bounding box
        for a, b, c, d in (
            (lx, rx, ly, ry),
            (lx, ry, ly, rx),
            (ly, rx, lx, ry),
            (ly, ry, lx, rx),
        ):
            w, assn = try_bounds(a, b, c, d)
            if w is not None and w < best_weight:
                best_weight = w
                best_assign = assn

        # Output results
        self._n = N
        self._x = X
        self._y = Y
        self._m = M
        self._gold_answer_perimeter = fence_length
        self._gold_answer_cost = best_weight
        self._oracle_answer = "".join(map(str, best_assign))

    def _prompt_generate(self) -> str:
        """Generate the prompt text for the problem."""
        if self._n is None:
            raise RuntimeError("No problem generated")
        return self.prompt_template.format(
            N=self._n,
            X_Y_M="\n".join(
                f"X[{i}]={Xi} Y[{i}]={Yi} M[{i}]={Mi}"
                for i, (Xi, Yi, Mi) in enumerate(
                    zip(self._x, self._y, self._m, strict=False)
                )
            ),
        )

    def _process(self, answer: str | None) -> str | None:
        """Process the answer string."""
        if answer is not None:
            answer = answer.strip()
            return answer
        else:
            return None

    def _score_answer(self, answer: str) -> float:
        """Score the answer based on perimeter and cost optimization.

        Returns:
            -1.0: wrong format
            0.0 to 1.0: weighted score based on perimeter and cost optimization
        """
        processed_result = self._process(answer)
        if processed_result is not None:
            if len(processed_result) != self._n:
                return 0.0
            if not all(c in "01" for c in processed_result):
                return 0.0
            X, Y = self._x.copy(), self._y.copy()
            answer_cost = 0
            gold_cost = self._gold_answer_cost
            for i, swap in enumerate(processed_result):
                if swap == "1":
                    X[i], Y[i] = Y[i], X[i]
                    answer_cost += self._m[i]
                elif swap == "0":
                    continue
                else:
                    assert False

            answer_perimeter = 2 * ((max(X) - min(X)) + (max(Y) - min(Y)))
            gold_perimeter = self._gold_answer_perimeter

            reward = 0.0

            # Perimeter reward (weight: 0.5, beta: 5.0)
            assert (
                gold_perimeter <= answer_perimeter
            ), "answer_perimeter should be greater than or equal to gold_perimeter"
            if answer_perimeter == 0:
                assert (
                    gold_perimeter == 0
                ), "If answer_perimeter is zero, gold_perimeter should also be zero"
                reward += 0.5 * 1.0
            else:
                reward += 0.5 * ((gold_perimeter / answer_perimeter) ** 5.0)

            # Cost reward (weight: 0.5, beta: 5.0) - only if perimeters match
            if gold_perimeter == answer_perimeter:
                assert (
                    gold_cost <= answer_cost
                ), "answer_cost should be greater than or equal to gold_cost"
                if answer_cost == 0:
                    assert (
                        gold_cost == 0
                    ), "If answer_cost is zero, gold_cost should also be zero"
                    reward += 0.5 * 1.0
                else:
                    reward += 0.5 * ((gold_cost / answer_cost) ** 5.0)

            return reward
        else:
            return 0.0

    def render(self) -> Image.Image | list[Image.Image] | None:
        """Render the rock garden puzzle as an image.

        Shows:
        - 2D scatter plot of points with their coordinates
        - Blue circles for points not to swap
        - Red circles for points to swap
        - Point labels with coordinates and swap cost
        - Optimal bounding rectangle
        """
        if self._n is None or self._x is None:
            raise RuntimeError("No problem generated")

        padding = self._padding
        cell_px = self._cell_px

        # Calculate bounds for the plot
        min_x = min(self._x)
        max_x = max(self._x)
        min_y = min(self._y)
        max_y = max(self._y)

        # Add some margin
        margin = 1
        plot_min_x = min_x - margin
        plot_max_x = max_x + margin
        plot_min_y = min_y - margin
        plot_max_y = max(self._y) + margin

        # Calculate canvas dimensions
        x_range = plot_max_x - plot_min_x
        y_range = plot_max_y - plot_min_y

        # Use cell_px as scale factor
        plot_width = int(x_range * cell_px)
        plot_height = int(y_range * cell_px)

        # Ensure minimum size
        plot_width = max(plot_width, 400)
        plot_height = max(plot_height, 400)

        title_height = 80
        legend_height = 120

        width = padding * 2 + plot_width
        height = padding * 3 + title_height + plot_height + legend_height

        img = Image.new("RGB", (width, height), (250, 250, 250))
        draw = ImageDraw.Draw(img)

        # Load font
        font_path = None
        font_path_candidate = self.assets_dir / "DejaVuSans.ttf"
        if font_path_candidate.exists():
            font_path = str(font_path_candidate)

        if font_path:
            font_title = ImageFont.truetype(font_path, 24)
            font_label = ImageFont.truetype(font_path, 12)
            font_legend = ImageFont.truetype(font_path, 14)
        else:
            font_title = ImageFont.load_default()
            font_label = ImageFont.load_default()
            font_legend = ImageFont.load_default()

        # Draw title
        title = f"Ska Rock Garden - {self._n} Points"
        title_bbox = draw.textbbox((0, 0), title, font=font_title)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = (width - title_width) // 2
        draw.text((title_x, 20), title, fill=(30, 30, 30), font=font_title)

        # Plot area offset
        plot_x = padding
        plot_y = padding + title_height

        # Helper function to convert coordinates to pixel positions
        def coord_to_pixel(x: float, y: float) -> tuple[int, int]:
            px = plot_x + int((x - plot_min_x) / x_range * plot_width)
            py = plot_y + plot_height - int((y - plot_min_y) / y_range * plot_height)
            return px, py

        # Draw grid lines
        grid_color = (220, 220, 220)
        num_grid_lines = 10
        for i in range(num_grid_lines + 1):
            # Vertical grid lines
            x = plot_x + int(i / num_grid_lines * plot_width)
            draw.line(
                [(x, plot_y), (x, plot_y + plot_height)], fill=grid_color, width=1
            )

            # Horizontal grid lines
            y = plot_y + int(i / num_grid_lines * plot_height)
            draw.line([(plot_x, y), (plot_x + plot_width, y)], fill=grid_color, width=1)

        # Draw plot border
        draw.rectangle(
            [plot_x, plot_y, plot_x + plot_width, plot_y + plot_height],
            outline=(80, 80, 80),
            width=2,
        )

        # Calculate optimal bounding box after swaps
        X_after_swap = self._x.copy()
        Y_after_swap = self._y.copy()
        for i, swap in enumerate(self._oracle_answer):
            if swap == "1":
                X_after_swap[i], Y_after_swap[i] = Y_after_swap[i], X_after_swap[i]

        optimal_min_x = min(X_after_swap)
        optimal_max_x = max(X_after_swap)
        optimal_min_y = min(Y_after_swap)
        optimal_max_y = max(Y_after_swap)

        # Draw optimal bounding rectangle
        rect_p1 = coord_to_pixel(optimal_min_x, optimal_min_y)
        rect_p2 = coord_to_pixel(optimal_max_x, optimal_max_y)
        draw.rectangle(
            [rect_p1[0], rect_p2[1], rect_p2[0], rect_p1[1]],
            outline=(50, 200, 50),
            width=3,
        )

        # Draw dashed effect for optimal rectangle
        dash_length = 10
        gap_length = 5
        # Top edge
        x1, y1 = rect_p1[0], rect_p2[1]
        x2, y2 = rect_p2[0], rect_p2[1]
        current_x = x1
        while current_x < x2:
            next_x = min(current_x + dash_length, x2)
            draw.line([(current_x, y1), (next_x, y1)], fill=(50, 200, 50), width=2)
            current_x = next_x + gap_length

        # Bottom edge
        y1 = rect_p1[1]
        current_x = x1
        while current_x < x2:
            next_x = min(current_x + dash_length, x2)
            draw.line([(current_x, y1), (next_x, y1)], fill=(50, 200, 50), width=2)
            current_x = next_x + gap_length

        # Left edge
        current_y = min(rect_p1[1], rect_p2[1])
        end_y = max(rect_p1[1], rect_p2[1])
        while current_y < end_y:
            next_y = min(current_y + dash_length, end_y)
            draw.line([(x1, current_y), (x1, next_y)], fill=(50, 200, 50), width=2)
            current_y = next_y + gap_length

        # Right edge
        x1 = rect_p2[0]
        current_y = min(rect_p1[1], rect_p2[1])
        while current_y < end_y:
            next_y = min(current_y + dash_length, end_y)
            draw.line([(x1, current_y), (x1, next_y)], fill=(50, 200, 50), width=2)
            current_y = next_y + gap_length

        # Draw points
        point_radius = 8
        for i in range(self._n):
            px, py = coord_to_pixel(self._x[i], self._y[i])

            # Color based on whether to swap
            if self._oracle_answer[i] == "0":
                point_color = (70, 130, 220)  # Blue - don't swap
                text_color = (70, 130, 220)
            else:
                point_color = (220, 70, 70)  # Red - swap
                text_color = (220, 70, 70)

            # Draw circle
            draw.ellipse(
                [
                    px - point_radius,
                    py - point_radius,
                    px + point_radius,
                    py + point_radius,
                ],
                fill=point_color,
                outline=(30, 30, 30),
                width=2,
            )

            # Draw label with coordinates and cost
            label = f"({self._x[i]},{self._y[i]}) M={self._m[i]}"
            label_bbox = draw.textbbox((0, 0), label, font=font_label)
            label_width = label_bbox[2] - label_bbox[0]
            label_height = label_bbox[3] - label_bbox[1]

            # Position label to avoid overlapping the point
            label_x = px - label_width // 2
            label_y = py - point_radius - label_height - 4

            # Draw label background
            draw.rectangle(
                [
                    label_x - 2,
                    label_y - 2,
                    label_x + label_width + 2,
                    label_y + label_height + 2,
                ],
                fill=(255, 255, 255),
                outline=None,
            )
            draw.text((label_x, label_y), label, fill=text_color, font=font_label)

        # Draw legend
        legend_y = plot_y + plot_height + padding

        # Title
        legend_title = "Legend:"
        draw.text(
            (padding, legend_y), legend_title, fill=(30, 30, 30), font=font_legend
        )
        legend_y += 25

        # Blue circle legend
        legend_point_radius = 6
        legend_x = padding + 20
        draw.ellipse(
            [
                legend_x - legend_point_radius,
                legend_y - legend_point_radius,
                legend_x + legend_point_radius,
                legend_y + legend_point_radius,
            ],
            fill=(70, 130, 220),
            outline=(30, 30, 30),
            width=2,
        )
        draw.text(
            (legend_x + 15, legend_y - 8),
            "Points NOT to swap (output '0')",
            fill=(30, 30, 30),
            font=font_label,
        )
        legend_y += 25

        # Red circle legend
        draw.ellipse(
            [
                legend_x - legend_point_radius,
                legend_y - legend_point_radius,
                legend_x + legend_point_radius,
                legend_y + legend_point_radius,
            ],
            fill=(220, 70, 70),
            outline=(30, 30, 30),
            width=2,
        )
        draw.text(
            (legend_x + 15, legend_y - 8),
            "Points TO swap (output '1')",
            fill=(30, 30, 30),
            font=font_label,
        )
        legend_y += 25

        # Rectangle legend
        draw.rectangle(
            [legend_x - 8, legend_y - 8, legend_x + 8, legend_y + 8],
            outline=(50, 200, 50),
            width=2,
        )
        draw.text(
            (legend_x + 15, legend_y - 8),
            f"Optimal bounding rectangle (perimeter={self._gold_answer_perimeter}, cost={self._gold_answer_cost})",
            fill=(30, 30, 30),
            font=font_label,
        )

        return img

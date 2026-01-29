"""Visible Line environment for gym-v (self-contained)."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation
from gym_v.logger import get_logger

logger = get_logger()


class RLVEVisibleLineEnv(Env):
    """RLVE Visible Line as a single-turn environment.

    Given N lines on the 2D plane (each represented as y = Ax + B), determine
    which lines are visible when viewed from y = +∞ (looking down). A line is
    visible if there exists at least one x-coordinate where this line has the
    maximum y-value among all lines.
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""You are given {N} lines on the 2D plane:
{lines}

We say a line is **visible** if any portion of it can be seen when viewed from y = +∞ (i.e., looking vertically downward). That is, a line is visible if there exists at least one x-coordinate such that this line lies on top (i.e., has the maximum y-value) at that x among all lines.

**Output Format:** A single line containing the indices of all visible lines, in any order, separated by spaces."""

    def __init__(
        self,
        max_n: int = 10,
        canvas_width: int = 800,
        canvas_height: int = 600,
        padding: int = 60,
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._max_n = max_n
        self._canvas_width = canvas_width
        self._canvas_height = canvas_height
        self._padding = padding
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._n: int | None = None
        self._lines: list[tuple[int, int]] | None = None
        self._prompt: str | None = None
        self._gold_answer: list[int] | None = None
        self._oracle_answer: str | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        if self._n:
            size_hint = f"{self._n} lines"
        else:
            size_hint = "N lines"

        return dedent(
            f"""
            Visible Line Problem:

            Given {size_hint} on a 2D plane, each represented as y = Ax + B,
            determine which lines are visible when viewed from y = +∞ (looking down).

            A line is visible if there exists at least one x-coordinate where this line
            has the maximum y-value among all lines at that x-coordinate.

            In the visualization:
            - Each line is drawn with a distinct color
            - Line indices are labeled on the lines
            - Visible lines are drawn with thicker, solid strokes
            - Non-visible lines are drawn with thinner, dashed strokes
            - The upper envelope (convex hull) is highlighted
            - A legend shows line equations and their visibility status

            The problem asks you to identify which lines form the upper envelope when
            viewed from above. Lines that are completely covered by other lines at all
            x-coordinates are not visible.

            Output format: A single line containing the indices of all visible lines,
            separated by spaces (order does not matter).
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
            text=state_text,
            metadata={
                "text_prompt": f"{state_text}\n\n{self.description}",
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
                "text_prompt": f"{state_text}\n\n{self.description}",
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
        """Generate a visible line problem instance.

        Ports generation logic from RLVE using self.np_random.
        """
        N = self._max_n
        if N < 3:
            raise ValueError("max_n must be >= 3")

        N = int(self.np_random.integers(3, N + 1))

        lines = set()
        while len(lines) < N:
            Ai = int(self.np_random.integers(-N, N + 1))
            Bi = int(self.np_random.integers(-N, N + 1))
            if (Ai, Bi) not in lines:
                lines.add((Ai, Bi))
        self._lines = list(lines)
        self.np_random.shuffle(self._lines)

        P = []
        for i, (A, B) in enumerate(self._lines):
            P.append((A, B, i))

        # Sort by slope A ascending, and for ties by intercept B descending
        P.sort(key=lambda x: (x[0], -x[1]))

        # Build the "upper hull" of visible lines
        BIN = []
        prevA = None
        for A, B, idx in P:
            # Skip duplicate slopes (only keep the one with highest intercept)
            if A == prevA:
                continue
            prevA = A

            # While the last segment and the new point make a non-left turn,
            # pop the last line (it's covered)
            while len(BIN) >= 2:
                A1, B1, _ = BIN[-2]
                A2, B2, _ = BIN[-1]
                # Cross product of vectors (A2-A1, B2-B1) and (A-A2, B-B2)
                if (A2 - A1) * (B - B2) - (B2 - B1) * (A - A2) >= 0:
                    BIN.pop()
                else:
                    break

            BIN.append((A, B, idx))

        # Sort visible lines by original input order (their ids)
        BIN.sort(key=lambda x: x[2])

        self._n = N
        self._gold_answer = [idx for A, B, idx in BIN]
        self._oracle_answer = " ".join(map(str, self._gold_answer))

    def _prompt_generate(self) -> str:
        """Generate the prompt text for the problem."""
        if self._n is None:
            raise RuntimeError("No problem generated")
        return self.prompt_template.format(
            N=self._n,
            lines="\n".join(
                f"Line {i}: y = {A}x + {B}" for i, (A, B) in enumerate(self._lines)
            ),
        )

    def _process(self, answer: str | None) -> set[int] | None:
        """Process the answer string into a set of integers."""
        if answer is not None:
            answer = answer.strip()
            if not answer:
                return None
            try:
                return set(map(int, answer.split()))
            except ValueError:
                return None
        else:
            return None

    def _score_answer(self, answer: str) -> float:
        """Score the answer using (intersection/union)^beta strategy.

        Returns:
            -1.0: wrong format (not valid integers or out of range)
             0.0 to 1.0: (intersection/union)^5
        """
        processed_result = self._process(answer)
        if processed_result is not None:
            answer_set = processed_result
            if not all(0 <= x < self._n for x in answer_set):
                return 0.0
            gold = set(self._gold_answer)

            intersection = len(answer_set & gold)
            union = len(answer_set | gold)
            if union == 0:
                return 1.0 if intersection == 0 else 0.0
            return (intersection / union) ** 5.0
        else:
            return 0.0

    def render(self) -> Image.Image | list[Image.Image] | None:
        """Render the visible line problem as a beautiful geometric visualization.

        Shows:
        - All lines plotted on a coordinate system
        - Visible lines highlighted with solid, thick strokes
        - Non-visible lines shown with dashed, thin strokes
        - Color-coded lines with labels
        - Legend showing line equations and visibility status
        - Visual indication of the upper envelope
        """
        if self._n is None or self._lines is None:
            raise RuntimeError("No problem generated")

        width = self._canvas_width
        height = self._canvas_height
        padding = self._padding

        img = Image.new("RGB", (width, height), (250, 250, 250))
        draw = ImageDraw.Draw(img)

        # Load font
        font_path = None
        font_path_candidate = self.assets_dir / "DejaVuSans.ttf"
        if font_path_candidate.exists():
            font_path = str(font_path_candidate)

        if font_path:
            font_title = ImageFont.truetype(font_path, 24)
            font_label = ImageFont.truetype(font_path, 14)
            font_legend = ImageFont.truetype(font_path, 12)
        else:
            font_title = ImageFont.load_default()
            font_label = ImageFont.load_default()
            font_legend = ImageFont.load_default()

        # Title
        title = f"Visible Line Problem - {self._n} Lines"
        title_bbox = draw.textbbox((0, 0), title, font=font_title)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = (width - title_width) // 2
        draw.text((title_x, 15), title, fill=(30, 30, 30), font=font_title)

        # Define plotting area
        plot_left = padding
        plot_right = width - 250  # Leave space for legend
        plot_top = padding + 40
        plot_bottom = height - padding

        plot_width = plot_right - plot_left
        plot_height = plot_bottom - plot_top

        # Draw plot frame
        draw.rectangle(
            [plot_left, plot_top, plot_right, plot_bottom],
            outline=(100, 100, 100),
            width=2,
        )

        # Determine x range for plotting
        x_range = 20  # Show lines from x=-10 to x=10
        x_min = -x_range // 2
        x_max = x_range // 2

        # Calculate y values at x_min and x_max for all lines to determine y range
        y_values = []
        for A, B in self._lines:
            y_values.append(A * x_min + B)
            y_values.append(A * x_max + B)

        if y_values:
            y_min = min(y_values)
            y_max = max(y_values)
            y_range = y_max - y_min
            if y_range == 0:
                y_range = 1
            # Add some padding to y range
            y_padding = y_range * 0.1
            y_min -= y_padding
            y_max += y_padding
            y_range = y_max - y_min
        else:
            y_min = -10
            y_max = 10
            y_range = 20

        def world_to_screen(x: float, y: float) -> tuple[int, int]:
            """Convert world coordinates to screen coordinates."""
            screen_x = plot_left + (x - x_min) / x_range * plot_width
            screen_y = plot_top + (1 - (y - y_min) / y_range) * plot_height
            return (int(screen_x), int(screen_y))

        # Draw grid lines
        num_grid_lines = 5
        for i in range(num_grid_lines + 1):
            # Vertical grid lines
            x = x_min + i * x_range / num_grid_lines
            sx, sy1 = world_to_screen(x, y_min)
            _, sy2 = world_to_screen(x, y_max)
            draw.line((sx, sy1, sx, sy2), fill=(220, 220, 220), width=1)

            # Horizontal grid lines
            y = y_min + i * y_range / num_grid_lines
            sx1, sy = world_to_screen(x_min, y)
            sx2, _ = world_to_screen(x_max, y)
            draw.line((sx1, sy, sx2, sy), fill=(220, 220, 220), width=1)

        # Draw axes if they're in range
        if y_min <= 0 <= y_max:
            sx1, sy = world_to_screen(x_min, 0)
            sx2, _ = world_to_screen(x_max, 0)
            draw.line((sx1, sy, sx2, sy), fill=(150, 150, 150), width=2)

        if x_min <= 0 <= x_max:
            sx, sy1 = world_to_screen(0, y_min)
            _, sy2 = world_to_screen(0, y_max)
            draw.line((sx, sy1, sx, sy2), fill=(150, 150, 150), width=2)

        # Generate distinct colors for lines
        visible_set = set(self._gold_answer)
        colors = self._generate_colors(self._n)

        # Draw non-visible lines first (so visible ones are on top)
        for i, (A, B) in enumerate(self._lines):
            if i not in visible_set:
                color = colors[i]
                x1, y1 = x_min, A * x_min + B
                x2, y2 = x_max, A * x_max + B
                sx1, sy1 = world_to_screen(x1, y1)
                sx2, sy2 = world_to_screen(x2, y2)

                # Draw dashed line for non-visible
                self._draw_dashed_line(draw, sx1, sy1, sx2, sy2, color, width=2)

        # Draw visible lines on top
        for i, (A, B) in enumerate(self._lines):
            if i in visible_set:
                color = colors[i]
                x1, y1 = x_min, A * x_min + B
                x2, y2 = x_max, A * x_max + B
                sx1, sy1 = world_to_screen(x1, y1)
                sx2, sy2 = world_to_screen(x2, y2)

                # Draw solid thick line for visible
                draw.line((sx1, sy1, sx2, sy2), fill=color, width=4)

        # Draw labels on lines
        for i, (A, B) in enumerate(self._lines):
            x_label = 0  # Label at x=0
            y_label = A * x_label + B
            sx, sy = world_to_screen(x_label, y_label)

            # Draw label background
            label_text = str(i)
            bbox = draw.textbbox((sx, sy), label_text, font=font_label)
            bg_padding = 3
            draw.rectangle(
                [
                    bbox[0] - bg_padding,
                    bbox[1] - bg_padding,
                    bbox[2] + bg_padding,
                    bbox[3] + bg_padding,
                ],
                fill=(255, 255, 255),
                outline=colors[i],
                width=2,
            )
            draw.text((sx, sy), label_text, fill=colors[i], font=font_label)

        # Draw legend
        legend_x = plot_right + 15
        legend_y = plot_top + 20

        draw.text(
            (legend_x, legend_y),
            "Line Equations:",
            fill=(30, 30, 30),
            font=font_legend,
        )
        legend_y += 25

        for i, (A, B) in enumerate(self._lines):
            color = colors[i]
            is_visible = i in visible_set

            # Draw color indicator
            indicator_size = 12
            draw.rectangle(
                [
                    legend_x,
                    legend_y - indicator_size // 2,
                    legend_x + indicator_size,
                    legend_y + indicator_size // 2,
                ],
                fill=color,
                outline=color,
            )

            # Draw line equation
            sign = "+" if B >= 0 else "-"
            eq_text = f"{i}: y={A}x{sign}{abs(B)}"
            draw.text(
                (legend_x + indicator_size + 5, legend_y - 6),
                eq_text,
                fill=color if is_visible else (150, 150, 150),
                font=font_legend,
            )

            # Draw visibility indicator
            if is_visible:
                draw.text(
                    (legend_x + 110, legend_y - 6),
                    "✓ visible",
                    fill=(0, 150, 0),
                    font=font_legend,
                )

            legend_y += 20

        # Add visibility explanation
        legend_y += 15
        draw.text(
            (legend_x, legend_y),
            "Visibility Rules:",
            fill=(30, 30, 30),
            font=font_legend,
        )
        legend_y += 20
        draw.text(
            (legend_x, legend_y),
            "Solid = Visible",
            fill=(0, 100, 0),
            font=font_legend,
        )
        legend_y += 18
        draw.text(
            (legend_x, legend_y),
            "Dashed = Hidden",
            fill=(150, 150, 150),
            font=font_legend,
        )

        return img

    def _generate_colors(self, n: int) -> list[tuple[int, int, int]]:
        """Generate n distinct colors using HSV color space."""
        colors = []
        for i in range(n):
            hue = (i * 360 / n) % 360
            # Convert HSV to RGB
            h = hue / 60
            x = 1 - abs(h % 2 - 1)
            if 0 <= h < 1:
                r, g, b = 1, x, 0
            elif 1 <= h < 2:
                r, g, b = x, 1, 0
            elif 2 <= h < 3:
                r, g, b = 0, 1, x
            elif 3 <= h < 4:
                r, g, b = 0, x, 1
            elif 4 <= h < 5:
                r, g, b = x, 0, 1
            else:
                r, g, b = 1, 0, x

            # Increase saturation and brightness for visibility
            s, v = 0.8, 0.85
            r = int((r * s + (1 - s)) * v * 255)
            g = int((g * s + (1 - s)) * v * 255)
            b = int((b * s + (1 - s)) * v * 255)
            colors.append((r, g, b))
        return colors

    def _draw_dashed_line(
        self,
        draw: ImageDraw.ImageDraw,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        color: tuple[int, int, int],
        width: int = 2,
        dash_length: int = 10,
    ) -> None:
        """Draw a dashed line."""
        dx = x2 - x1
        dy = y2 - y1
        length = (dx**2 + dy**2) ** 0.5
        if length == 0:
            return

        num_dashes = int(length / (dash_length * 2))
        if num_dashes == 0:
            num_dashes = 1

        for i in range(num_dashes):
            t1 = i * 2 / (num_dashes * 2)
            t2 = (i * 2 + 1) / (num_dashes * 2)
            sx1 = int(x1 + t1 * dx)
            sy1 = int(y1 + t1 * dy)
            sx2 = int(x1 + t2 * dx)
            sy2 = int(y1 + t2 * dy)
            draw.line((sx1, sy1, sx2, sy2), fill=color, width=width)

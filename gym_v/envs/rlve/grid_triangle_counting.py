"""Grid triangle counting environment for gym-v (self-contained)."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation
from gym_v.logger import get_logger

logger = get_logger()


class RLVEGridTriangleCountingEnv(Env):
    """RLVE Grid Triangle Counting as a single-turn environment.

    Count the number of non-degenerate triangles with all three vertices
    at integer coordinate points (x, y) where 0 ≤ x ≤ N and 0 ≤ y ≤ M.

    Source: https://www.luogu.com.cn/problem/P3166
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""How many non-degenerate triangles have all three vertices located at integer coordinate points (x, y) where 0 ≤ x ≤ {N} and 0 ≤ y ≤ {M}?"""

    def __init__(
        self,
        max_n_m: int = 12,
        cell_px: int = 48,
        padding: int = 32,
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._max_n_m = max_n_m
        self._cell_px = cell_px
        self._padding = padding
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._n: int | None = None
        self._m: int | None = None
        self._prompt: str | None = None
        self._reference_answer: int | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        if self._n is not None and self._m is not None:
            size_hint = f"0 ≤ x ≤ {self._n}, 0 ≤ y ≤ {self._m}"
        else:
            size_hint = "0 ≤ x ≤ N, 0 ≤ y ≤ M"

        return dedent(
            f"""
            Grid Triangle Counting Problem:

            Count the number of non-degenerate triangles with all three vertices
            located at integer coordinate points (x, y) where {size_hint}.

            A non-degenerate triangle is one where the three vertices are not
            collinear (don't lie on the same line).

            The grid forms a ({size_hint}) lattice of points.

            In the visualization:
            - The grid shows all available lattice points
            - Each dot represents a valid vertex location
            - Grid dimensions indicate the range of x and y coordinates

            Output format: A single positive integer (the count of triangles).
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
        """Generate a grid triangle counting problem instance.

        Uses Euler's totient function and inclusion-exclusion to compute
        the number of non-degenerate triangles on an N×M lattice.
        """
        max_n_m = self._max_n_m
        if max_n_m < 1:
            raise ValueError("max_n_m must be >= 1")

        N = int(self.np_random.integers(1, max_n_m + 1))
        M = int(self.np_random.integers(1, max_n_m + 1))

        if N > M:
            N, M = M, N

        # Sieve to compute Euler's totient function up to N
        phi = [0] * (N + 1)
        mark = [False] * (N + 1)
        primes = []
        phi[1] = 1
        for i in range(2, N + 1):
            if not mark[i]:
                primes.append(i)
                phi[i] = i - 1
            for p in primes:
                ip = i * p
                if ip > N:
                    break
                mark[ip] = True
                if i % p == 0:
                    phi[ip] = phi[i] * p
                    break
                else:
                    phi[ip] = phi[i] * (p - 1)

        # Combination function C(x, 3) = x*(x-1)*(x-2)/6
        def C(x):
            return x * (x - 1) * (x - 2) // 6

        # Compute the contribution from degenerate (colinear) triples
        degenerate = 0
        for d in range(2, N + 1):
            term = phi[d]
            term *= (N - d + N % d + 2) * (N // d)
            term *= (M - d + M % d + 2) * (M // d)
            degenerate += term // 2

        # Total number of triples of points minus colinear ones
        total_points = (N + 1) * (M + 1)
        total_triples = C(total_points)
        subtract_N_lines = (M + 1) * C(N + 1)
        subtract_M_lines = (N + 1) * C(M + 1)

        answer = total_triples - subtract_N_lines - subtract_M_lines - degenerate

        if answer <= 0:
            raise ValueError(f"Generated invalid answer: {answer}")

        self._n = N
        self._m = M
        self._reference_answer = answer

    def _prompt_generate(self) -> str:
        """Generate the prompt text for the problem."""
        if self._n is None:
            raise RuntimeError("No problem generated")
        return self.prompt_template.format(N=self._n, M=self._m)

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
            -1.0: wrong format (not a positive integer)
             0.0: wrong answer
             1.0: correct answer
        """
        processed_result = self._process(answer)
        if processed_result is not None:
            if processed_result <= 0:
                return -1.0

            if processed_result == self._reference_answer:
                return 1.0
            else:
                return 0.0
        else:
            return -1.0

    def render(self) -> Image.Image | list[Image.Image] | None:
        """Render the grid triangle counting problem as an image.

        Shows:
        - Grid of lattice points (integer coordinates)
        - Highlighted example triangles to illustrate the problem
        - Visual representation of the coordinate space
        """
        if self._n is None:
            raise RuntimeError("No problem generated")

        cell_px = self._cell_px
        padding = self._padding

        # Calculate dimensions
        grid_width = self._n * cell_px
        grid_height = self._m * cell_px

        # Extra space for title and footer
        title_height = 60
        footer_height = 80

        width = padding * 2 + grid_width
        height = padding * 2 + title_height + grid_height + footer_height

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

        # Draw title
        title = "Grid Triangle Counting"
        bbox = draw.textbbox((0, 0), title, font=font_large)
        tw = bbox[2] - bbox[0]
        draw.text(
            (width // 2 - tw // 2, padding),
            title,
            fill=(30, 30, 30),
            font=font_large
        )

        # Grid origin
        grid_x = padding
        grid_y = padding + title_height

        # Draw grid lines (light)
        for i in range(self._n + 1):
            x = grid_x + i * cell_px
            draw.line(
                (x, grid_y, x, grid_y + grid_height),
                fill=(200, 200, 200),
                width=1
            )
        for j in range(self._m + 1):
            y = grid_y + j * cell_px
            draw.line(
                (grid_x, y, grid_x + grid_width, y),
                fill=(200, 200, 200),
                width=1
            )

        # Draw axes (darker)
        draw.line(
            (grid_x, grid_y + grid_height, grid_x + grid_width, grid_y + grid_height),
            fill=(80, 80, 80),
            width=2
        )
        draw.line(
            (grid_x, grid_y, grid_x, grid_y + grid_height),
            fill=(80, 80, 80),
            width=2
        )

        # Draw lattice points
        point_radius = 4
        for i in range(self._n + 1):
            for j in range(self._m + 1):
                x = grid_x + i * cell_px
                y = grid_y + (self._m - j) * cell_px  # Flip y for standard coords
                draw.ellipse(
                    [x - point_radius, y - point_radius,
                     x + point_radius, y + point_radius],
                    fill=(50, 100, 200),
                    outline=(30, 60, 120),
                    width=1
                )

        # Draw example triangles if grid is large enough
        if self._n >= 2 and self._m >= 2:
            # Example triangle 1: Simple right triangle
            t1_points = [
                (0, 0),
                (min(2, self._n), 0),
                (0, min(1, self._m))
            ]
            self._draw_triangle(
                draw, grid_x, grid_y, grid_height, cell_px,
                t1_points, (255, 100, 100, 80)
            )

            # Example triangle 2: Different shape
            if self._n >= 2 and self._m >= 2:
                t2_points = [
                    (min(1, self._n), min(1, self._m)),
                    (min(2, self._n), min(2, self._m)),
                    (min(2, self._n), min(1, self._m))
                ]
                self._draw_triangle(
                    draw, grid_x, grid_y, grid_height, cell_px,
                    t2_points, (100, 200, 100, 80)
                )

        # Draw axis labels
        # X-axis
        for i in [0, self._n // 2, self._n]:
            if i <= self._n:
                x = grid_x + i * cell_px
                y = grid_y + grid_height + 5
                label = str(i)
                bbox = draw.textbbox((0, 0), label, font=font_small)
                tw = bbox[2] - bbox[0]
                draw.text((x - tw // 2, y), label, fill=(80, 80, 80), font=font_small)

        # Y-axis
        for j in [0, self._m // 2, self._m]:
            if j <= self._m:
                x = grid_x - 20
                y = grid_y + (self._m - j) * cell_px - 6
                label = str(j)
                draw.text((x, y), label, fill=(80, 80, 80), font=font_small)

        # Axis names
        draw.text(
            (grid_x + grid_width + 10, grid_y + grid_height - 10),
            "x",
            fill=(80, 80, 80),
            font=font_medium
        )
        draw.text(
            (grid_x + 5, grid_y - 20),
            "y",
            fill=(80, 80, 80),
            font=font_medium
        )

        # Draw footer information
        footer_y = grid_y + grid_height + 35
        info1 = f"Lattice points: 0 ≤ x ≤ {self._n}, 0 ≤ y ≤ {self._m}"
        draw.text((padding, footer_y), info1, fill=(30, 30, 30), font=font_medium)
        footer_y += 25
        info2 = f"Total points: {(self._n + 1) * (self._m + 1)}"
        draw.text((padding, footer_y), info2, fill=(80, 80, 80), font=font_small)
        footer_y += 20
        info3 = "Count all non-degenerate triangles"
        draw.text((padding, footer_y), info3, fill=(80, 80, 80), font=font_small)

        return img

    def _draw_triangle(
        self,
        draw: ImageDraw.ImageDraw,
        grid_x: int,
        grid_y: int,
        grid_height: int,
        cell_px: int,
        points: list[tuple[int, int]],
        color: tuple[int, int, int, int]
    ) -> None:
        """Draw a semi-transparent triangle on the grid.

        Args:
            draw: PIL ImageDraw object
            grid_x: Grid origin x
            grid_y: Grid origin y
            grid_height: Height of grid in pixels
            cell_px: Size of each cell
            points: List of 3 (x, y) coordinate tuples
            color: RGBA color tuple
        """
        if len(points) != 3:
            return

        # Convert lattice coordinates to pixel coordinates
        pixel_points = []
        for x, y in points:
            px = grid_x + x * cell_px
            py = grid_y + (grid_height // cell_px - y) * cell_px
            pixel_points.append((px, py))

        # Draw the triangle with semi-transparent fill
        # Note: PIL doesn't support alpha for fill in RGB mode, so we draw outline only
        draw.polygon(
            pixel_points,
            outline=(color[0], color[1], color[2]),
            width=2
        )

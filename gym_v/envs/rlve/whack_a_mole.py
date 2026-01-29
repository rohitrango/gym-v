"""Whack-a-Mole environment for gym-v (self-contained)."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation
from gym_v.logger import get_logger

logger = get_logger()


class RLVEWhackAMoleEnv(Env):
    """RLVE Whack-a-Mole as a single-turn environment.

    Given an N×M grid where each cell contains moles, find the minimum number
    of hammer swings needed to remove all moles. The hammer has a fixed r×c size
    chosen before starting, and each swing removes exactly 1 mole from each cell
    in the hammer's coverage area.
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""You are given an {N} × {M} grid, where each cell contains a non-negative integer representing the number of moles in that hole:
{grid}

You are allowed to define a **fixed** hammer size of r × c (1 ≤ r ≤ {N}, 1 ≤ c ≤ {M}) before starting. Each time you swing the hammer:
- You choose an r × c subrectangle in the grid (without rotation).
- This subrectangle must be fully within the grid.
- Each cell in the subrectangle must contain at least 1 mole.
- Each cell in the subrectangle has exactly 1 mole removed (so r × c moles are removed per swing).

You may swing the hammer multiple times, but you cannot change its size after choosing r and c. Your goal is to remove all the moles from the grid with the **minimum number of swings**.

**Output Format:** Your final answer should be a single integer — the **minimum number of hammer swings** required to remove all moles from the grid."""

    def __init__(
        self,
        max_n_m: int = 4,
        max_beat: int = 3,
        cell_px: int = 80,
        padding: int = 24,
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._max_n_m = max_n_m
        self._max_beat = max_beat
        self._cell_px = cell_px
        self._padding = padding
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        if max_beat < 1:
            raise ValueError("max_beat should be >= 1")

        self._n: int | None = None
        self._m: int | None = None
        self._grid: list[list[int]] | None = None
        self._prompt: str | None = None
        self._oracle_answer: int | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        if self._n and self._m:
            size_hint = f"{self._n} × {self._m}"
        else:
            size_hint = "N × M"

        return dedent(
            f"""
            Whack-a-Mole Problem:

            Given a {size_hint} grid where each cell contains moles (non-negative integers),
            find the minimum number of hammer swings to remove all moles.

            Rules:
            1) Choose a fixed hammer size r × c before starting (1 ≤ r ≤ N, 1 ≤ c ≤ M)
            2) Each swing covers an r × c subrectangle within the grid
            3) Each cell in the subrectangle must have at least 1 mole
            4) Each swing removes exactly 1 mole from each covered cell
            5) Cannot change hammer size after choosing

            Goal: Minimize the total number of hammer swings.

            In the visualization:
            - Grid cells show the number of moles in each hole
            - Numbers indicate mole count per cell
            - Darker cells have more moles
            - Sample hammer coverage areas are shown below the grid

            Output format: A single integer (minimum number of swings).
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
            text=state_text,
            metadata={
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
        """Generate a whack-a-mole problem instance.

        Ports generation logic from RLVE using self.np_random.
        """
        MAX_N_M = self._max_n_m
        if MAX_N_M < 2:
            raise ValueError("max_n_m must be >= 2")

        N = int(self.np_random.integers(2, MAX_N_M + 1))
        M = int(self.np_random.integers(2, MAX_N_M + 1))

        R = int(self.np_random.integers(1, N + 1))
        C = int(self.np_random.integers(1, M + 1))

        # Generate grid using 2D difference array technique
        grid = [[0] * M for _ in range(N)]
        for i in range(N - R + 1):
            for j in range(M - C + 1):
                num_moles = int(self.np_random.integers(0, self._max_beat + 1))
                grid[i][j] += num_moles
                if i + R < N:
                    grid[i + R][j] -= num_moles
                if j + C < M:
                    grid[i][j + C] -= num_moles
                if i + R < N and j + C < M:
                    grid[i + R][j + C] += num_moles

        # Compute prefix sum to get actual grid values
        for i in range(N):
            for j in range(M):
                if i > 0:
                    grid[i][j] += grid[i - 1][j]
                if j > 0:
                    grid[i][j] += grid[i][j - 1]
                if i > 0 and j > 0:
                    grid[i][j] -= grid[i - 1][j - 1]

        total = sum(sum(row) for row in grid)
        if total == 0:
            self._n = N
            self._m = M
            self._grid = grid
            self._oracle_answer = 0
            return

        best_area = 0

        # Try every possible hammer size r x c, largest area first
        for area in range(N * M, 0, -1):
            if total % area != 0:
                continue
            if area <= best_area:
                continue
            for r in range(1, area + 1):
                if area % r != 0:
                    continue
                c = area // r
                if not (1 <= r <= N and 1 <= c <= M):
                    continue
                if area <= best_area:
                    continue

                # 2D difference array, size (N+1)x(M+1)
                diff = [[0] * (M + 1) for _ in range(N + 1)]
                ok = True

                # Sweep through the grid, maintaining prefix-sum of diff
                for i in range(N):
                    for j in range(M):
                        # Accumulate 2D prefix sum at (i,j)
                        if i > 0:
                            diff[i][j] += diff[i - 1][j]
                        if j > 0:
                            diff[i][j] += diff[i][j - 1]
                        if i > 0 and j > 0:
                            diff[i][j] -= diff[i - 1][j - 1]

                        # If we've hit more moles here than exist, fail
                        if diff[i][j] > grid[i][j]:
                            ok = False
                            break

                        # If we haven't hit enough, schedule hammer swings
                        if diff[i][j] < grid[i][j]:
                            # Must be able to place an r×c rectangle here
                            if i + r > N or j + c > M:
                                ok = False
                                break
                            t = grid[i][j] - diff[i][j]
                            # 2D-difference updates for adding t to rectangle [i..i+r-1][j..j+c-1]
                            diff[i][j] += t
                            diff[i + r][j] -= t
                            diff[i][j + c] -= t
                            diff[i + r][j + c] += t
                    if not ok:
                        break

                if ok:
                    best_area = area

        # The minimum number of swings is total moles divided by the largest valid hammer area
        self._n = N
        self._m = M
        self._grid = grid
        self._oracle_answer = total // best_area

    def _prompt_generate(self) -> str:
        """Generate the prompt text for the problem."""
        if self._n is None:
            raise RuntimeError("No problem generated")
        return self.prompt_template.format(
            N=self._n,
            M=self._m,
            grid="\n".join(" ".join(map(str, row)) for row in self._grid),
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
             0.0: wrong answer
            +1.0: correct answer
        """
        processed_result = self._process(answer)
        if processed_result is not None:
            if processed_result == self._oracle_answer:
                return 1.0
            else:
                return 0.0
        else:
            return 0.0

    def render(self) -> Image.Image | list[Image.Image] | None:
        """Render the whack-a-mole grid as an image.

        Shows:
        - Grid with mole counts in each cell
        - Visual intensity based on mole count
        - Sample hammer coverage areas
        """
        if self._n is None or self._grid is None:
            raise RuntimeError("No problem generated")

        cell_px = self._cell_px
        padding = self._padding

        rows = self._n
        cols = self._m

        # Calculate dimensions
        grid_width = cols * cell_px
        grid_height = rows * cell_px

        # Space for hammer examples
        hammer_legend_height = 140

        width = padding * 2 + grid_width
        height = padding * 3 + grid_height + hammer_legend_height

        img = Image.new("RGB", (width, height), (250, 250, 250))
        draw = ImageDraw.Draw(img)

        # Load font
        font_path = None
        font_path_candidate = self.assets_dir / "DejaVuSans.ttf"
        if font_path_candidate.exists():
            font_path = str(font_path_candidate)

        if font_path:
            font_large = ImageFont.truetype(font_path, int(cell_px * 0.4))
            font_medium = ImageFont.truetype(font_path, 18)
            font_small = ImageFont.truetype(font_path, 14)
        else:
            font_large = ImageFont.load_default()
            font_medium = ImageFont.load_default()
            font_small = ImageFont.load_default()

        # Find max mole count for color scaling
        max_moles = max(max(row) for row in self._grid)
        if max_moles == 0:
            max_moles = 1  # Avoid division by zero

        # Draw grid
        grid_x = padding
        grid_y = padding

        # Draw grid cells with color intensity based on mole count
        for r in range(rows):
            for c in range(cols):
                x = grid_x + c * cell_px
                y = grid_y + r * cell_px

                mole_count = self._grid[r][c]
                # Color gradient: white (0 moles) to brown (max moles)
                intensity_ratio = mole_count / max_moles
                # Brown color: (139, 69, 19)
                # White: (255, 255, 255)
                red = int(255 - intensity_ratio * (255 - 139))
                green = int(255 - intensity_ratio * (255 - 69))
                blue = int(255 - intensity_ratio * (255 - 19))
                cell_color = (red, green, blue)

                draw.rectangle(
                    [x + 2, y + 2, x + cell_px - 2, y + cell_px - 2],
                    fill=cell_color,
                    outline=None,
                )

        # Draw grid lines
        for r in range(rows + 1):
            y = grid_y + r * cell_px
            draw.line((grid_x, y, grid_x + grid_width, y), fill=(30, 30, 30), width=2)
        for c in range(cols + 1):
            x = grid_x + c * cell_px
            draw.line((x, grid_y, x, grid_y + grid_height), fill=(30, 30, 30), width=2)

        # Draw mole counts
        for r in range(rows):
            for c in range(cols):
                v = str(self._grid[r][c])
                bbox = draw.textbbox((0, 0), v, font=font_large)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                cx = grid_x + c * cell_px + cell_px // 2
                cy = grid_y + r * cell_px + cell_px // 2

                # Use white text for dark cells, black for light cells
                mole_count = self._grid[r][c]
                intensity_ratio = mole_count / max_moles
                text_color = (255, 255, 255) if intensity_ratio > 0.5 else (10, 10, 10)

                draw.text(
                    (cx - tw // 2, cy - th // 2), v, fill=text_color, font=font_large
                )

        # Draw hammer coverage examples
        legend_y = grid_y + grid_height + padding * 2

        # Title
        title = "Example Hammer Coverage Patterns:"
        draw.text((padding, legend_y), title, fill=(30, 30, 30), font=font_medium)
        legend_y += 30

        # Show a few example hammer sizes
        example_sizes = []
        if rows >= 1 and cols >= 1:
            example_sizes.append((1, 1))
        if rows >= 2 and cols >= 2:
            example_sizes.append((2, 2))
        if rows >= 1 and cols >= 2:
            example_sizes.append((1, 2))
        if rows >= 2 and cols >= 1:
            example_sizes.append((2, 1))

        # Take up to 3 examples
        example_sizes = example_sizes[:3]

        small_cell = 20
        example_x = padding + 20

        for r_size, c_size in example_sizes:
            # Draw small grid showing hammer coverage
            for r in range(r_size):
                for c in range(c_size):
                    x = example_x + c * small_cell
                    y = legend_y + r * small_cell
                    draw.rectangle(
                        [x, y, x + small_cell, y + small_cell],
                        fill=(200, 100, 100),  # Reddish to indicate hammer coverage
                        outline=(30, 30, 30),
                        width=1,
                    )

            # Label
            label = f"{r_size}×{c_size} hammer"
            label_x = example_x + c_size * small_cell + 10
            label_y = legend_y + (r_size * small_cell) // 2 - 7
            draw.text((label_x, label_y), label, fill=(30, 30, 30), font=font_small)

            example_x += c_size * small_cell + 100

        legend_y += (
            max(
                20,
                max(r_size for r_size, _ in example_sizes) * small_cell
                if example_sizes
                else 0,
            )
            + 15
        )

        # Add constraint text
        constraint_text = (
            "Choose ONE hammer size. Each swing removes 1 mole from each covered cell."
        )
        draw.text(
            (padding, legend_y), constraint_text, fill=(100, 100, 100), font=font_small
        )

        return img

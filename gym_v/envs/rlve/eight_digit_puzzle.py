"""Eight-digit puzzle (8-puzzle) environment for gym-v (self-contained)."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation
from gym_v.logger import get_logger

logger = get_logger()


class RLVEEightDigitPuzzleEnv(Env):
    """RLVE Eight-digit puzzle (8-puzzle sliding tile puzzle) as a single-turn environment."""

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""You are given a {N} × {M} grid, where each cell contains a digit from `0` to `{NM_minus_1}`. At any time, you can **swap the `0`** with one of its four (existing) neighbors:
- `U` = up
- `D` = down
- `L` = left
- `R` = right

You start with the following grid:
{start_grid}

Your goal is to reach the following grid:
{destination_grid}

**Output Format:** Output a single line containing the sequence of moves made by the `0`, represented by a string of characters (`U`, `D`, `L`, `R`). For example, `RRDDLLUU` (do **NOT** include backticks or quotes) means: right, right, down, down, left, left, up, up."""

    action2delta = {
        "L": (0, -1),
        "R": (0, +1),
        "U": (-1, 0),
        "D": (+1, 0),
    }

    def __init__(
        self,
        n: int = 3,
        m: int = 3,
        steps: int = 10,
        cell_px: int = 72,
        padding: int = 32,
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._n = n
        self._m = m
        self._steps = steps
        self._cell_px = cell_px
        self._padding = padding
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._start_grid: list[list[int]] | None = None
        self._destination_grid: list[list[int]] | None = None
        self._zero_i: int = 0
        self._zero_j: int = 0
        self._prompt: str | None = None
        self._oracle_answer: str | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        size_hint = f"{self._n} x {self._m}" if self._start_grid else "N x M"
        return dedent(
            f"""
            This is a sliding tile puzzle where you rearrange numbered tiles by
            sliding them into an empty space (represented by 0) to transform a
            starting grid configuration into a goal configuration.

            Eight-digit puzzle (8-puzzle) rules:
            1) The grid contains digits 0 to {self._n * self._m - 1}.
            2) Digit 0 represents the empty space.
            3) You can slide tiles into the empty space by moving the 0.
            4) Moves: U (up), D (down), L (left), R (right).
            5) Goal: Transform the start grid into the destination grid.

            In the image:
            - The grid is {size_hint}
            - Empty space (0) is shown with a light blue background
            - Numbered tiles (1-{self._n * self._m - 1}) have white backgrounds
            - Grid lines clearly separate each cell

            Output Format: Output a single line containing the sequence of moves
            made by the 0, represented by a string of characters (U, D, L, R).
            For example, RRDDLLUU (do NOT include backticks or quotes) means:
            right, right, down, down, left, left, up, up.
            """
        ).strip()

    def _get_state_text(self) -> str:
        """Return text representation of the puzzle state."""
        if self._start_grid is None or self._destination_grid is None:
            return ""
        start_str = "\n".join(" ".join(map(str, row)) for row in self._start_grid)
        dest_str = "\n".join(" ".join(map(str, row)) for row in self._destination_grid)
        return f"Start:\n{start_str}\n\nDestination:\n{dest_str}"

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
        """Generate a random 8-puzzle instance."""
        N = self._n
        M = self._m

        if N < 2 or M < 2:
            raise ValueError("N and M must be >= 2")

        # Generate random starting permutation
        start_permutation = list(range(N * M))
        self.np_random.shuffle(start_permutation)
        self._start_grid = [
            [start_permutation[i * M + j] for j in range(M)] for i in range(N)
        ]

        # Find position of 0 (empty space)
        for i in range(N):
            for j in range(M):
                if self._start_grid[i][j] == 0:
                    self._zero_i = i
                    self._zero_j = j
                    break

        # Create destination grid by making random moves from start
        self._destination_grid = [row.copy() for row in self._start_grid]
        zero_i, zero_j = self._zero_i, self._zero_j

        # Generate random action distribution
        action_weights = [int(self.np_random.integers(1, N * M + 1)) for _ in range(4)]
        action_sum = sum(action_weights)
        action_distribution = [w / action_sum for w in action_weights]

        self._oracle_answer = ""
        for step in range(self._steps):
            while True:
                # Choose random action based on distribution
                action = self.np_random.choice(
                    ["U", "D", "L", "R"], p=action_distribution
                )
                new_zero_i = zero_i + self.action2delta[action][0]
                new_zero_j = zero_j + self.action2delta[action][1]

                # Check if move is valid
                if 0 <= new_zero_i < N and 0 <= new_zero_j < M:
                    self._oracle_answer += action
                    # Swap 0 with the tile at new position
                    (
                        self._destination_grid[zero_i][zero_j],
                        self._destination_grid[new_zero_i][new_zero_j],
                    ) = (
                        self._destination_grid[new_zero_i][new_zero_j],
                        self._destination_grid[zero_i][zero_j],
                    )
                    zero_i, zero_j = new_zero_i, new_zero_j
                    break

    def _prompt_generate(self) -> str:
        """Generate the prompt text for the puzzle."""
        if self._start_grid is None or self._destination_grid is None:
            raise RuntimeError("No grid generated")
        N = len(self._start_grid)
        M = len(self._start_grid[0])
        return self.prompt_template.format(
            N=N,
            M=M,
            NM_minus_1=N * M - 1,
            start_grid="\n".join(" ".join(map(str, row)) for row in self._start_grid),
            destination_grid="\n".join(
                " ".join(map(str, row)) for row in self._destination_grid
            ),
        )

    def _process(self, answer: str | None) -> str | None:
        """Process the answer string."""
        if answer is None:
            return None
        return answer.strip()

    def _score_answer(self, answer: str) -> float:
        """Score the submitted answer."""
        processed_result = self._process(answer)
        if processed_result is None or processed_result == "":
            return 0.0
        destination_grid = [row.copy() for row in self._start_grid]
        zero_i, zero_j = self._zero_i, self._zero_j
        N = len(self._start_grid)
        M = len(self._start_grid[0])

        for action in processed_result:
            # Check if action is valid
            if action not in self.action2delta:
                return 0.0
            new_zero_i = zero_i + self.action2delta[action][0]
            new_zero_j = zero_j + self.action2delta[action][1]

            # Check if move is within bounds
            if 0 <= new_zero_i < N and 0 <= new_zero_j < M:
                # Swap 0 with the tile at new position
                (
                    destination_grid[zero_i][zero_j],
                    destination_grid[new_zero_i][new_zero_j],
                ) = (
                    destination_grid[new_zero_i][new_zero_j],
                    destination_grid[zero_i][zero_j],
                )
                zero_i, zero_j = new_zero_i, new_zero_j
            else:
                return 0.0
        matches = sum(
            sum(int(a == b) for a, b in zip(gold_row, answer_row, strict=False))
            for gold_row, answer_row in zip(
                self._destination_grid, destination_grid, strict=False
            )
        )
        return (matches / (N * M)) ** 10

    def render(self) -> Image.Image | list[Image.Image] | None:
        """Render the puzzle as an image showing the start and destination grids."""
        if self._start_grid is None or self._destination_grid is None:
            raise RuntimeError("No grid generated")

        rows = len(self._start_grid)
        cols = len(self._start_grid[0])
        cell_px = self._cell_px
        padding = self._padding
        gap = padding * 2  # Gap between the two grids

        # Calculate total image size (two grids side by side)
        grid_width = cols * cell_px
        grid_height = rows * cell_px
        total_width = padding * 2 + grid_width * 2 + gap
        total_height = padding * 2 + grid_height

        img = Image.new("RGB", (total_width, total_height), (245, 245, 245))
        draw = ImageDraw.Draw(img)

        # Load font
        font_path = None
        font_path_candidate = self.assets_dir / "DejaVuSans.ttf"
        if font_path_candidate.exists():
            font_path = str(font_path_candidate)

        if font_path:
            font = ImageFont.truetype(font_path, int(cell_px * 0.4))
            font_label = ImageFont.truetype(font_path, int(cell_px * 0.25))
        else:
            font = ImageFont.load_default()
            font_label = ImageFont.load_default()

        # Helper function to draw a grid
        def draw_grid(grid, x_offset):
            # Draw grid lines
            for r in range(rows + 1):
                y = padding + r * cell_px
                draw.line(
                    (
                        x_offset,
                        y,
                        x_offset + grid_width,
                        y,
                    ),
                    fill=(40, 40, 40),
                    width=2,
                )
            for c in range(cols + 1):
                x = x_offset + c * cell_px
                draw.line(
                    (
                        x,
                        padding,
                        x,
                        padding + grid_height,
                    ),
                    fill=(40, 40, 40),
                    width=2,
                )

            # Draw cell backgrounds and numbers
            for r in range(rows):
                for c in range(cols):
                    v = grid[r][c]
                    cell_x = x_offset + c * cell_px
                    cell_y = padding + r * cell_px

                    # Fill cell background (light blue for 0, white for others)
                    if v == 0:
                        draw.rectangle(
                            (
                                cell_x + 2,
                                cell_y + 2,
                                cell_x + cell_px - 2,
                                cell_y + cell_px - 2,
                            ),
                            fill=(180, 220, 240),
                        )
                    else:
                        draw.rectangle(
                            (
                                cell_x + 2,
                                cell_y + 2,
                                cell_x + cell_px - 2,
                                cell_y + cell_px - 2,
                            ),
                            fill=(255, 255, 255),
                        )

                    # Draw number
                    text = str(v)
                    bbox = draw.textbbox((0, 0), text, font=font)
                    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    cx = cell_x + cell_px // 2
                    cy = cell_y + cell_px // 2
                    text_color = (100, 100, 100) if v == 0 else (20, 20, 20)
                    draw.text(
                        (cx - tw // 2, cy - th // 2),
                        text,
                        fill=text_color,
                        font=font,
                    )

        # Draw start grid (left)
        draw_grid(self._start_grid, padding)

        # Draw destination grid (right)
        draw_grid(self._destination_grid, padding + grid_width + gap)

        # Add labels
        label_start = "Start"
        label_dest = "Goal"

        bbox_start = draw.textbbox((0, 0), label_start, font=font_label)
        tw_start = bbox_start[2] - bbox_start[0]
        cx_start = padding + grid_width // 2

        bbox_dest = draw.textbbox((0, 0), label_dest, font=font_label)
        tw_dest = bbox_dest[2] - bbox_dest[0]
        cx_dest = padding + grid_width + gap + grid_width // 2

        label_y = padding // 2 - 5

        draw.text(
            (cx_start - tw_start // 2, label_y),
            label_start,
            fill=(60, 60, 60),
            font=font_label,
        )
        draw.text(
            (cx_dest - tw_dest // 2, label_y),
            label_dest,
            fill=(60, 60, 60),
            font=font_label,
        )

        return img

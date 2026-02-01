"""Twiddle puzzle environment for gym-v (self-contained)."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation
from gym_v.logger import get_logger

logger = get_logger()


class RLVETwiddlePuzzleEnv(Env):
    """RLVE Twiddle puzzle as a single-turn environment.

    A twiddle puzzle presents a grid of numbers that can be transformed
    by rotating K x K subgrids 90 degrees counterclockwise.
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""You are given a {N} × {M} grid, where each cell contains a digit from `0` to `{NM_minus_1}`. At any time, you may select a cell `(i, j)` such that 0 ≤ i ≤ {N} - {K} and 0 ≤ j ≤ {M} - {K}. Then, you perform a **90-degree counterclockwise rotation** on the {K} × {K} subgrid starting at position `(i, j)`.

You start with the following grid:
{start_grid}

Your goal is to transform it into the following grid:
{destination_grid}

**Output Format:** Each action should be written on its own line as `i j`, where `i` and `j` are the row and column indices of the top-left corner of the rotated subgrid. Example: `0 1` (do **NOT** include backticks or quotes). Output one action per line in the order they should be performed."""

    def __init__(
        self,
        max_n_m: int = 4,
        steps: int = 3,
        cell_px: int = 56,
        padding: int = 24,
        num_players: int = 1,
        **kwargs: Any,
    ):
        """Initialize the RLVETwiddlePuzzleEnv.

        Args:
            max_n_m: Maximum grid dimension (N and M are chosen in [2, max_n_m]).
            steps: Number of random rotations to generate the destination grid.
            cell_px: Size of each cell in pixels for rendering.
            padding: Padding around the grid in pixels.
            num_players: Number of players (default 1).
            **kwargs: Additional arguments passed to parent Env.
        """
        super().__init__(**kwargs)
        self._max_n_m = max_n_m
        self._steps = steps
        self._cell_px = cell_px
        self._padding = padding
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._n: int | None = None
        self._m: int | None = None
        self._k: int | None = None
        self._start_grid: list[list[int]] | None = None
        self._destination_grid: list[list[int]] | None = None
        self._prompt: str | None = None
        self._oracle_answer: str | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        if self._n and self._m and self._k:
            size_hint = f"{self._n} x {self._m}, K={self._k}"
        else:
            size_hint = "N x M, K=rotation size"
        return dedent(
            f"""
            Twiddle puzzle rules:
            1) You have a grid with numbers from 0 to N*M-1.
            2) Select a position (i, j) where 0 ≤ i ≤ N - K and 0 ≤ j ≤ M - K.
            3) Rotate the K × K subgrid starting at (i, j) by 90 degrees counterclockwise.
            4) Repeat until you match the destination grid.

            In the image:
            - LEFT: Starting grid
            - RIGHT: Destination (goal) grid
            - The grid is {size_hint}

            Output format: One action per line as 'i j' (space-separated row and column indices).
            Each action rotates a K × K subgrid starting at position (i, j).
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
        """Generate a twiddle puzzle instance."""
        max_n_m = self._max_n_m
        if max_n_m < 2:
            raise ValueError("max_n_m must be >= 2")

        steps = self._steps
        if steps < 1:
            raise ValueError("steps must be >= 1")

        # Generate random N and M
        N = int(self.np_random.integers(2, max_n_m + 1))
        M = int(self.np_random.integers(2, max_n_m + 1))
        self._n = N
        self._m = M

        # Generate K (rotation size)
        K = int(self.np_random.integers(2, min(N, M) + 1))
        self._k = K

        # Generate starting grid as a random permutation of 0 to N*M-1
        start_permutation = list(range(N * M))
        self.np_random.shuffle(start_permutation)
        self._start_grid = [
            [start_permutation[i * M + j] for j in range(M)] for i in range(N)
        ]

        # Generate destination grid by applying random rotations
        destination_grid = [row.copy() for row in self._start_grid]
        reference_answer_lines = []

        for _ in range(steps):
            # Random valid position for K x K rotation
            i = int(self.np_random.integers(0, N - K + 1))
            j = int(self.np_random.integers(0, M - K + 1))
            reference_answer_lines.append(f"{i} {j}")

            # Apply 90-degree counterclockwise rotation to K x K subgrid
            new_grid = [row.copy() for row in destination_grid]
            for x in range(K):
                for y in range(K):
                    # Counterclockwise: (x, y) -> (K-1-y, x)
                    new_grid[i + K - 1 - y][j + x] = destination_grid[i + x][j + y]
            destination_grid = new_grid

        self._destination_grid = destination_grid
        self._oracle_answer = "\n".join(reference_answer_lines)

    def _prompt_generate(self) -> str:
        """Generate the prompt text for the puzzle."""
        if self._start_grid is None or self._destination_grid is None:
            raise RuntimeError("No puzzle generated")
        N, M = self._n, self._m
        return self.prompt_template.format(
            N=N,
            M=M,
            NM_minus_1=N * M - 1,
            K=self._k,
            start_grid="\n".join(" ".join(map(str, row)) for row in self._start_grid),
            destination_grid="\n".join(
                " ".join(map(str, row)) for row in self._destination_grid
            ),
        )

    def _process(self, answer: str | None) -> list[tuple[int, int]] | None:
        """Process the answer string into a list of (i, j) actions.

        Args:
            answer: The answer string containing actions.

        Returns:
            List of (i, j) tuples or None if format is invalid.
        """
        if answer is None:
            return None
        answer = answer.strip()

        # Empty answer is considered invalid format
        if not answer:
            return None

        actions = []
        for line in answer.splitlines():
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) != 2:
                    return None
                try:
                    i = int(parts[0])
                    j = int(parts[1])
                    actions.append((i, j))
                except ValueError:
                    return None
        return actions

    def _score_answer(self, answer: str) -> float:
        """Score the answer by simulating the actions and comparing to destination.

        Args:
            answer: The answer string containing actions.

        Returns:
            Reward score: -1.0 for wrong format, -0.5 for invalid solution,
            or a score based on how well the result matches the destination.
        """
        processed_result = self._process(answer)
        if processed_result is None:
            return 0.0
        N, M, K = self._n, self._m, self._k
        destination_grid = [row.copy() for row in self._start_grid]

        # Apply each action
        for i, j in processed_result:
            # Check if action is valid
            if not (0 <= i <= N - K and 0 <= j <= M - K):
                return 0.0
            new_grid = [row.copy() for row in destination_grid]
            for x in range(K):
                for y in range(K):
                    new_grid[i + K - 1 - y][j + x] = destination_grid[i + x][j + y]
            destination_grid = new_grid

        # Compare with expected destination
        matching_cells = sum(
            1
            for i in range(N)
            for j in range(M)
            if destination_grid[i][j] == self._destination_grid[i][j]
        )
        total_cells = N * M
        accuracy = matching_cells / total_cells

        # Perfect match gets 1.0, otherwise use power function for partial credit
        if accuracy == 1.0:
            return 1.0
        else:
            # Use the same scoring strategy as RLVE: (accuracy)^5
            # This ensures wrong answers get very low scores
            return accuracy**5.0

    def render(self) -> Image.Image | list[Image.Image] | None:
        """Render the puzzle as an image showing start and destination grids.

        Returns:
            PIL Image showing the start grid (left) and destination grid (right).
        """
        if self._start_grid is None or self._destination_grid is None:
            raise RuntimeError("No puzzle generated")

        rows, cols = self._n, self._m
        cell_px = self._cell_px
        padding = self._padding
        grid_spacing = padding * 2  # Space between the two grids

        # Total width: two grids side by side with spacing
        single_grid_width = cols * cell_px
        single_grid_height = rows * cell_px
        total_width = padding * 2 + single_grid_width * 2 + grid_spacing
        total_height = padding * 2 + single_grid_height

        img = Image.new("RGB", (total_width, total_height), (250, 250, 250))
        draw = ImageDraw.Draw(img)

        # Load font
        font_path = None
        font_path_candidate = self.assets_dir / "DejaVuSans.ttf"
        if font_path_candidate.exists():
            font_path = str(font_path_candidate)

        if font_path:
            font = ImageFont.truetype(font_path, int(cell_px * 0.40))
            label_font = ImageFont.truetype(font_path, int(cell_px * 0.35))
        else:
            font = ImageFont.load_default()
            label_font = ImageFont.load_default()

        # Draw both grids
        grids = [
            (self._start_grid, "START", padding),
            (
                self._destination_grid,
                "GOAL",
                padding + single_grid_width + grid_spacing,
            ),
        ]

        for grid, label, x_offset in grids:
            # Draw label above grid
            label_bbox = draw.textbbox((0, 0), label, font=label_font)
            label_width = label_bbox[2] - label_bbox[0]
            label_height = label_bbox[3] - label_bbox[1]
            label_x = x_offset + single_grid_width // 2 - label_width // 2
            label_y = padding // 2 - label_height // 2
            draw.text((label_x, label_y), label, fill=(60, 60, 60), font=label_font)

            # Draw grid lines
            for r in range(rows + 1):
                y = padding + r * cell_px
                draw.line(
                    (x_offset, y, x_offset + single_grid_width, y),
                    fill=(30, 30, 30),
                    width=2,
                )
            for c in range(cols + 1):
                x = x_offset + c * cell_px
                draw.line(
                    (x, padding, x, padding + single_grid_height),
                    fill=(30, 30, 30),
                    width=2,
                )

            # Draw numbers
            for r in range(rows):
                for c in range(cols):
                    v = str(grid[r][c])
                    bbox = draw.textbbox((0, 0), v, font=font)
                    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    cx = x_offset + c * cell_px + cell_px // 2
                    cy = padding + r * cell_px + cell_px // 2
                    draw.text(
                        (cx - tw // 2, cy - th // 2),
                        v,
                        fill=(10, 10, 10),
                        font=font,
                    )

        return img

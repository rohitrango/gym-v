"""Matrix permutation main diagonal one environment for gym-v (self-contained)."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation
from gym_v.logger import get_logger

logger = get_logger()


class MatrixPermutationMainDiagonalOneEnv(Env):
    # Meta: source=RLVE, category=algorithmic, turn=single
    """RLVE matrix permutation with main diagonal constraint as a single-turn environment."""

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""You are given a square matrix of size {N} × {N}, where each element is either `0` or `1`. This matrix is 0-indexed.

Please find:
- a permutation of the row indices: a[0], ..., a[{N_minus_1}] (a reordering of `0` to `{N_minus_1}`),
- a permutation of the column indices: b[0], ..., b[{N_minus_1}] (a reordering of `0` to `{N_minus_1}`),
- such that after applying these permutations to the rows and columns of the matrix A (i.e., the element at position (i, j) becomes A[a[i]][b[j]]), the **main diagonal** of the resulting matrix contains only `1`s (main diagonal refers to the elements at position (i, i) for i from `0` to `{N_minus_1}`).

Matrix A is given as follows:
{A}

**Output Format:** Output two lines:
- The first line contains the row permutation: a[0] a[1] ... a[{N_minus_1}]
- The second line contains the column permutation: b[0] b[1] ... b[{N_minus_1}]
(Use spaces to separate adjacent integers. Do **not** include backticks or quotes.)"""

    def __init__(
        self,
        N: int = 4,
        cell_px: int = 64,
        padding: int = 24,
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._N = N
        self._cell_px = cell_px
        self._padding = padding
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._matrix: list[list[int]] | None = None
        self._prompt: str | None = None
        self._oracle_answer: str | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        if self._matrix:
            size = len(self._matrix)
            size_hint = f"{size} x {size}"
            n_minus_1 = size - 1
        else:
            size_hint = "N x N"
            n_minus_1 = "N-1"
        return dedent(
            f"""
            You are given a square binary matrix. Find permutations of row indices and
            column indices such that after applying these permutations, the main diagonal
            of the resulting matrix contains only 1s.

            Matrix Permutation Main Diagonal One rules:
            1) You are given a {size_hint} binary matrix (elements are 0 or 1).
            2) Find permutations of row indices and column indices.
            3) After applying these permutations, the main diagonal must contain all 1s.
            4) Main diagonal = elements at position (i, i) for i from 0 to N-1.

            In the image:
            - The matrix is displayed as a grid with binary values (0 or 1).
            - Main diagonal cells are highlighted with a distinctive color (gold/yellow).
            - Non-diagonal cells use different colors based on their values.
            - Blue tint = cells containing 1, lighter = cells containing 0.

            Output Format: Output two lines:
            - The first line contains the row permutation: a[0] a[1] ... a[{n_minus_1}]
            - The second line contains the column permutation: b[0] b[1] ... b[{n_minus_1}]
            (Use spaces to separate adjacent integers. Do not include backticks or quotes.)
            """
        ).strip()

    def _get_state_text(self) -> str:
        """Return the text representation of the current state."""
        return self._prompt if self._prompt else ""

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
            metadata={"state_text": state_text, "text_prompt": self._prompt},
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
            metadata={"state_text": state_text, "text_prompt": self._prompt},
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
        """Generate a random binary matrix with a valid solution."""
        N = self._N
        if N < 2:
            raise ValueError("N must be at least 2")

        # Generate matrix with random 0s and 1s
        one_probability = float(self.np_random.random()) / 2.0
        A = [
            [1 if self.np_random.random() < one_probability else 0 for _ in range(N)]
            for _ in range(N)
        ]

        # Create random permutations to ensure a valid solution exists
        row_permutation = list(range(N))
        self.np_random.shuffle(row_permutation)
        column_permutation = list(range(N))
        self.np_random.shuffle(column_permutation)

        # Place 1s on the diagonal according to the permutations
        for i in range(N):
            A[row_permutation[i]][column_permutation[i]] = 1

        self._matrix = A
        self._oracle_answer = (
            " ".join(map(str, row_permutation))
            + "\n"
            + " ".join(map(str, column_permutation))
        )

    def _prompt_generate(self) -> str:
        """Generate the prompt text from the matrix."""
        if self._matrix is None:
            raise RuntimeError("No matrix generated")
        N = len(self._matrix)
        return self.prompt_template.format(
            N=N,
            N_minus_1=N - 1,
            A="\n".join("".join(map(str, row)) for row in self._matrix),
        )

    def _process(self, answer: str | None) -> tuple[list[int], list[int]] | None:
        """Process the answer string into two permutations."""
        if answer is None:
            return None
        answer = answer.strip()
        try:
            permutations = []
            for line in answer.splitlines():
                line = line.strip()
                if line:
                    permutations.append(list(map(int, line.split())))
            if len(permutations) == 2:
                return permutations[0], permutations[1]
            else:
                return None
        except ValueError:
            return None

    def _score_answer(self, answer: str) -> float:
        """Score the answer using (satisfied/all)^beta strategy."""
        processed_result = self._process(answer)
        if processed_result is None:
            return 0.0
        row_permutation, column_permutation = processed_result
        N = len(self._matrix)

        # Validate permutations
        if not (len(row_permutation) == N and set(row_permutation) == set(range(N))):
            return 0.0
        if not (
            len(column_permutation) == N and set(column_permutation) == set(range(N))
        ):
            return 0.0
        B = [
            [self._matrix[row_permutation[i]][column_permutation[j]] for j in range(N)]
            for i in range(N)
        ]

        # Count how many diagonal elements are 1
        satisfied = sum(B[i][i] for i in range(N))

        # Use (satisfied/all)^beta strategy
        beta = 5.0
        return (satisfied / N) ** beta

    def render(self) -> Image.Image | list[Image.Image] | None:
        """Render the binary matrix with main diagonal highlighted."""
        if self._matrix is None:
            raise RuntimeError("No matrix generated")

        N = len(self._matrix)
        cell_px = self._cell_px
        padding = self._padding

        width = padding * 2 + N * cell_px
        height = padding * 2 + N * cell_px
        img = Image.new("RGB", (width, height), (250, 250, 250))
        draw = ImageDraw.Draw(img)

        font_path = None
        font_path_candidate = self.assets_dir / "DejaVuSans.ttf"
        if font_path_candidate.exists():
            font_path = str(font_path_candidate)

        if font_path:
            font = ImageFont.truetype(font_path, int(cell_px * 0.5))
        else:
            font = ImageFont.load_default()

        # Draw matrix cells with highlighting for main diagonal
        for r in range(N):
            for c in range(N):
                x0 = padding + c * cell_px
                y0 = padding + r * cell_px
                x1 = x0 + cell_px
                y1 = y0 + cell_px

                value = self._matrix[r][c]
                is_diagonal = r == c

                # Color scheme:
                # - Diagonal cells: gold/yellow background
                # - Non-diagonal with 1: light blue
                # - Non-diagonal with 0: light gray
                if is_diagonal:
                    # Main diagonal cells - gold/yellow gradient
                    if value == 1:
                        fill_color = (255, 215, 0)  # Bright gold
                    else:
                        fill_color = (255, 235, 120)  # Light gold
                else:
                    # Non-diagonal cells
                    if value == 1:
                        fill_color = (180, 200, 255)  # Light blue
                    else:
                        fill_color = (230, 230, 230)  # Light gray

                draw.rectangle([x0, y0, x1, y1], fill=fill_color)

        # Draw grid lines
        for r in range(N + 1):
            y = padding + r * cell_px
            # Thicker line for main diagonal boundaries
            width_line = 3
            draw.line(
                (padding, y, padding + N * cell_px, y),
                fill=(30, 30, 30),
                width=width_line,
            )
        for c in range(N + 1):
            x = padding + c * cell_px
            width_line = 3
            draw.line(
                (x, padding, x, padding + N * cell_px),
                fill=(30, 30, 30),
                width=width_line,
            )

        # Draw values (0 or 1) in each cell
        for r in range(N):
            for c in range(N):
                value = self._matrix[r][c]
                value_str = str(value)
                is_diagonal = r == c

                bbox = draw.textbbox((0, 0), value_str, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                cx = padding + c * cell_px + cell_px // 2
                cy = padding + r * cell_px + cell_px // 2

                # Text color - dark for visibility
                if is_diagonal:
                    text_color = (0, 0, 0)  # Black for diagonal
                else:
                    if value == 1:
                        text_color = (0, 0, 100)  # Dark blue
                    else:
                        text_color = (100, 100, 100)  # Gray

                draw.text(
                    (cx - tw // 2, cy - th // 2), value_str, fill=text_color, font=font
                )

        # Draw a subtle overlay on the diagonal to further emphasize it
        for i in range(N):
            x0 = padding + i * cell_px
            y0 = padding + i * cell_px
            x1 = x0 + cell_px
            y1 = y0 + cell_px
            # Draw a border around diagonal cells
            draw.rectangle([x0, y0, x1, y1], outline=(200, 160, 0), width=4)

        return img

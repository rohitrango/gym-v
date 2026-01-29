"""Matrix pooling environment for gym-v (self-contained)."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation
from gym_v.logger import get_logger

logger = get_logger()


class RLVEMatrixPoolingEnv(Env):
    """RLVE matrix pooling as a single-turn environment."""

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""You are given a matrix of size {N} × {M}. Perform a **max pooling** operation with a kernel size of {K} × {K}. In max pooling, each output cell contains the **maximum value** in the corresponding {K} × {K} submatrix of the input.

The matrix is:
{matrix}

**Output Format:** Your output should contain {output_rows} lines, each with {output_cols} integers separated by **spaces**. Each integer represents the maximum value in the respective pooling region."""

    def __init__(
        self,
        max_n_m: int = 7,
        cell_px: int = 56,
        padding: int = 24,
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._max_n_m = max_n_m
        self._cell_px = cell_px
        self._padding = padding
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._matrix: list[list[int]] | None = None
        self._n: int = 0
        self._m: int = 0
        self._k: int = 0
        self._prompt: str | None = None
        self._oracle_answer: list[list[int]] | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        if self._matrix:
            rows = len(self._matrix)
            cols = len(self._matrix[0]) if self._matrix else 0
            k = self._k
            size_hint = f"{rows} x {cols} matrix with {k}x{k} kernel"
        else:
            size_hint = "N x M matrix with KxK kernel"
        return dedent(
            """
            Matrix Pooling rules:
            1) Given a matrix of size N × M with integer values.
            2) Apply max pooling with a kernel size of K × K.
            3) Each output cell contains the maximum value from a K × K submatrix.
            4) The output matrix has dimensions (N-K+1) × (M-K+1).
            5) Pooling windows slide across the matrix with stride 1.

            In the image:
            - The full input matrix is shown with color-coded cells (heatmap style).
            - Brighter/warmer colors indicate higher values.
            - Red borders highlight the K×K pooling windows at each position.
            - The pooling operation type (MAX) is displayed in the title.

            Output format: (N-K+1) lines, each with (M-K+1) space-separated integers representing the max values in each pooling region.
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
            "oracle_answer": "\n".join(
                " ".join(map(str, row)) for row in self._oracle_answer
            ),
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
            "oracle_answer": "\n".join(
                " ".join(map(str, row)) for row in self._oracle_answer
            ),
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
        """Generate a random matrix and compute max pooling result."""
        max_n_m = self._max_n_m
        if max_n_m < 3:
            raise ValueError("max_n_m must be >= 3")

        N = int(self.np_random.integers(3, max_n_m + 1))
        M = int(self.np_random.integers(3, max_n_m + 1))
        K = int(self.np_random.integers(2, min(N, M)))

        self._n = N
        self._m = M
        self._k = K

        # Generate random matrix
        matrix = [
            [int(self.np_random.integers(0, N * M + 1)) for _ in range(M)]
            for _ in range(N)
        ]
        self._matrix = matrix

        # Compute max pooling
        gold_answer = [
            [
                max(matrix[i + di][j + dj] for di in range(K) for dj in range(K))
                for j in range(M - K + 1)
            ]
            for i in range(N - K + 1)
        ]
        self._oracle_answer = gold_answer

    def _prompt_generate(self) -> str:
        if self._matrix is None:
            raise RuntimeError("No matrix generated")
        output_rows = self._n - self._k + 1
        output_cols = self._m - self._k + 1
        return self.prompt_template.format(
            N=self._n,
            M=self._m,
            K=self._k,
            matrix="\n".join(" ".join(map(str, row)) for row in self._matrix),
            output_rows=output_rows,
            output_cols=output_cols,
        )

    def _process(self, answer: str | None) -> list[list[int]] | None:
        """Process the answer string into a 2D list of integers."""
        if answer is None:
            return None
        answer = answer.strip()
        try:
            matrix = []
            for line in answer.splitlines():
                line = line.strip()
                if line:
                    matrix.append(list(map(int, line.split())))
            return matrix
        except ValueError:
            return None

    def _score_answer(self, answer: str) -> float:
        """Score the answer using mean([gold=answer])^beta strategy."""
        processed_result = self._process(answer)
        if processed_result is None:
            return 0.0
        pool = processed_result
        expected_rows = self._n - self._k + 1
        expected_cols = self._m - self._k + 1

        if len(pool) != expected_rows:
            return 0.0
        if not all(len(row) == expected_cols for row in pool):
            return 0.0
        beta = 5.0
        correct_count = sum(
            sum(
                answer == gold
                for answer, gold in zip(answer_row, gold_row, strict=False)
            )
            for answer_row, gold_row in zip(pool, self._oracle_answer, strict=False)
        )
        total_count = expected_rows * expected_cols
        return (correct_count / total_count) ** beta

    def render(self) -> Image.Image | list[Image.Image] | None:
        """Render the matrix with pooling windows highlighted."""
        if self._matrix is None:
            raise RuntimeError("No matrix generated")

        rows, cols = len(self._matrix), len(self._matrix[0])
        k = self._k
        cell_px = self._cell_px
        padding = self._padding
        title_height = 60

        width = padding * 2 + cols * cell_px
        height = padding * 2 + rows * cell_px + title_height
        img = Image.new("RGB", (width, height), (250, 250, 250))
        draw = ImageDraw.Draw(img)

        font_path = None
        font_path_candidate = self.assets_dir / "DejaVuSans.ttf"
        if font_path_candidate.exists():
            font_path = str(font_path_candidate)

        if font_path:
            font = ImageFont.truetype(font_path, int(cell_px * 0.4))
            title_font = ImageFont.truetype(font_path, 28)
        else:
            font = ImageFont.load_default()
            title_font = ImageFont.load_default()

        # Draw title
        title = f"Matrix Pooling (MAX) - {k}x{k} Kernel"
        title_bbox = draw.textbbox((0, 0), title, font=title_font)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = (width - title_width) // 2
        draw.text((title_x, 15), title, fill=(30, 30, 30), font=title_font)

        # Calculate value range for heatmap coloring
        min_val = min(min(row) for row in self._matrix)
        max_val = max(max(row) for row in self._matrix)
        val_range = max_val - min_val if max_val > min_val else 1

        # Draw matrix cells with heatmap colors
        for r in range(rows):
            for c in range(cols):
                x0 = padding + c * cell_px
                y0 = padding + r * cell_px + title_height
                x1 = x0 + cell_px
                y1 = y0 + cell_px

                # Heatmap color based on value
                value = self._matrix[r][c]
                normalized = (value - min_val) / val_range

                # Color gradient: blue (low) -> yellow (mid) -> red (high)
                if normalized < 0.5:
                    # Blue to yellow
                    t = normalized * 2
                    r_val = int(100 + t * 155)
                    g_val = int(150 + t * 105)
                    b_val = int(255 - t * 155)
                else:
                    # Yellow to red
                    t = (normalized - 0.5) * 2
                    r_val = 255
                    g_val = int(255 - t * 100)
                    b_val = int(100 - t * 100)

                fill_color = (r_val, g_val, b_val)
                draw.rectangle([x0, y0, x1, y1], fill=fill_color)

        # Draw pooling windows with red borders
        output_rows = rows - k + 1
        output_cols = cols - k + 1
        for pr in range(output_rows):
            for pc in range(output_cols):
                # Draw pooling window border
                x0 = padding + pc * cell_px
                y0 = padding + pr * cell_px + title_height
                x1 = x0 + k * cell_px
                y1 = y0 + k * cell_px

                # Red border for pooling window
                draw.rectangle([x0, y0, x1, y1], outline=(220, 20, 20), width=3)

        # Draw grid lines
        for r in range(rows + 1):
            y = padding + r * cell_px + title_height
            draw.line(
                (padding, y, padding + cols * cell_px, y),
                fill=(80, 80, 80),
                width=1,
            )
        for c in range(cols + 1):
            x = padding + c * cell_px
            draw.line(
                (x, padding + title_height, x, padding + rows * cell_px + title_height),
                fill=(80, 80, 80),
                width=1,
            )

        # Draw values in each cell
        for r in range(rows):
            for c in range(cols):
                value = self._matrix[r][c]
                value_str = str(value)
                bbox = draw.textbbox((0, 0), value_str, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                cx = padding + c * cell_px + cell_px // 2
                cy = padding + r * cell_px + cell_px // 2 + title_height

                # Determine text color based on background brightness
                normalized = (value - min_val) / val_range
                text_color = (255, 255, 255) if normalized > 0.5 else (0, 0, 0)

                # Draw text with shadow for better visibility
                if normalized > 0.5:
                    draw.text(
                        (cx - tw // 2 + 1, cy - th // 2 + 1),
                        value_str,
                        fill=(0, 0, 0),
                        font=font,
                    )
                draw.text(
                    (cx - tw // 2, cy - th // 2), value_str, fill=text_color, font=font
                )

        return img

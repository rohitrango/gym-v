"""Matrix RMQ (Range Maximum Query) counting environment for gym-v (self-contained)."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation
from gym_v.logger import get_logger

logger = get_logger()


class RLVEMatrixRmqCountingEnv(Env):
    """RLVE matrix RMQ counting as a single-turn environment."""

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""Count the number of matrices `A` of size {H} × {W} (1-indexed, meaning row indices range from 1 to {H} and column indices from 1 to {W}) such that:
1. Each element of `A` is an integer between 1 and {M}, inclusive.
2. The matrix satisfies the following {N} constraints, where `max(A[x1 : x2 + 1, y1 : y2 + 1])` denotes the maximum value in the contiguous submatrix defined by the corners (x1, y1) and (x2, y2) (inclusive):
{constraints}

Output a single integer — the number of such matrices modulo {MOD}."""

    def __init__(
        self,
        H_W_range: int = 2,
        max_MOD: int = 1000000,
        cell_px: int = 48,
        padding: int = 24,
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._H_W_range = H_W_range
        self._max_MOD = max_MOD
        self._cell_px = cell_px
        self._padding = padding
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._H: int = 0
        self._W: int = 0
        self._M: int = 0
        self._N: int = 0
        self._MOD: int = 0
        self._constraints: list[tuple[int, int, int, int, int]] = []
        self._prompt: str | None = None
        self._oracle_answer: int | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        size_hint = f"{self._H} x {self._W}" if self._H > 0 else "H x W"
        return dedent(
            f"""
            Matrix RMQ (Range Maximum Query) Counting rules:
            1) A matrix of size {size_hint} must be filled with integers between 1 and M.
            2) Given N constraints specifying rectangular regions and their maximum values.
            3) Each constraint specifies: max(A[x1:x2+1, y1:y2+1]) = v for some region.
            4) Count how many valid matrices satisfy ALL constraints.
            5) Output is computed modulo a given MOD value.

            In the image:
            - The matrix is shown as a heatmap with color intensity representing value range.
            - Query regions (constraints) are highlighted with colored borders.
            - Each region shows the required maximum value.
            - Warmer colors (red/orange) indicate higher values, cooler colors (blue) indicate lower values.

            Output format: A single integer — the number of valid matrices modulo MOD.
            """
        ).strip()

    def _get_state_text(self) -> str:
        """Return the text representation of the current state."""
        return self._prompt or ""

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        super().reset(seed=seed)

        # Get N from options
        if options and "N" in options:
            self._N = options["N"]
        else:
            self._N = 2  # Default value

        self._generate()
        self._prompt = self._prompt_generate()
        self._last_image = self.render()

        state_text = self._get_state_text()
        obs = Observation(
            image=self._last_image,
            text=state_text,
            metadata={"text_prompt": self._prompt},
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
            metadata={"text_prompt": self._prompt},
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
        """Generate random matrix RMQ problem using self.np_random."""
        N = self._N
        if N < 2:
            N = 2

        H = int(self.np_random.integers(1, N * self._H_W_range + 1))
        W = int(self.np_random.integers(1, N * self._H_W_range + 1))
        M = int(self.np_random.integers(1, (N * self._H_W_range) ** 2 + 1))

        self._H = H
        self._W = W
        self._M = M

        # Generate a sample matrix to derive constraints
        A = [
            [int(self.np_random.integers(1, M + 1)) for _ in range(W)] for _ in range(H)
        ]

        constraints = []
        for _ in range(N):
            row_length = int(self.np_random.integers(1, H + 1))
            col_length = int(self.np_random.integers(1, W + 1))
            x1 = int(self.np_random.integers(1, H - row_length + 2))
            y1 = int(self.np_random.integers(1, W - col_length + 2))
            x2, y2 = x1 + row_length - 1, y1 + col_length - 1
            v = max(
                A[i - 1][j - 1] for i in range(x1, x2 + 1) for j in range(y1, y2 + 1)
            )
            constraints.append((x1, y1, x2, y2, v))

        self._constraints = constraints
        self._MOD = int(self.np_random.integers(2, self._max_MOD + 1))

        # Compute reference answer
        self._oracle_answer = self._compute_answer()

    def _compute_answer(self) -> int:
        """Compute the number of valid matrices using inclusion-exclusion."""
        H = self._H
        W = self._W
        M = self._M
        N = self._N
        MOD = self._MOD
        constraints = self._constraints

        pos = []
        X = [1, H + 1]
        Y = [1, W + 1]

        # Collect coordinates for compression
        for x1, y1, x2, y2, v in constraints:
            # include x2+1, y2+1 as open intervals
            pos.append((x1, y1, x2 + 1, y2 + 1, v))
            X.append(x1)
            X.append(x2 + 1)
            Y.append(y1)
            Y.append(y2 + 1)

        # Coordinate compression
        X = sorted(set(X))
        Y = sorted(set(Y))
        xi = {x: i for i, x in enumerate(X)}
        yi = {y: i for i, y in enumerate(Y)}

        # Precompute block ranges for each constraint
        ranges = []
        for x1, y1, x2p, y2p, v in pos:
            xl = xi[x1]
            xr = xi[x2p]
            yl = yi[y1]
            yr = yi[y2p]
            ranges.append((xl, xr, yl, yr, v))

        # Number of blocks in compressed grid
        Wb = len(X) - 1
        Hb = len(Y) - 1
        ans = 0

        # Inclusion-exclusion over subsets of constraints
        for mask in range(1 << N):
            # Initialize each block's max allowed value to M
            arr = [[M] * Hb for _ in range(Wb)]
            # Apply each constraint, reducing allowed max by 1 if in the subset
            for j in range(N):
                bit = (mask >> j) & 1
                xl, xr, yl, yr, v = ranges[j]
                limit = v - bit
                for xi_ in range(xl, xr):
                    row = arr[xi_]
                    for yi_ in range(yl, yr):
                        if row[yi_] > limit:
                            row[yi_] = limit

            # Compute number of fillings for this configuration
            tmp = 1
            for xi_ in range(Wb):
                dx = X[xi_ + 1] - X[xi_]
                for yi_ in range(Hb):
                    dy = Y[yi_ + 1] - Y[yi_]
                    area = dx * dy
                    val = arr[xi_][yi_]
                    # pow handles zero and mod efficiently
                    tmp = tmp * pow(val, area, MOD) % MOD
                    if tmp == 0:
                        break
                if tmp == 0:
                    break

            # Inclusion-exclusion sign
            if bin(mask).count("1") & 1:
                ans = (ans - tmp) % MOD
            else:
                ans = (ans + tmp) % MOD

        return ans

    def _prompt_generate(self) -> str:
        """Generate the prompt string."""
        return self.prompt_template.format(
            H=self._H,
            W=self._W,
            M=self._M,
            N=self._N,
            constraints="\n".join(
                f"max(A[{x1} : {x2} + 1, {y1} : {y2} + 1]) = {v}"
                for x1, y1, x2, y2, v in self._constraints
            ),
            MOD=self._MOD,
        )

    def _process(self, answer: str | None) -> int | None:
        """Process the answer string into an integer."""
        if answer is None:
            return None
        answer = answer.strip()
        try:
            int_answer = int(answer)
            return int_answer
        except ValueError:
            return None

    def _score_answer(self, answer: str) -> float:
        """Score the answer."""
        processed_result = self._process(answer)
        if processed_result is None:
            return 0.0
        if not (0 <= processed_result < self._MOD):
            return 0.0
        if processed_result == self._oracle_answer:
            return 1.0
        else:
            return 0.0

    def render(self) -> Image.Image | list[Image.Image] | None:
        """Render the matrix with heatmap and highlighted query regions."""
        if self._H == 0 or self._W == 0:
            raise RuntimeError("No matrix generated")

        H, W = self._H, self._W
        M = self._M
        cell_px = self._cell_px
        padding = self._padding

        # Add extra space for labels
        label_space = 40
        width = padding * 2 + W * cell_px + label_space
        height = padding * 2 + H * cell_px + label_space
        img = Image.new("RGB", (width, height), (245, 245, 245))
        draw = ImageDraw.Draw(img)

        font_path = None
        font_path_candidate = self.assets_dir / "DejaVuSans.ttf"
        if font_path_candidate.exists():
            font_path = str(font_path_candidate)

        if font_path:
            font = ImageFont.truetype(font_path, int(cell_px * 0.25))
            font_small = ImageFont.truetype(font_path, int(cell_px * 0.18))
        else:
            font = ImageFont.load_default()
            font_small = ImageFont.load_default()

        # Create a heatmap based on value distribution (we'll use conceptual values)
        # Since we don't have actual values, we'll create a gradient based on constraints
        for r in range(H):
            for c in range(W):
                x0 = padding + c * cell_px
                y0 = padding + r * cell_px
                x1 = x0 + cell_px
                y1 = y0 + cell_px

                # Create gradient based on position
                # This creates a visual representation even without actual values
                intensity = (r * W + c) / (H * W)

                # Blue to red gradient
                red = int(100 + intensity * 120)
                green = int(150 - intensity * 80)
                blue = int(230 - intensity * 150)

                fill_color = (red, green, blue)
                draw.rectangle(
                    [x0, y0, x1, y1], fill=fill_color, outline=(200, 200, 200), width=1
                )

        # Draw constraint regions with colored borders
        colors = [
            (255, 0, 0),  # Red
            (0, 150, 255),  # Blue
            (0, 200, 0),  # Green
            (255, 140, 0),  # Orange
            (200, 0, 200),  # Purple
            (0, 200, 200),  # Cyan
        ]

        for idx, (x1, y1, x2, y2, v) in enumerate(self._constraints):
            color = colors[idx % len(colors)]

            # Convert 1-indexed to 0-indexed for drawing
            r1, c1 = x1 - 1, y1 - 1
            r2, c2 = x2 - 1, y2 - 1

            # Draw thick border around the constraint region
            border_x0 = padding + c1 * cell_px - 2
            border_y0 = padding + r1 * cell_px - 2
            border_x1 = padding + (c2 + 1) * cell_px + 2
            border_y1 = padding + (r2 + 1) * cell_px + 2

            # Draw thick colored border
            draw.rectangle(
                [border_x0, border_y0, border_x1, border_y1], outline=color, width=4
            )

            # Draw label with max value
            label = f"max={v}"
            bbox = draw.textbbox((0, 0), label, font=font_small)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

            # Position label at top-left of region
            label_x = border_x0 + 4
            label_y = border_y0 - th - 4
            if label_y < 5:
                label_y = border_y0 + 2

            # Draw background for label
            draw.rectangle(
                [label_x - 2, label_y - 2, label_x + tw + 2, label_y + th + 2],
                fill=(255, 255, 255),
                outline=color,
                width=1,
            )
            draw.text((label_x, label_y), label, fill=color, font=font_small)

        # Draw grid lines
        for r in range(H + 1):
            y = padding + r * cell_px
            draw.line(
                (padding, y, padding + W * cell_px, y), fill=(100, 100, 100), width=1
            )
        for c in range(W + 1):
            x = padding + c * cell_px
            draw.line(
                (x, padding, x, padding + H * cell_px), fill=(100, 100, 100), width=1
            )

        # Draw axis labels
        # Column labels (1-indexed)
        for c in range(W):
            label = str(c + 1)
            bbox = draw.textbbox((0, 0), label, font=font_small)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            cx = padding + c * cell_px + cell_px // 2
            cy = padding + H * cell_px + 8
            draw.text((cx - tw // 2, cy), label, fill=(80, 80, 80), font=font_small)

        # Row labels (1-indexed)
        for r in range(H):
            label = str(r + 1)
            bbox = draw.textbbox((0, 0), label, font=font_small)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            cx = padding + W * cell_px + 8
            cy = padding + r * cell_px + cell_px // 2
            draw.text((cx, cy - th // 2), label, fill=(80, 80, 80), font=font_small)

        # Draw title
        title = f"Matrix {H}×{W}, M={M}, {len(self._constraints)} constraints"
        bbox = draw.textbbox((0, 0), title, font=font_small)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((padding, 5), title, fill=(40, 40, 40), font=font_small)

        return img

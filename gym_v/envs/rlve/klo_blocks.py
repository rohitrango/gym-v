"""Klo blocks environment for gym-v (self-contained)."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation
from gym_v.logger import get_logger

logger = get_logger()


class RLVEKloBlocksEnv(Env):
    """RLVE Klo Blocks as a single-turn environment.

    Source: https://www.luogu.com.cn/problem/P3503

    The environment presents a block manipulation problem where you can transfer
    values between adjacent blocks to maximize a contiguous subarray.
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""You have an array A of {N} integers, initially it is: {A}
You can perform any number of actions. One action is to pick one item that is **greater than** {K}, subtract 1 from it, and add 1 to an **adjacent** item (either to the left or right, if such an item exists).
Please maximize the length of the longest contiguous subarray where each item is **greater than or equal to** {K}; output its length."""

    def __init__(
        self,
        N: int = 10,
        cell_px: int = 60,
        padding: int = 30,
        num_players: int = 1,
        **kwargs: Any,
    ):
        """Initialize the KloBlocks environment.

        Args:
            N: Size of the array (must be >= 3).
            cell_px: Cell width in pixels for rendering.
            padding: Padding around the rendered image.
            num_players: Number of players (default 1).
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)
        self._N = N
        self._cell_px = cell_px
        self._padding = padding
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._A: list[int] | None = None
        self._K: int | None = None
        self._prompt: str | None = None
        self._oracle_answer: int | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        if self._A is not None and self._K is not None:
            size_hint = f"N={len(self._A)}, K={self._K}"
        else:
            size_hint = "N=?, K=?"
        return dedent(
            f"""
            Klo Blocks (Block Manipulation Problem):

            You have an array A of N integers. You can perform actions where you:
            1. Pick an item with value > K
            2. Subtract 1 from it
            3. Add 1 to an adjacent item (left or right)

            Goal: Maximize the length of the longest contiguous subarray where
            each item is >= K.

            Current parameters: {size_hint}

            The image shows:
            - The initial array A
            - The threshold K indicated by a red line
            - Blocks colored based on their relationship to K:
              * Green: value >= K
              * Orange: value < K

            Output format: A single integer representing the maximum length.
            """
        ).strip()

    def _get_state_text(self) -> str:
        """Return the text representation of the current state."""
        if self._A is None or self._K is None:
            raise RuntimeError("No problem generated")
        return self.prompt_template.format(
            N=len(self._A),
            A=" ".join(map(str, self._A)),
            K=self._K,
        )

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
        """Generate a klo_blocks problem instance.

        Uses the algorithm from RLVE to find the maximum length of contiguous
        subarray where all values can be made >= K through allowed operations.
        """
        N = self._N
        if N < 3:
            raise ValueError("N should be greater than or equal to 3")

        while True:
            A = [int(self.np_random.integers(1, 2 * N + 1)) for _ in range(N)]
            min_A, max_A = min(A), max(A)
            if not (min_A + 1 <= max_A - 1):
                continue
            K = int(self.np_random.integers(min_A + 1, max_A))

            # b[0] = 0, b[i] = prefix sum of (A[j] - K) up to j = i
            b = [0] * (N + 1)
            stack = []  # will store indices with strictly decreasing b-values
            ans = 0

            # Forward pass: build b[], track any prefix >= 0 and build monotonic stack
            for i in range(1, N + 1):
                b[i] = b[i - 1] + A[i - 1] - K
                if b[i] >= 0:
                    # we can take the whole prefix 1..i
                    ans = i
                # maintain stack of indices where b is strictly decreasing
                if not stack or b[i] < b[stack[-1]]:
                    stack.append(i)

            # Backward pass: match later indices i with earlier minima in stack
            for i in range(N, 0, -1):
                # while we can form a non-negative sum from stack[-1]+1 .. i
                while stack and b[i] - b[stack[-1]] >= 0:
                    ans = max(ans, i - stack[-1])
                    stack.pop()

            if ans != 1 and ans != N:
                self._A = A
                self._K = K
                self._oracle_answer = ans
                break

    def _prompt_generate(self) -> str:
        """Generate the problem prompt."""
        if self._A is None or self._K is None:
            raise RuntimeError("No problem generated")
        return self.prompt_template.format(
            N=len(self._A),
            A=" ".join(map(str, self._A)),
            K=self._K,
        )

    def _process(self, answer: str | None) -> int | None:
        """Process the answer string into an integer.

        Args:
            answer: The answer string from the agent.

        Returns:
            The parsed integer or None if parsing fails.
        """
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
        """Score the answer.

        Returns:
            1.0 for correct answer, 0.0 for wrong answer, -1.0 for wrong format.
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
        """Render the block array as an image.

        Creates a beautiful visualization showing:
        - Each block with its value
        - Blocks colored based on whether they are >= K (green) or < K (orange)
        - A horizontal red line indicating the threshold K
        - Clear spacing and professional styling
        """
        if self._A is None or self._K is None:
            raise RuntimeError("No problem generated")

        N = len(self._A)
        cell_px = self._cell_px
        padding = self._padding
        K = self._K

        # Calculate dimensions
        width = padding * 2 + N * cell_px
        max_val = max(self._A)
        min_val = min(self._A)
        value_range = max_val - min_val if max_val > min_val else 1

        # Height includes space for the tallest block plus some headroom
        block_height = cell_px * 4
        height = padding * 3 + block_height

        # Create image
        img = Image.new("RGB", (width, height), (245, 245, 250))
        draw = ImageDraw.Draw(img)

        # Load font
        font_path = None
        font_path_candidate = self.assets_dir / "DejaVuSans.ttf"
        if font_path_candidate.exists():
            font_path = str(font_path_candidate)

        if font_path:
            font = ImageFont.truetype(font_path, int(cell_px * 0.3))
            font_small = ImageFont.truetype(font_path, int(cell_px * 0.25))
        else:
            font = ImageFont.load_default()
            font_small = font

        # Calculate baseline for blocks (where K line should be)
        baseline_y = padding + block_height - padding

        # Draw each block
        for i, val in enumerate(self._A):
            x_left = padding + i * cell_px
            x_right = x_left + cell_px - 2

            # Calculate block height based on value
            # Normalize to fit within the available space
            normalized_height = ((val - min_val) / value_range) * (
                block_height - 2 * padding
            )
            if value_range == 1:
                normalized_height = block_height // 2

            y_top = baseline_y - normalized_height
            y_bottom = baseline_y

            # Color based on whether value is >= K
            if val >= K:
                color = (100, 200, 100)  # Green
            else:
                color = (255, 165, 80)  # Orange

            # Draw the block
            draw.rectangle(
                [x_left + 3, y_top, x_right - 3, y_bottom],
                fill=color,
                outline=(50, 50, 50),
                width=2,
            )

            # Draw the value on the block
            text = str(val)
            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            cx = (x_left + x_right) // 2
            cy = (y_top + y_bottom) // 2
            draw.text((cx - tw // 2, cy - th // 2), text, fill=(10, 10, 10), font=font)

            # Draw index below
            idx_text = str(i)
            bbox = draw.textbbox((0, 0), idx_text, font=font_small)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text(
                (cx - tw // 2, baseline_y + 5),
                idx_text,
                fill=(80, 80, 80),
                font=font_small,
            )

        # Draw threshold line for K
        k_y = baseline_y - ((K - min_val) / value_range) * (block_height - 2 * padding)
        if value_range == 1:
            k_y = baseline_y - block_height // 2

        draw.line(
            [(padding - 10, k_y), (padding + N * cell_px + 10, k_y)],
            fill=(220, 50, 50),
            width=3,
        )

        # Draw K label
        k_label = f"K = {K}"
        bbox = draw.textbbox((0, 0), k_label, font=font_small)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle(
            [
                padding + N * cell_px + 15 - 3,
                k_y - th // 2 - 3,
                padding + N * cell_px + 15 + tw + 3,
                k_y + th // 2 + 3,
            ],
            fill=(245, 245, 250),
        )
        draw.text(
            (padding + N * cell_px + 15, k_y - th // 2),
            k_label,
            fill=(220, 50, 50),
            font=font_small,
        )

        return img

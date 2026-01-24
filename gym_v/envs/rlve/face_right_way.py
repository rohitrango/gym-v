"""Face Right Way environment for gym-v (self-contained)."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation
from gym_v.logger import get_logger

logger = get_logger()


class RLVEFaceRightWayEnv(Env):
    """RLVE Face Right Way as a single-turn environment.

    Given a 0/1 array representing arrows (0=right, 1=left), find the minimum
    number of flip operations M and the minimum window size K needed to make all
    arrows face right (all zeros). Each operation flips K consecutive elements.
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""There is a 0/1 array A of length {N}, and initially it is: {A}

Please do the following:
- First, pick a positive integer K, which must remain fixed throughout the process.
- Then, perform M operations. In each operation, you choose an index l (1 ≤ l ≤ {N} - K + 1) and flip all values A[i] with l ≤ i < l + K (i.e., a contiguous subarray of length K).
- Finally, all elements of A must become 0.

Your goal is:
1. Minimize M (the total number of operations).
2. Among all strategies with minimal M, minimize K.

**Output Format:** Output M lines, each containing two integers l and l + K - 1 (separated by a space), representing the closed interval [l, l + K - 1] flipped in that operation. All intervals must have the same length K."""

    def __init__(
        self,
        max_n: int = 10,
        cell_px: int = 60,
        padding: int = 24,
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._max_n = max_n
        self._cell_px = cell_px
        self._padding = padding
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._n: int | None = None
        self._array: list[int] | None = None
        self._prompt: str | None = None
        self._reference_answer: str | None = None
        self._gold_answer: dict[str, int] | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        if self._n:
            size_hint = f"{self._n} elements"
        else:
            size_hint = "N elements"

        return dedent(
            f"""
            Face Right Way Problem:

            Given a binary array of length {size_hint} representing arrow directions:
            - 0 = right-facing arrow (→)
            - 1 = left-facing arrow (←)

            Goal: Make all arrows face right (all 0s) using minimal operations.

            Rules:
            1) Choose a fixed window size K (must remain constant)
            2) Each operation flips K consecutive elements
            3) Minimize M (total number of operations)
            4) Among all solutions with minimal M, minimize K

            In the visualization:
            - Top row shows the initial array state
            - Arrows are displayed left-to-right with clear directional indicators
            - Right-facing arrows (→) shown in green when 0
            - Left-facing arrows (←) shown in red when 1
            - Visual representation helps identify flip patterns

            Output format: M lines, each with two space-separated integers "l r"
            representing the closed interval [l, r] to flip (1-indexed).
            All intervals must have the same length K.

            Example output:
            1 3
            4 6
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
                "rlve_reference_answer": self._reference_answer,
            },
        )
        info = {
            "reference_answer": self._reference_answer,
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
                "rlve_reference_answer": self._reference_answer,
            },
        )
        info = {
            "reference_answer": self._reference_answer,
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
        """Generate a face-right-way problem instance.

        Ports generation logic from RLVE using self.np_random.
        """
        N = self._max_n
        if N < 4:
            raise ValueError("max_n must be >= 4")

        A = [0] * N
        K = int(self.np_random.integers(2, N + 1))

        left_endpoints = list(range(0, N - K + 1))
        num_flips = int(self.np_random.integers(1, len(left_endpoints) + 1))
        selected_endpoints = self.np_random.choice(
            left_endpoints, size=num_flips, replace=False
        )

        for l in selected_endpoints:
            for i in range(l, l + K):
                A[i] ^= 1

        # Ensure array is not all zeros
        if not any(A):
            # If all zeros, flip one position
            A[0] = 1

        # Greedy solution with K=1
        ansK = 1
        ansM = sum(A)
        self._reference_answer = "\n".join(
            f"{i} {i}" for i, Ai in enumerate(A, start=1) if Ai
        )

        A_indexed = [None] + A  # 1-indexed

        # Try every K and compute the minimal number of flips M for that K in O(N)
        for K in range(1, N + 1):
            flip = [0] * (N + 1)  # flip[i] == 1 if we start a flip at position i
            curr = 0  # parity of active flips affecting current position
            m = 0
            possible = True

            current_answer = ""

            for i in range(1, N + 1):
                # Remove the effect of a flip that ends before i
                if i - K >= 1:
                    curr ^= flip[i - K]

                # After applying current parity, do we still see a '1' at i?
                need_flip = A_indexed[i] ^ (curr == 1)
                if need_flip:
                    # Can't start a K-flip if it would exceed N
                    if i + K - 1 > N:
                        possible = False
                        break
                    current_answer += f"{i} {i + K - 1}\n"
                    flip[i] = 1
                    curr ^= 1
                    m += 1

            if possible and m < ansM:
                ansM = m
                ansK = K
                self._reference_answer = current_answer.strip()

        self._n = N
        self._array = A
        self._gold_answer = {"K": ansK, "M": ansM}

    def _prompt_generate(self) -> str:
        """Generate the prompt text for the problem."""
        if self._n is None:
            raise RuntimeError("No problem generated")
        return self.prompt_template.format(
            N=self._n,
            A="; ".join(f"A[{i}]={Ai}" for i, Ai in enumerate(self._array, start=1)),
        )

    def _process(self, answer: str | None) -> list[tuple[int, int]] | None:
        """Process the answer string into a list of intervals."""
        if answer is not None:
            answer = answer.strip()
            try:
                operations = []
                for line in answer.splitlines():
                    line = line.strip()
                    if line:
                        l, r = map(int, line.split())
                        operations.append((l, r))
                return operations
            except Exception:
                return None
        else:
            return None

    def _score_answer(self, answer: str) -> float:
        """Score the answer based on correctness and optimality.

        Returns:
            -1.0: wrong format
            -0.5: invalid solution (out of bounds or inconsistent K)
            -0.2: unsuccessful solution (doesn't make all arrows face right)
            0.0 to 1.0: partial credit based on M and K quality
        """
        processed_result = self._process(answer)
        if processed_result is None:
            return -1.0

        A = self._array.copy()

        K = None
        for l, r in processed_result:
            if not (1 <= l <= r <= self._n):
                return -0.5
            if K is None:
                K = r - l + 1
            if K != r - l + 1:
                return -0.5
            for i in range(l, r + 1):
                A[i - 1] ^= 1

        if any(A):
            return -0.2

        reward = 0.0

        answer_M, gold_M = len(processed_result), self._gold_answer["M"]
        if gold_M > answer_M:
            # This should not happen in valid solutions
            return -0.2

        # Reward strategy for M: (gold/answer)^beta
        rewarding_weight_M = 0.5
        rewarding_beta_M = 5.0
        reward += rewarding_weight_M * ((gold_M / answer_M) ** rewarding_beta_M)

        # Only reward K if M is optimal
        if gold_M == answer_M:
            answer_K, gold_K = K, self._gold_answer["K"]
            if gold_K > answer_K:
                # This should not happen in valid solutions
                return -0.2

            # Reward strategy for K: (gold/answer)^beta
            rewarding_weight_K = 0.5
            rewarding_beta_K = 5.0
            reward += rewarding_weight_K * ((gold_K / answer_K) ** rewarding_beta_K)

        return reward

    def render(self) -> Image.Image | list[Image.Image] | None:
        """Render the arrow array as an image.

        Shows:
        - Initial array state with arrows
        - Clear visual distinction between left and right arrows
        - Color coding: green for right-facing (0), red for left-facing (1)
        """
        if self._n is None or self._array is None:
            raise RuntimeError("No problem generated")

        cell_px = self._cell_px
        padding = self._padding

        n = self._n

        # Calculate dimensions
        title_height = 70
        array_height = cell_px + 40
        legend_height = 80

        width = padding * 2 + n * cell_px
        height = padding * 2 + title_height + array_height + legend_height

        img = Image.new("RGB", (width, height), (250, 250, 250))
        draw = ImageDraw.Draw(img)

        # Load font
        font_path = None
        font_path_candidate = self.assets_dir / "DejaVuSans.ttf"
        if font_path_candidate.exists():
            font_path = str(font_path_candidate)

        if font_path:
            font_large = ImageFont.truetype(font_path, 24)
            font_medium = ImageFont.truetype(font_path, 18)
            font_small = ImageFont.truetype(font_path, 14)
            font_arrow = ImageFont.truetype(font_path, int(cell_px * 0.5))
        else:
            font_large = ImageFont.load_default()
            font_medium = ImageFont.load_default()
            font_small = ImageFont.load_default()
            font_arrow = ImageFont.load_default()

        # Draw title
        title = "Face Right Way - Arrow Flipping"
        bbox = draw.textbbox((0, 0), title, font=font_large)
        tw = bbox[2] - bbox[0]
        draw.text(
            (width // 2 - tw // 2, padding),
            title,
            fill=(40, 90, 140),
            font=font_large,
        )

        # Draw subtitle
        subtitle = "Make all arrows face right (→) using minimal flip operations"
        bbox = draw.textbbox((0, 0), subtitle, font=font_small)
        tw = bbox[2] - bbox[0]
        draw.text(
            (width // 2 - tw // 2, padding + 35),
            subtitle,
            fill=(100, 100, 100),
            font=font_small,
        )

        # Draw array
        array_y = padding + title_height

        # Draw "Initial State" label
        label = "Initial State:"
        draw.text((padding, array_y), label, fill=(60, 60, 60), font=font_medium)

        arrow_y = array_y + 30

        # Draw arrows
        for i in range(n):
            x = padding + i * cell_px
            value = self._array[i]

            # Background color
            if value == 0:
                # Right-facing: green
                bg_color = (200, 255, 200)
                border_color = (50, 150, 50)
                arrow_symbol = "→"
                arrow_color = (0, 120, 0)
            else:
                # Left-facing: red
                bg_color = (255, 200, 200)
                border_color = (200, 50, 50)
                arrow_symbol = "←"
                arrow_color = (180, 0, 0)

            # Draw cell background
            draw.rectangle(
                [x + 2, arrow_y, x + cell_px - 2, arrow_y + cell_px - 2],
                fill=bg_color,
                outline=border_color,
                width=2,
            )

            # Draw arrow
            bbox = draw.textbbox((0, 0), arrow_symbol, font=font_arrow)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            cx = x + cell_px // 2
            cy = arrow_y + cell_px // 2
            draw.text(
                (cx - tw // 2, cy - th // 2),
                arrow_symbol,
                fill=arrow_color,
                font=font_arrow,
            )

            # Draw index below
            index_label = str(i + 1)
            bbox = draw.textbbox((0, 0), index_label, font=font_small)
            tw = bbox[2] - bbox[0]
            draw.text(
                (cx - tw // 2, arrow_y + cell_px + 5),
                index_label,
                fill=(80, 80, 80),
                font=font_small,
            )

        # Draw legend
        legend_y = arrow_y + cell_px + 35

        # Legend title
        draw.text(
            (padding, legend_y), "Legend:", fill=(60, 60, 60), font=font_medium
        )

        legend_items_y = legend_y + 25

        # Right arrow legend
        legend_x = padding + 20
        draw.rectangle(
            [legend_x, legend_items_y, legend_x + 40, legend_items_y + 30],
            fill=(200, 255, 200),
            outline=(50, 150, 50),
            width=2,
        )
        draw.text(
            (legend_x + 5, legend_items_y + 2), "→", fill=(0, 120, 0), font=font_arrow
        )
        draw.text(
            (legend_x + 50, legend_items_y + 8),
            "A[i]=0 (Right-facing, goal state)",
            fill=(60, 60, 60),
            font=font_small,
        )

        # Left arrow legend
        legend_x = padding + 280
        draw.rectangle(
            [legend_x, legend_items_y, legend_x + 40, legend_items_y + 30],
            fill=(255, 200, 200),
            outline=(200, 50, 50),
            width=2,
        )
        draw.text(
            (legend_x + 5, legend_items_y + 2), "←", fill=(180, 0, 0), font=font_arrow
        )
        draw.text(
            (legend_x + 50, legend_items_y + 8),
            "A[i]=1 (Left-facing, needs flipping)",
            fill=(60, 60, 60),
            font=font_small,
        )

        return img

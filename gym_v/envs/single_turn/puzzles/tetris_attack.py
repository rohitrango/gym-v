"""Tetris Attack environment for gym-v (self-contained)."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation
from gym_v.logger import get_logger

logger = get_logger()


class TetrisAttackEnv(Env):
    # Meta: source=RLVE, category=puzzles, turn=single
    """RLVE Tetris Attack as a single-turn environment.

    Given an array of length 2×N containing each integer from 0 to N-1 exactly
    twice, remove all elements using the minimum number of swaps. Adjacent equal
    elements are automatically removed and the array is compacted after each
    removal. After stabilization, swap two adjacent elements to trigger more
    removals.
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""There is an array A (initially it is of length 2 × {N}, containing each integer from 0 to {N_minus_1} exactly twice). Initially, the array A is: {A}

The array follows this rule:
- If there are two adjacent equal elements A[i] == A[i + 1], they are both removed from the array.
- After each removal, the array is compacted (i.e., elements are re-indexed from 0 to the new length), and the process continues as long as such adjacent pairs exist.

Once the array becomes stable (i.e., no adjacent equal pairs remain), you may perform a **swap** between any two adjacent elements A[i] and A[i + 1] (0 ≤ i < current array length - 1). After a swap, the same removal process restarts and continues until stable again. Please **remove all elements from the array**, using the **minimum number of swaps**. Output a single line containing the indices of the swaps (space-separated), where each index `i` indicates a swap between A[i] and A[i + 1]."""

    def __init__(
        self,
        max_n: int = 8,
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
        self._oracle_answer: str | None = None
        self._gold_answer: int | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        if self._n:
            size_hint = f"length 2 × {self._n}"
        else:
            size_hint = "length 2 × N"

        return dedent(
            f"""
            Tetris Attack Problem:

            Given an array of {size_hint} containing each integer from 0 to N-1 exactly twice,
            remove all elements using the minimum number of swaps.

            Rules:
            1) Adjacent equal elements A[i] == A[i+1] are automatically removed
            2) After removal, the array is compacted and the process continues
            3) When stable (no adjacent pairs), you may swap two adjacent elements
            4) After a swap, the removal process restarts
            5) Goal: Remove all elements with minimum swaps

            In the image:
            - Each block represents an array element with its value
            - Different colors distinguish different values
            - The horizontal layout shows the initial array configuration
            - The legend maps colors to values

            Output a single line containing the indices of the swaps (space-separated),
            where each index `i` indicates a swap between A[i] and A[i + 1].
            """
        ).strip()

    def _get_state_text(self) -> str:
        """Return the text representation of the current state."""
        return self._prompt

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
        """Generate a Tetris Attack problem instance.

        Ports generation logic from RLVE using self.np_random.
        """
        N = int(self.np_random.integers(2, self._max_n + 1))

        A = list(range(N)) + list(range(N))
        while True:
            self.np_random.shuffle(A)
            if all(a != b for a, b in zip(A, A[1:], strict=False)):
                break

        vis = [False] * N
        st = []
        swaps = []
        for x in A:
            if vis[x]:
                tax = []
                while st[-1] != x:
                    swaps.append(len(st) - 1)
                    tax.append(st.pop())
                # remove the matching element
                st.pop()
                # restore the other elements
                while tax:
                    st.append(tax.pop())
            else:
                st.append(x)
                vis[x] = True

        if not swaps:
            # If no swaps needed, regenerate
            return self._generate()

        self._n = N
        self._array = A
        self._gold_answer = len(swaps)
        self._oracle_answer = " ".join(map(str, swaps))

    def _prompt_generate(self) -> str:
        """Generate the prompt text for the problem."""
        if self._n is None:
            raise RuntimeError("No problem generated")
        return self.prompt_template.format(
            N=self._n,
            N_minus_1=self._n - 1,
            A=" ".join(f"A[{i}]={Ai}" for i, Ai in enumerate(self._array)),
        )

    def _process(self, answer: str | None) -> list[int] | None:
        """Process the answer string into a list of integers."""
        if answer is not None:
            answer = answer.strip()
            if not answer:
                return None
            try:
                answer_array = list(map(int, answer.split()))
                return answer_array
            except ValueError:
                return None
        else:
            return None

    def _score_answer(self, answer: str) -> float:
        """Score the answer based on correctness.

        Returns:
            -1.0: wrong format
            -0.5: invalid solution (invalid index)
            -0.2: unsuccessful solution (array not empty)
            (gold/answer)^5: valid solution with optimality score
        """
        processed_result = self._process(answer)
        if processed_result is not None:
            if not isinstance(processed_result, list):
                return 0.0
            A = self._array.copy()

            def removal() -> bool:
                nonlocal A
                removed = False
                i = 0
                while i < len(A) - 1:
                    if A[i] == A[i + 1]:
                        A.pop(i)
                        A.pop(i)
                        i = max(0, i - 1)
                        removed = True
                    else:
                        i += 1
                return removed

            # Check that initial array doesn't have adjacent pairs
            if removal():
                return 0.0
            for i in processed_result:
                if not (0 <= i < len(A) - 1):
                    return 0.0
                A[i], A[i + 1] = A[i + 1], A[i]
                removal()
                # After removal, should be stable again
                if removal():
                    return 0.0
            if A:
                return 0.0

            gold, answer_len = self._gold_answer, len(processed_result)
            if not (0 < gold <= answer_len):
                return 0.0
            beta = 5.0
            return (gold / answer_len) ** beta
        else:
            return 0.0

    def render(self) -> Image.Image | list[Image.Image] | None:
        """Render the Tetris Attack array as an image.

        Shows:
        - Horizontal array of colored blocks with values
        - Each value has a distinct color
        - Legend mapping colors to values
        - Clear visual representation of the initial configuration
        """
        if self._n is None or self._array is None:
            raise RuntimeError("No problem generated")

        cell_px = self._cell_px
        padding = self._padding

        array_len = len(self._array)
        n = self._n

        # Calculate dimensions
        grid_width = array_len * cell_px
        grid_height = cell_px

        # Space for legend
        legend_height = 100

        width = padding * 2 + grid_width
        height = padding * 3 + grid_height + legend_height

        img = Image.new("RGB", (width, height), (250, 250, 250))
        draw = ImageDraw.Draw(img)

        # Load font
        font_path = None
        font_path_candidate = self.assets_dir / "DejaVuSans.ttf"
        if font_path_candidate.exists():
            font_path = str(font_path_candidate)

        if font_path:
            font_large = ImageFont.truetype(font_path, int(cell_px * 0.4))
            font_medium = ImageFont.truetype(font_path, 16)
            font_small = ImageFont.truetype(font_path, 14)
        else:
            font_large = ImageFont.load_default()
            font_medium = ImageFont.load_default()
            font_small = ImageFont.load_default()

        # Define distinct colors for each value (0 to N-1)
        # Use a color palette that's visually distinct
        def get_color_for_value(value: int, total: int) -> tuple[int, int, int]:
            """Generate distinct colors using HSV color space."""
            if total == 1:
                return (100, 150, 200)
            # Use hue rotation for distinct colors
            hue = (value * 360 // total) % 360
            # Convert HSV to RGB
            h = hue / 60.0
            c = 200  # Chroma
            x = c * (1 - abs(h % 2 - 1))
            if h < 1:
                r, g, b = c, x, 0
            elif h < 2:
                r, g, b = x, c, 0
            elif h < 3:
                r, g, b = 0, c, x
            elif h < 4:
                r, g, b = 0, x, c
            elif h < 5:
                r, g, b = x, 0, c
            else:
                r, g, b = c, 0, x
            # Add brightness and convert to 0-255
            return (int(r + 55), int(g + 55), int(b + 55))

        # Draw array blocks
        grid_x = padding
        grid_y = padding

        for idx, value in enumerate(self._array):
            x = grid_x + idx * cell_px
            y = grid_y

            # Get color for this value
            block_color = get_color_for_value(value, n)

            # Draw block
            draw.rectangle(
                [x + 2, y + 2, x + cell_px - 2, y + cell_px - 2],
                fill=block_color,
                outline=(30, 30, 30),
                width=2,
            )

            # Draw value text
            v = str(value)
            bbox = draw.textbbox((0, 0), v, font=font_large)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            cx = x + cell_px // 2
            cy = y + cell_px // 2

            # Use white text for darker colors, black for lighter colors
            # Calculate brightness
            brightness = (block_color[0] + block_color[1] + block_color[2]) / 3
            text_color = (255, 255, 255) if brightness < 150 else (10, 10, 10)

            draw.text((cx - tw // 2, cy - th // 2), v, fill=text_color, font=font_large)

            # Draw index below each block
            index_text = f"[{idx}]"
            bbox_idx = draw.textbbox((0, 0), index_text, font=font_small)
            tw_idx = bbox_idx[2] - bbox_idx[0]
            draw.text(
                (cx - tw_idx // 2, y + cell_px + 4),
                index_text,
                fill=(80, 80, 80),
                font=font_small,
            )

        # Draw legend
        legend_y = grid_y + grid_height + padding * 2 + 20

        # Title
        title = "Legend (Value → Color):"
        draw.text((padding, legend_y), title, fill=(30, 30, 30), font=font_medium)
        legend_y += 30

        # Show color mapping
        legend_x = padding + 20
        legend_cell = 30
        items_per_row = min(n, 8)

        for value in range(n):
            col = value % items_per_row
            row = value // items_per_row

            x = legend_x + col * 100
            y = legend_y + row * 40

            # Draw small color block
            block_color = get_color_for_value(value, n)
            draw.rectangle(
                [x, y, x + legend_cell, y + legend_cell],
                fill=block_color,
                outline=(30, 30, 30),
                width=1,
            )

            # Label
            label = f"= {value}"
            draw.text(
                (x + legend_cell + 8, y + 6), label, fill=(30, 30, 30), font=font_small
            )

        return img

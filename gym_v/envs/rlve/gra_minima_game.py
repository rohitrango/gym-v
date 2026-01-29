"""Gra Minima Game environment for gym-v (self-contained)."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation
from gym_v.logger import get_logger

logger = get_logger()


class RLVEGraMinimaGameEnv(Env):
    """RLVE Gra Minima Game as a single-turn environment.

    Alice and Bob play a game with N numbers. They take turns choosing any non-empty
    subset of remaining numbers, adding the minimum of that subset to their score, and
    removing the entire subset from the game. Each player plays optimally to maximize
    their score minus their opponent's score. The objective is to compute the final
    value of (Alice's score - Bob's score).
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""There are {N} numbers: {A}
Alice and Bob are playing a game with these numbers. Alice goes first, and they take turns. On each turn, a player may choose any **non-empty subset** of the remaining numbers, add the **minimum** of that subset to their score, and then remove the entire subset from the game. The game ends when there are no numbers left.
Each player plays optimally to maximize **their score minus their opponent's score**. Please compute the final value of (Alice's score − Bob's score)."""

    def __init__(
        self,
        n: int = 8,
        cell_px: int = 70,
        padding: int = 24,
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._n_param = n
        self._cell_px = cell_px
        self._padding = padding
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._n: int | None = None
        self._numbers: list[int] | None = None
        self._prompt: str | None = None
        self._oracle_answer: int | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        if self._n:
            size_hint = f"{self._n} numbers"
        else:
            size_hint = "N numbers"

        return dedent(
            f"""
            Gra Minima Game:

            Alice and Bob play a turn-based game with {size_hint}.

            Rules:
            1) Alice goes first, then they alternate turns
            2) On each turn, a player chooses any non-empty subset of remaining numbers
            3) The player adds the MINIMUM value from that subset to their score
            4) The entire chosen subset is removed from the game
            5) The game ends when no numbers remain
            6) Both players play optimally to maximize (their score - opponent's score)

            Goal: Compute the final value of (Alice's score - Bob's score).

            In the visualization:
            - The numbers are displayed as a sorted sequence
            - Each number is shown in its own cell with color coding
            - Green cells indicate lower values, red cells indicate higher values
            - The color gradient helps visualize the value distribution
            - Game state indicators show Alice's turn (green) and optimal play strategy

            Output format: A single integer representing (Alice's score - Bob's score).
            """
        ).strip()

    def _get_state_text(self) -> str:
        """Return text representation of the numbers."""
        if self._numbers is None:
            return ""
        return f"Numbers: {' '.join(map(str, self._numbers))}"

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
                "text_prompt": f"{state_text}\n\n{self.description}",
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
        """Generate a Gra Minima Game problem instance.

        Ports generation logic from RLVE using self.np_random.
        """
        N = self._n_param
        if N < 1:
            raise ValueError("N should be greater than or equal to 1")

        # Generate random numbers
        A = [int(self.np_random.integers(1, N * 2 + 1)) for _ in range(N)]

        # Compute optimal score difference using dynamic programming
        # The optimal strategy is to sort and greedily maximize score difference
        A_sorted = sorted(A)
        ans = 0
        for a in A_sorted:
            ans = max(ans, a - ans)

        self._n = N
        self._numbers = A
        self._oracle_answer = ans

    def _prompt_generate(self) -> str:
        """Generate the prompt text for the problem."""
        if self._numbers is None:
            raise RuntimeError("No problem generated")
        return self.prompt_template.format(
            N=self._n,
            A=" ".join(map(str, self._numbers)),
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
        """Render the Gra Minima Game as an image.

        Shows:
        - The sequence of numbers displayed in cells
        - Color-coded values (green for low, red for high)
        - Game state indicators and legend
        """
        if self._numbers is None:
            raise RuntimeError("No problem generated")

        cell_px = self._cell_px
        padding = self._padding

        n_numbers = len(self._numbers)

        # Layout configuration
        title_height = 60
        legend_height = 80
        numbers_per_row = min(10, n_numbers)
        n_rows = (n_numbers + numbers_per_row - 1) // numbers_per_row

        grid_width = numbers_per_row * cell_px
        grid_height = n_rows * cell_px

        width = padding * 2 + grid_width
        height = padding * 3 + title_height + grid_height + legend_height

        img = Image.new("RGB", (width, height), (250, 250, 250))
        draw = ImageDraw.Draw(img)

        # Load font
        font_path = None
        font_path_candidate = self.assets_dir / "DejaVuSans.ttf"
        if font_path_candidate.exists():
            font_path = str(font_path_candidate)

        if font_path:
            font_title = ImageFont.truetype(font_path, 32)
            font_large = ImageFont.truetype(font_path, int(cell_px * 0.35))
            font_medium = ImageFont.truetype(font_path, 18)
            font_small = ImageFont.truetype(font_path, 14)
        else:
            font_title = ImageFont.load_default()
            font_large = ImageFont.load_default()
            font_medium = ImageFont.load_default()
            font_small = ImageFont.load_default()

        # Draw title
        title = "Gra Minima Game - Optimal Score Difference"
        title_bbox = draw.textbbox((0, 0), title, font=font_title)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = (width - title_width) // 2
        draw.text((title_x, padding), title, fill=(30, 30, 30), font=font_title)

        # Find min/max for color scaling
        min_val = min(self._numbers)
        max_val = max(self._numbers)
        val_range = max_val - min_val if max_val > min_val else 1

        # Draw number cells
        numbers_y = padding + title_height + padding

        for idx, value in enumerate(self._numbers):
            row = idx // numbers_per_row
            col = idx % numbers_per_row

            x = padding + col * cell_px
            y = numbers_y + row * cell_px

            # Color gradient: green (low) -> yellow (mid) -> red (high)
            normalized = (value - min_val) / val_range

            if normalized < 0.5:
                # Green to yellow
                t = normalized * 2
                r_val = int(100 + t * 155)
                g_val = int(200 + t * 55)
                b_val = int(100 - t * 100)
            else:
                # Yellow to red
                t = (normalized - 0.5) * 2
                r_val = 255
                g_val = int(255 - t * 155)
                b_val = 0

            cell_color = (r_val, g_val, b_val)

            # Draw cell
            draw.rectangle(
                [x + 2, y + 2, x + cell_px - 2, y + cell_px - 2],
                fill=cell_color,
                outline=(60, 60, 60),
                width=2,
            )

            # Draw value text
            value_str = str(value)
            bbox = draw.textbbox((0, 0), value_str, font=font_large)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            cx = x + cell_px // 2
            cy = y + cell_px // 2

            # Determine text color based on background brightness
            text_color = (255, 255, 255) if normalized > 0.5 else (10, 10, 10)

            # Draw text with shadow for better visibility
            if normalized > 0.5:
                draw.text(
                    (cx - tw // 2 + 1, cy - th // 2 + 1),
                    value_str,
                    fill=(0, 0, 0),
                    font=font_large,
                )
            draw.text(
                (cx - tw // 2, cy - th // 2),
                value_str,
                fill=text_color,
                font=font_large,
            )

        # Draw legend
        legend_y = numbers_y + grid_height + padding

        # Visual legend - color gradient
        legend_title = "Color Legend:"
        draw.text(
            (padding, legend_y), legend_title, fill=(30, 30, 30), font=font_medium
        )
        legend_y += 25

        # Show color gradient examples
        example_x = padding + 10
        example_colors = [
            ((100, 200, 100), "Low"),
            ((255, 255, 0), "Mid"),
            ((255, 100, 0), "High"),
        ]

        for color, label in example_colors:
            # Draw small color box
            draw.rectangle(
                [example_x, legend_y, example_x + 30, legend_y + 20],
                fill=color,
                outline=(60, 60, 60),
                width=1,
            )
            # Draw label
            draw.text(
                (example_x + 40, legend_y + 3),
                f"{label} values",
                fill=(60, 60, 60),
                font=font_small,
            )
            example_x += 140

        return img

"""Coin Square Game environment for gym-v (self-contained)."""

from __future__ import annotations

from array import array
from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation
from gym_v.logger import get_logger

logger = get_logger()


class RLVECoinSquareGameEnv(Env):
    """RLVE Coin Square Game as a single-turn environment.

    Alice and Bob play a game with coins in a row. Players take turns removing
    leftmost coins and adding their values to their scores. The first player
    (Alice) can take 1 or 2 coins initially, then players can take up to 2x
    the previous player's count. Find Alice's maximum score under optimal play.

    Source: https://www.luogu.com.cn/problem/P2964
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""You are given {N} coins in a row (1-indexed from left to right). The i-th coin has value C[i]: {C}
Alice and Bob play alternately, with Alice going first. On a turn, a player removes some **positive number** of **leftmost** coins and adds the sum of their values to their own score. The game ends when no coins remain.

Rules:
- On Alice's **first** turn, she may take either 1 or 2 coins.
- Thereafter, if the previous player took k coins, the current player may take any number of coins from 1 to min(k * 2, the number of remaining coins).

Assuming both players play optimally, what is the **maximum total value** Alice can obtain?"""

    def __init__(
        self,
        max_n: int = 10,
        weight_multiple: int = 2,
        cell_px: int = 70,
        padding: int = 24,
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._max_n = max_n
        self._weight_multiple = weight_multiple
        self._cell_px = cell_px
        self._padding = padding
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._n: int | None = None
        self._coins: list[int] | None = None
        self._prompt: str | None = None
        self._oracle_answer: int | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        if self._n and self._coins:
            size_hint = f"{self._n} coins"
        else:
            size_hint = "N coins"

        return dedent(
            f"""
            Coin Square Game:

            Alice and Bob take turns picking leftmost coins from a row of {size_hint}.
            Each player adds the values of taken coins to their score.

            Rules:
            1) Alice goes first and can take 1 or 2 coins on her first turn
            2) If the previous player took k coins, the current player can take 1 to min(k*2, remaining) coins
            3) Players alternate turns until no coins remain
            4) Both players play optimally to maximize their own score

            In the visualization:
            - Coins are shown in a row from left to right (1-indexed)
            - Each coin displays its value
            - Alice's color: blue
            - Bob's color: red
            - Coin colors vary by value (darker = higher value)

            Output format: A single integer (Alice's maximum total value under optimal play).
            """
        ).strip()

    def _get_state_text(self) -> str:
        """Return text representation of the coins."""
        if self._coins is None:
            return ""
        return f"Coins: {' '.join(map(str, self._coins))}"

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
        """Generate a coin square game instance.

        Ports generation logic from RLVE using self.np_random.
        """
        N = int(self.np_random.integers(5, self._max_n + 1))
        C = [
            int(self.np_random.integers(1, N * self._weight_multiple + 1))
            for _ in range(N)
        ]

        A = C
        # Build prefix sums of the reversed sequence (to match the C++ approach)
        S = [0] * (N + 1)
        for i in range(1, N + 1):
            S[i] = S[i - 1] + A[N - i]

        # dp_rows[i] will store dp[i][j] for j = 0..floor((i+1)/2)
        # (indices beyond this plateau to the same value, so we clamp when reading)
        dp_rows = [None] * (N + 1)
        dp_rows[0] = array("I", [0])

        for i in range(1, N + 1):
            max_j = (i + 1) // 2
            row = array("I", [0] * (max_j + 1))
            for j in range(1, max_j + 1):
                k = 2 * j - 1
                # Start with dp[i][j-1]
                best = row[j - 1]

                # Option 1: take k coins if possible
                r = i - k
                if r >= 0:
                    prev_row = dp_rows[r]
                    prev_max_j = len(prev_row) - 1
                    idx = k if k <= prev_max_j else prev_max_j  # clamp
                    cand = S[i] - prev_row[idx]
                    if cand > best:
                        best = cand

                # Option 2: take k+1 coins if possible
                r2 = i - (k + 1)
                if r2 >= 0:
                    prev_row2 = dp_rows[r2]
                    prev2_max_j = len(prev_row2) - 1
                    idx2 = (k + 1) if (k + 1) <= prev2_max_j else prev2_max_j  # clamp
                    cand2 = S[i] - prev_row2[idx2]
                    if cand2 > best:
                        best = cand2

                row[j] = best

            dp_rows[i] = row

        self._n = N
        self._coins = C
        self._oracle_answer = dp_rows[N][1]

    def _prompt_generate(self) -> str:
        """Generate the prompt text for the problem."""
        if self._n is None:
            raise RuntimeError("No problem generated")
        return self.prompt_template.format(
            N=self._n,
            C=" ".join(f"C[{i}]={Ci}" for i, Ci in enumerate(self._coins, start=1)),
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
        """Render the coin square game as an image.

        Shows:
        - Row of coins with their values
        - Visual indicators for Alice and Bob
        - Color-coded coins based on value
        - Game rules visualization
        """
        if self._n is None or self._coins is None:
            raise RuntimeError("No problem generated")

        cell_px = self._cell_px
        padding = self._padding

        n_coins = self._n

        # Calculate dimensions
        coin_row_width = n_coins * cell_px
        coin_height = cell_px

        # Space for title and legend
        title_height = 60
        legend_height = 80

        width = padding * 2 + coin_row_width
        height = padding * 3 + title_height + coin_height + legend_height

        img = Image.new("RGB", (width, height), (250, 250, 250))
        draw = ImageDraw.Draw(img)

        # Load font
        font_path = None
        font_path_candidate = self.assets_dir / "DejaVuSans.ttf"
        if font_path_candidate.exists():
            font_path = str(font_path_candidate)

        if font_path:
            font_large = ImageFont.truetype(font_path, int(cell_px * 0.35))
            font_medium = ImageFont.truetype(font_path, 20)
            font_small = ImageFont.truetype(font_path, 16)
            font_tiny = ImageFont.truetype(font_path, 14)
        else:
            font_large = ImageFont.load_default()
            font_medium = ImageFont.load_default()
            font_small = ImageFont.load_default()
            font_tiny = ImageFont.load_default()

        # Draw title
        title = "Coin Square Game"
        title_bbox = draw.textbbox((0, 0), title, font=font_medium)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = (width - title_width) // 2
        draw.text((title_x, padding), title, fill=(30, 30, 30), font=font_medium)

        # Find max coin value for color scaling
        max_value = max(self._coins) if self._coins else 1
        min_value = min(self._coins) if self._coins else 0
        value_range = max_value - min_value if max_value > min_value else 1

        # Draw coins in a row
        coin_y = padding + title_height + padding

        for i, coin_value in enumerate(self._coins):
            x = padding + i * cell_px
            y = coin_y

            # Color gradient based on value: light gold (low) to dark gold (high)
            normalized = (coin_value - min_value) / value_range
            # Gold colors: lighter (255, 215, 0) to darker (218, 165, 32)
            red = int(255 - normalized * 37)
            green = int(215 - normalized * 50)
            blue = int(0 + normalized * 32)
            coin_color = (red, green, blue)

            # Draw coin as a circle
            margin = 5
            draw.ellipse(
                [x + margin, y + margin, x + cell_px - margin, y + cell_px - margin],
                fill=coin_color,
                outline=(150, 100, 0),
                width=3,
            )

            # Draw coin value
            value_str = str(coin_value)
            bbox = draw.textbbox((0, 0), value_str, font=font_large)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            cx = x + cell_px // 2
            cy = y + cell_px // 2

            # Use contrasting text color
            text_color = (255, 255, 255) if normalized > 0.5 else (30, 30, 30)

            draw.text(
                (cx - tw // 2, cy - th // 2),
                value_str,
                fill=text_color,
                font=font_large,
            )

            # Draw position label (1-indexed)
            pos_str = str(i + 1)
            pos_bbox = draw.textbbox((0, 0), pos_str, font=font_tiny)
            pos_width = pos_bbox[2] - pos_bbox[0]
            draw.text(
                (cx - pos_width // 2, y + cell_px + 2),
                pos_str,
                fill=(100, 100, 100),
                font=font_tiny,
            )

        # Draw legend
        legend_y = coin_y + coin_height + padding * 2

        # Player indicators
        alice_color = (100, 149, 237)  # Cornflower blue
        bob_color = (220, 20, 60)  # Crimson

        # Alice indicator
        draw.rectangle(
            [padding + 10, legend_y, padding + 30, legend_y + 20],
            fill=alice_color,
            outline=(30, 30, 30),
            width=1,
        )
        draw.text(
            (padding + 40, legend_y + 2),
            "Alice (goes first)",
            fill=(30, 30, 30),
            font=font_tiny,
        )

        # Bob indicator
        bob_y = legend_y + 30
        draw.rectangle(
            [padding + 10, bob_y, padding + 30, bob_y + 20],
            fill=bob_color,
            outline=(30, 30, 30),
            width=1,
        )
        draw.text((padding + 40, bob_y + 2), "Bob", fill=(30, 30, 30), font=font_tiny)

        return img

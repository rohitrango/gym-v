"""Stone game environment for gym-v (self-contained)."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation
from gym_v.logger import get_logger

logger = get_logger()


class RLVEStoneGameEnv(Env):
    """RLVE Stone Game as a single-turn environment.

    Stan and Ollie play a game with heaps of stones. Players take turns splitting
    heaps into smaller heaps. The player who cannot make a move loses.
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""Stan and Ollie are playing a game. The game rules are as follows:
+ There are **{N}** heaps of stones: {Stones}.
+ Stan and Ollie take turns playing, and **Stan** goes first.
+ On a player's turn, they must select a heap that contains at least **{F}** stones.
+ Then, they choose an integer **M** (at least 2) and split the selected heap into **M** smaller heaps such that the sizes of the smaller heaps differ by at most 1 (i.e., as evenly as possible).
+ After splitting, the game continues with the updated heap configuration.
+ If a player cannot make a move (i.e., no heap contains at least **{F}** stones), they lose.

If both players always play optimally, who will win — Stan or Ollie?

**Output Format:** Your final answer should be a single word: either `Stan` or `Ollie` (do **NOT** include quotes or backticks), indicating the winner."""

    def __init__(
        self,
        max_sum: int = 30,
        pile_width: int = 60,
        pile_max_height: int = 200,
        padding: int = 24,
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._max_sum = max_sum
        self._pile_width = pile_width
        self._pile_max_height = pile_max_height
        self._padding = padding
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        if max_sum < 2:
            raise ValueError("max_sum should be >= 2")

        self._n: int | None = None
        self._f: int | None = None
        self._stones: list[int] | None = None
        self._prompt: str | None = None
        self._oracle_answer: str | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        if self._n and self._stones:
            size_hint = f"{self._n} heaps with threshold F={self._f}"
        else:
            size_hint = "N heaps with threshold F"

        return dedent(
            f"""
            Stan and Ollie are playing a game. The game rules are as follows:
            there are N heaps of stones, and Stan and Ollie take turns playing
            (Stan goes first). On a player's turn, they must select a heap that
            contains at least F stones, choose an integer M (at least 2) and
            split the selected heap into M smaller heaps such that the sizes of
            the smaller heaps differ by at most 1 (i.e., as evenly as possible).
            If a player cannot make a move (i.e., no heap contains at least F
            stones), they lose. If both players always play optimally, who will
            win - Stan or Ollie?

            This instance has {size_hint}.

            Rules:
            1) There are N heaps of stones shown as vertical piles
            2) Players alternate turns (Stan goes first)
            3) On each turn, select a heap with at least F stones
            4) Split it into M smaller heaps (M >= 2) as evenly as possible
            5) The player who cannot move loses

            In the visualization:
            - Each vertical pile shows the number of stones in that heap
            - Pile heights are proportional to stone counts
            - The threshold F determines which heaps can be split
            - Darker piles indicate more stones
            - Game information is displayed at the top

            Output Format: Your final answer should be a single word: either
            Stan or Ollie (do NOT include quotes or backticks), indicating
            the winner.
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
        """Generate a stone game problem instance.

        Ports generation logic from RLVE using self.np_random.
        """
        MAX_SUM = self._max_sum
        if MAX_SUM < 2:
            raise ValueError("MAX_SUM should be >= 2")

        self._oracle_answer = "Stan" if self.np_random.random() < 0.5 else "Ollie"

        while True:
            SUM = int(self.np_random.integers(2, MAX_SUM + 1))
            N = int(self.np_random.integers(1, min(SUM // 2, 100) + 1))
            if N == 1:
                Stones = [SUM]
            else:
                cuts = sorted(
                    self.np_random.choice(range(1, SUM), N - 1, replace=False)
                )
                Stones = (
                    [cuts[0]]
                    + [cuts[i] - cuts[i - 1] for i in range(1, N - 1)]
                    + [SUM - cuts[-1]]
                )
            F = int(self.np_random.integers(1, max(Stones) + 2))

            def check(n: int, f: int, stones: list[int]) -> bool:
                sg = [-1] * (max(stones) + 5)
                exist = [0] * (max(stones) + 5)
                for i in range(0, min(max(stones) + 1, f)):
                    sg[i] = 0

                def get_sg(x: int) -> int:
                    if sg[x] != -1:
                        return sg[x]
                    i = 2
                    while i <= x:
                        k = x // (x // i)
                        for j in range(i, min(i + 1, k) + 1):
                            s = 0
                            if (x % j) % 2 == 1:
                                s ^= get_sg(x // j + 1)
                            if (j - (x % j)) % 2 == 1:
                                s ^= get_sg(x // j)
                            exist[s] = x
                        i = k + 1
                    i = 0
                    while True:
                        if exist[i] != x:
                            sg[x] = i
                            return i
                        i += 1

                nim_sum = 0
                for pile_size in stones:
                    nim_sum ^= get_sg(pile_size)
                return nim_sum != 0

            if ("Stan" if check(N, F, Stones) else "Ollie") == self._oracle_answer:
                self._n = N
                self._f = F
                self._stones = Stones
                break

    def _prompt_generate(self) -> str:
        """Generate the prompt text for the problem."""
        if self._n is None:
            raise RuntimeError("No problem generated")
        return self.prompt_template.format(
            N=self._n,
            F=self._f,
            Stones=", ".join(map(str, self._stones)),
        )

    def _process(self, answer: str | None) -> str | None:
        """Process the answer string."""
        if answer is not None:
            return answer.strip()
        else:
            return None

    def _score_answer(self, answer: str) -> float:
        """Score the answer based on correctness.

        Returns:
            -1.0: wrong format (None)
            -0.5: invalid answer (not Stan or Ollie)
             0.0: wrong answer
            +1.0: correct answer
        """
        processed_result = self._process(answer)
        if processed_result is not None:
            if processed_result not in ("Stan", "Ollie"):
                return 0.0
            if processed_result == self._oracle_answer:
                return 1.0
            else:
                return 0.0
        else:
            return 0.0

    def render(self) -> Image.Image | list[Image.Image] | None:
        """Render the stone game as an image.

        Shows:
        - Vertical piles representing each heap
        - Stone counts displayed on each pile
        - Pile heights proportional to stone counts
        - Game parameters (N heaps, threshold F)
        - Visual distinction between piles above and below threshold
        """
        if self._n is None or self._stones is None:
            raise RuntimeError("No problem generated")

        pile_width = self._pile_width
        pile_max_height = self._pile_max_height
        padding = self._padding

        num_piles = self._n
        max_stones = max(self._stones)

        # Calculate dimensions
        title_height = 120
        footer_height = 60
        piles_width = num_piles * pile_width + (num_piles - 1) * 10

        width = padding * 2 + piles_width
        height = padding * 3 + title_height + pile_max_height + footer_height

        img = Image.new("RGB", (width, height), (245, 245, 250))
        draw = ImageDraw.Draw(img)

        # Load font
        font_path = None
        font_path_candidate = self.assets_dir / "DejaVuSans.ttf"
        if font_path_candidate.exists():
            font_path = str(font_path_candidate)

        if font_path:
            font_title = ImageFont.truetype(font_path, 26)
            font_large = ImageFont.truetype(font_path, 20)
            font_medium = ImageFont.truetype(font_path, 16)
            font_small = ImageFont.truetype(font_path, 14)
        else:
            font_title = ImageFont.load_default()
            font_large = ImageFont.load_default()
            font_medium = ImageFont.load_default()
            font_small = ImageFont.load_default()

        # Draw title
        title = "Stone Game - Two Player Strategy"
        bbox = draw.textbbox((0, 0), title, font=font_title)
        title_width = bbox[2] - bbox[0]
        title_x = (width - title_width) // 2
        draw.text((title_x, padding + 10), title, fill=(40, 60, 100), font=font_title)

        # Draw game info
        info_y = padding + 50
        info_text = f"Players: Stan (first) vs Ollie  |  Heaps: {self._n}  |  Split Threshold: F = {self._f}"
        bbox = draw.textbbox((0, 0), info_text, font=font_medium)
        info_width = bbox[2] - bbox[0]
        info_x = (width - info_width) // 2
        draw.text((info_x, info_y), info_text, fill=(80, 80, 80), font=font_medium)

        # Draw rules summary
        rules_y = padding + 80
        rules_text = (
            "Rule: Split heaps with ≥F stones into M parts (M≥2). Cannot move = lose."
        )
        bbox = draw.textbbox((0, 0), rules_text, font=font_small)
        rules_width = bbox[2] - bbox[0]
        rules_x = (width - rules_width) // 2
        draw.text((rules_x, rules_y), rules_text, fill=(120, 120, 120), font=font_small)

        # Draw piles
        piles_y_base = padding + title_height + pile_max_height
        start_x = (width - piles_width) // 2

        for i, stone_count in enumerate(self._stones):
            pile_x = start_x + i * (pile_width + 10)

            # Calculate pile height (proportional to stone count)
            if max_stones > 0:
                pile_height = int((stone_count / max_stones) * pile_max_height)
            else:
                pile_height = 0

            pile_y_top = piles_y_base - pile_height

            # Determine color based on whether pile can be split
            if stone_count >= self._f:
                # Can be split - use darker brown/orange
                base_color = (139, 90, 43)
                intensity = min(1.0, stone_count / (max_stones * 1.2))
                pile_color = (
                    int(base_color[0] * (0.6 + 0.4 * intensity)),
                    int(base_color[1] * (0.6 + 0.4 * intensity)),
                    int(base_color[2] * (0.6 + 0.4 * intensity)),
                )
                border_color = (100, 60, 30)
            else:
                # Cannot be split - use lighter gray/beige
                pile_color = (180, 170, 160)
                border_color = (140, 130, 120)

            # Draw pile with 3D effect
            # Main pile body
            draw.rectangle(
                [pile_x, pile_y_top, pile_x + pile_width, piles_y_base],
                fill=pile_color,
                outline=border_color,
                width=2,
            )

            # Add subtle vertical lines for texture
            num_lines = 4
            for j in range(1, num_lines):
                line_x = pile_x + (pile_width * j) // num_lines
                draw.line(
                    [(line_x, pile_y_top), (line_x, piles_y_base)],
                    fill=(
                        max(0, pile_color[0] - 15),
                        max(0, pile_color[1] - 15),
                        max(0, pile_color[2] - 15),
                    ),
                    width=1,
                )

            # Draw stone count on pile
            count_text = str(stone_count)
            bbox = draw.textbbox((0, 0), count_text, font=font_large)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            text_x = pile_x + (pile_width - text_width) // 2
            text_y = pile_y_top + pile_height // 2 - text_height // 2

            # Choose text color for visibility
            if stone_count >= self._f:
                text_color = (255, 255, 255)
                # Add shadow for better readability
                draw.text(
                    (text_x + 1, text_y + 1),
                    count_text,
                    fill=(0, 0, 0),
                    font=font_large,
                )
            else:
                text_color = (60, 60, 60)

            draw.text((text_x, text_y), count_text, fill=text_color, font=font_large)

            # Draw heap number below pile
            heap_label = f"Heap {i + 1}"
            bbox = draw.textbbox((0, 0), heap_label, font=font_small)
            label_width = bbox[2] - bbox[0]
            label_x = pile_x + (pile_width - label_width) // 2
            label_y = piles_y_base + 10
            draw.text(
                (label_x, label_y), heap_label, fill=(100, 100, 100), font=font_small
            )

        # Draw ground line
        ground_y = piles_y_base + 2
        draw.line(
            [(padding, ground_y), (width - padding, ground_y)],
            fill=(100, 100, 100),
            width=3,
        )

        # Draw legend at bottom
        legend_y = height - footer_height + 15

        # Legend for pile colors
        legend_x = padding + 20
        draw.rectangle(
            [legend_x, legend_y, legend_x + 25, legend_y + 20],
            fill=(139, 90, 43),
            outline=(100, 60, 30),
            width=2,
        )
        draw.text(
            (legend_x + 35, legend_y + 3),
            f"≥{self._f} stones (splittable)",
            fill=(60, 60, 60),
            font=font_small,
        )

        legend_x2 = legend_x + 220
        draw.rectangle(
            [legend_x2, legend_y, legend_x2 + 25, legend_y + 20],
            fill=(180, 170, 160),
            outline=(140, 130, 120),
            width=2,
        )
        draw.text(
            (legend_x2 + 35, legend_y + 3),
            f"<{self._f} stones",
            fill=(60, 60, 60),
            font=font_small,
        )

        return img

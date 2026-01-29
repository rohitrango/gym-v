"""Stone Intervals Game environment for gym-v (self-contained)."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation
from gym_v.logger import get_logger

logger = get_logger()


class RLVEStoneIntervalsGameEnv(Env):
    """RLVE Stone Intervals Game as a single-turn environment.

    Alice and Bob play an optimal stone collection game on N piles where
    players can only take from piles adjacent to empty ones.
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""There are {N} piles of stones. Initially, the i-th pile contains A[i] stones, given as: {A}
Alice and Bob play a game with the following rules:
- Alice goes first. They alternate turns.
- On each turn, a player selects a pile `i` such that **at least one of its adjacent piles** (`i - 1` or `i + 1`, if within bounds) contains **0 stones** (noting that the first/last pile has ONLY ONE adjacent pile). The player then collects **all stones** from pile `i` (pile `i` becomes 0).
- The game ends when there are no piles with any stones remaining.

Assuming both players play optimally to maximize their own total number of collected stones, output the number of stones Alice will collect."""

    def __init__(
        self,
        max_n: int = 10,
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._max_n = max_n
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._n: int | None = None
        self._a: list[int] | None = None
        self._prompt: str | None = None
        self._oracle_answer: int | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        if self._n:
            n_hint = f"{self._n} piles"
        else:
            n_hint = "N piles"

        return dedent(
            f"""
            Stone Intervals Game:

            Given {n_hint} of stones where pile i contains A[i] stones.
            Alice and Bob take turns collecting stones optimally.

            Rules:
            1) Alice goes first, then they alternate
            2) On each turn, select a pile i where at least one adjacent pile (i-1 or i+1) has 0 stones
            3) Collect all stones from pile i (pile i becomes 0)
            4) First/last piles have only ONE adjacent pile
            5) Game ends when all piles are empty
            6) Both players maximize their own total

            In the visualization:
            - Number line shows all pile positions (0 to N-1)
            - Vertical bars represent stone counts in each pile
            - Red bars indicate piles with 0 stones (empty)
            - Blue bars indicate piles with stones
            - Height of bar corresponds to number of stones
            - Numbers on bars show exact stone count
            - Valid moves are shown with green highlights (piles adjacent to empty ones)

            Output format: A single integer (number of stones Alice will collect).
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
        """Generate a stone intervals game problem instance.

        Ports generation logic from RLVE using self.np_random.
        """
        N = int(self.np_random.integers(3, self._max_n + 1))

        A = [int(self.np_random.integers(1, N * 2 + 1)) for _ in range(N)]
        zero_count = int(self.np_random.integers(1, N - 1))
        for zero_index in self.np_random.choice(N, zero_count, replace=False):
            A[zero_index] = 0

        v = A.copy()
        SumVal = sum(v)

        # mark which piles are non-zero
        tag = [x != 0 for x in v]

        # doubly-linked list over 0..N-1
        prev_ = [i - 1 for i in range(N)]
        next_ = [i + 1 for i in range(N)]
        prev_[0] = None
        next_[N - 1] = None

        head = 0
        tail = N - 1

        # 1) Triple-compression: whenever three consecutive non-zero piles
        #    form a "peak" (middle >= both neighbors), merge them into the rightmost.
        i = head
        while i is not None:
            while (
                prev_[i] is not None
                and prev_[prev_[i]] is not None
                and tag[i]
                and tag[prev_[i]]
                and tag[prev_[prev_[i]]]
                and v[prev_[i]] >= v[prev_[prev_[i]]]
                and v[prev_[i]] >= v[i]
            ):
                p = prev_[i]
                pp = prev_[p]
                new_prev = prev_[pp]
                # merge: v[i] = v[pp] + v[i] - v[p]
                v[i] = v[pp] + v[i] - v[p]
                # remove pp and p by re-linking new_prev <-> i
                prev_[i] = new_prev
                if new_prev is not None:
                    next_[new_prev] = i
                else:
                    head = i
            i = next_[i]

        # 2) Edge-peeling: greedily remove matching monotonic pairs at the ends,
        #    accumulating their difference into S
        L, R = head, tail
        S = 0
        # left side
        while True:
            nl = next_[L]
            if nl is None or not (tag[L] and tag[nl]) or v[L] < v[nl]:
                break
            S += v[nl] - v[L]
            L = next_[nl]
        # right side
        while True:
            pr = prev_[R]
            if pr is None or not (tag[R] and tag[pr]) or v[R] < v[pr]:
                break
            S += v[pr] - v[R]
            R = prev_[pr]

        # 3) Collect the remaining non-zero segments between L and R
        segments = []
        i = L
        while True:
            if tag[i]:
                segments.append(v[i])
            if i == R:
                break
            i = next_[i]

        # 4) Sort descending, append the peeled sum S, then do an alternating sum
        segments.sort(reverse=True)
        segments.append(S)
        score = 0
        for idx, val in enumerate(segments):
            score += val if idx % 2 == 0 else -val

        # 5) Recover each player's total
        self._n = N
        self._a = A
        self._oracle_answer = (SumVal + score) // 2

    def _prompt_generate(self) -> str:
        """Generate the prompt text for the problem."""
        if self._n is None:
            raise RuntimeError("No problem generated")
        return self.prompt_template.format(
            N=self._n,
            A=" ".join(f"A[{i}]={Ai}" for i, Ai in enumerate(self._a)),
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
        """Render the stone intervals game as a beautiful visualization.

        Shows:
        - Number line with pile positions
        - Bar chart showing stone counts
        - Empty piles highlighted in red
        - Valid moves (adjacent to empty) highlighted in green
        - Legend explaining the rules
        """
        if self._n is None or self._a is None:
            raise RuntimeError("No problem generated")

        # Image dimensions
        padding = 40
        pile_spacing = 80
        max_bar_height = 200
        number_line_y = 300
        legend_start_y = 380
        legend_height = 60

        width = padding * 2 + (self._n - 1) * pile_spacing + 100
        height = number_line_y + legend_height + padding

        img = Image.new("RGB", (width, height), (250, 250, 250))
        draw = ImageDraw.Draw(img)

        # Load fonts
        font_path = None
        font_path_candidate = self.assets_dir / "DejaVuSans.ttf"
        if font_path_candidate.exists():
            font_path = str(font_path_candidate)

        if font_path:
            font_large = ImageFont.truetype(font_path, 24)
            font_medium = ImageFont.truetype(font_path, 18)
            font_small = ImageFont.truetype(font_path, 14)
            font_title = ImageFont.truetype(font_path, 28)
        else:
            font_large = ImageFont.load_default()
            font_medium = ImageFont.load_default()
            font_small = ImageFont.load_default()
            font_title = ImageFont.load_default()

        # Draw title
        title = "Stone Intervals Game"
        title_bbox = draw.textbbox((0, 0), title, font=font_title)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = (width - title_width) // 2
        draw.text((title_x, 15), title, fill=(30, 30, 30), font=font_title)

        # Find valid moves (piles adjacent to empty ones)
        valid_moves = set()
        for i in range(self._n):
            if self._a[i] > 0:
                # Check if adjacent pile is empty
                has_empty_neighbor = False
                if i > 0 and self._a[i - 1] == 0:
                    has_empty_neighbor = True
                if i < self._n - 1 and self._a[i + 1] == 0:
                    has_empty_neighbor = True
                if has_empty_neighbor:
                    valid_moves.add(i)

        # Find max stone count for scaling
        max_stones = max(self._a) if max(self._a) > 0 else 1

        # Draw number line
        line_y = number_line_y
        line_start_x = padding + 50
        line_end_x = line_start_x + (self._n - 1) * pile_spacing

        # Draw horizontal line
        draw.line(
            (line_start_x, line_y, line_end_x, line_y), fill=(80, 80, 80), width=3
        )

        # Draw piles as bars
        bar_width = 50
        for i in range(self._n):
            pile_x = line_start_x + i * pile_spacing

            # Draw tick on number line
            draw.line(
                (pile_x, line_y - 8, pile_x, line_y + 8), fill=(80, 80, 80), width=2
            )

            # Draw pile index
            index_text = str(i)
            index_bbox = draw.textbbox((0, 0), index_text, font=font_medium)
            index_width = index_bbox[2] - index_bbox[0]
            draw.text(
                (pile_x - index_width // 2, line_y + 15),
                index_text,
                fill=(30, 30, 30),
                font=font_medium,
            )

            # Draw bar for stone count
            stone_count = self._a[i]
            if stone_count > 0:
                bar_height = int((stone_count / max_stones) * max_bar_height)
                bar_top = line_y - bar_height - 10

                # Color based on validity
                if i in valid_moves:
                    bar_color = (46, 204, 64)  # Green for valid moves
                    outline_color = (30, 150, 40)
                    outline_width = 3
                else:
                    bar_color = (100, 150, 255)  # Blue for non-empty
                    outline_color = (60, 100, 200)
                    outline_width = 2

                draw.rectangle(
                    [
                        pile_x - bar_width // 2,
                        bar_top,
                        pile_x + bar_width // 2,
                        line_y - 10,
                    ],
                    fill=bar_color,
                    outline=outline_color,
                    width=outline_width,
                )

                # Draw stone count on bar
                count_text = str(stone_count)
                count_bbox = draw.textbbox((0, 0), count_text, font=font_large)
                count_width = count_bbox[2] - count_bbox[0]
                count_height = count_bbox[3] - count_bbox[1]
                draw.text(
                    (pile_x - count_width // 2, bar_top + 5),
                    count_text,
                    fill=(255, 255, 255),
                    font=font_large,
                )
            else:
                # Empty pile - draw red indicator
                empty_size = 12
                draw.ellipse(
                    [
                        pile_x - empty_size,
                        line_y - 25 - empty_size,
                        pile_x + empty_size,
                        line_y - 25 + empty_size,
                    ],
                    fill=(255, 65, 54),
                    outline=(200, 40, 30),
                    width=2,
                )
                draw.text(
                    (pile_x - 5, line_y - 33),
                    "0",
                    fill=(255, 255, 255),
                    font=font_small,
                )

        # Draw simple footer
        legend_y = legend_start_y
        footer_text = (
            "Green bars = valid moves | Red circles = empty piles | Blue bars = blocked"
        )
        draw.text((padding, legend_y), footer_text, fill=(80, 80, 80), font=font_small)

        return img

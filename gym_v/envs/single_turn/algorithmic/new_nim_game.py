"""New Nim Game environment for gym-v (self-contained)."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation
from gym_v.logger import get_logger

logger = get_logger()


class NewNimGameEnv(Env):
    # Meta: source=RLVE, category=algorithmic, turn=single
    """RLVE New Nim Game as a single-turn environment.

    A strategic Nim-like game with modified rules:
    - First round has two phases: player removes heaps, then opponent removes heaps
    - From second round onward, standard Nim rules apply
    - Both players play optimally
    - Goal: Choose heaps to remove in first move to guarantee a win while minimizing
      total matches removed
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""You are given a Nim-like game with heaps of matches. There are {N} heaps with the following sizes (1-indexed): {A}
Game rules:
- **First round** has two phases:
  1) **Your move (first player):** You may remove **any number of entire heaps** (possibly zero), but you are **not allowed** to remove **all** heaps.
  2) **Opponent's move (second player):** Then the opponent may remove **any number of entire heaps** (possibly zero), but likewise cannot remove **all remaining** heaps.
- **From the second round onward:** Standard Nim rules apply on the remaining heaps: players alternate; a move removes any positive number of matches from **exactly one** heap; the player who takes the last match **wins**.
- Both players play optimally.

Your task: Choose which heaps to remove **in your first move** so that you **guarantee a win**; if multiple winning choices exist, choose one that **minimizes the total number of matches** you remove (i.e., the sum of sizes of the heaps you remove). Output the distinct **indices** (1-based) of the heaps you remove in your first move, in any order, separated by spaces; if you can guarantee victory without removing any heap, output an **empty line**."""

    def __init__(
        self,
        match_number_range_coefficient: int = 2,
        wrong_format: float = -1.0,
        invalid_solution: float = -0.5,
        unsuccessful_solution: float = -0.2,
        rewarding_strategy: str = "(gold/answer)^beta",
        rewarding_weight: float = 1.0,
        rewarding_beta: float = 3.0,
        pile_width: int = 100,
        pile_height: int = 300,
        padding: int = 40,
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._match_number_range_coefficient = match_number_range_coefficient
        self._rewards = {
            "wrong_format": wrong_format,
            "invalid_solution": invalid_solution,
            "unsuccessful_solution": unsuccessful_solution,
            "rewarding_strategy": rewarding_strategy,
            "rewarding_weight": rewarding_weight,
            "rewarding_beta": rewarding_beta,
        }
        self._pile_width = pile_width
        self._pile_height = pile_height
        self._padding = padding
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._n: int | None = None
        self._heaps: list[int] | None = None
        self._prompt: str | None = None
        self._oracle_answer: int | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        if self._n and self._heaps:
            size_hint = f"{self._n} heaps"
        else:
            size_hint = "N heaps"

        return dedent(
            f"""
            New Nim Game Problem:

            A strategic Nim-like game with {size_hint} of matches.

            Rules:
            1) First round has two phases:
               - Your move: Remove any number of entire heaps (but not all)
               - Opponent's move: Remove any number of remaining heaps (but not all)
            2) From second round onward: Standard Nim rules apply
               - Players alternate turns
               - Each move removes matches from exactly one heap
               - Player who takes the last match wins
            3) Both players play optimally

            Goal: Choose which heaps to remove in your first move to guarantee a win,
            while minimizing the total number of matches removed.

            In the visualization:
            - Each pile shows the number of matches (stones) it contains
            - Pile heights are proportional to the number of matches
            - The game state shows current pile configurations
            - Legend explains the strategic game rules

            Output Format: Output the distinct indices (1-based) of the heaps you
            remove in your first move, in any order, separated by spaces; if you can
            guarantee victory without removing any heap, output an empty line.
            """
        ).strip()

    def _get_state_text(self) -> str:
        """Return the text representation of the current state."""
        return self._prompt or ""

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        super().reset(seed=seed)

        # Get N from options if provided, otherwise use random N
        if options and "N" in options:
            self._n = options["N"]
        else:
            self._n = int(self.np_random.integers(3, 8))

        self._generate()
        self._prompt = self._prompt_generate()
        self._last_image = self.render()

        # Generate the oracle answer based on gold_answer
        # The oracle answer should be the indices of heaps to remove
        # If gold_answer is 0, output empty string
        oracle_answer = self._generate_oracle_answer()

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
            "oracle_answer": oracle_answer,
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
        """Generate a new Nim game problem instance.

        Ports generation logic from RLVE using self.np_random.
        """
        N = self._n
        if N < 3:
            raise ValueError("N must be at least 3")

        # Generate random heap sizes
        self._heaps = [
            int(
                self.np_random.integers(1, N * self._match_number_range_coefficient + 1)
            )
            for _ in range(N)
        ]

        # Compute gold answer using linear basis (xor-basis)
        A = self._heaps.copy()
        A.sort(reverse=True)

        max_bit = max(A).bit_length()
        D = [0] * max_bit  # linear basis, dynamic size based on input
        ans = 0

        def add(x: int) -> bool:
            """Try to insert x into the xor-basis D."""
            for i in range(max_bit - 1, -1, -1):
                if (x >> i) & 1:
                    if D[i]:
                        x ^= D[i]
                    else:
                        D[i] = x
                        return True
            return False

        for x in A:
            if not add(x):
                ans += x

        self._oracle_answer = ans

    def _generate_oracle_answer(self) -> str:
        """Generate the oracle answer (indices of heaps to remove).

        This finds ONE optimal solution that minimizes total matches removed.
        If multiple solutions exist with same total, returns any valid one.
        """
        N = self._n
        A = self._heaps
        gold_answer = self._oracle_answer

        # Try all possible subsets of heaps to find optimal solution
        best_indices = None
        best_sum = float("inf")

        for mask in range(1 << N):
            # Cannot remove all heaps or no heaps would remain
            if mask == (1 << N) - 1:
                continue

            # Build removed array
            removed = [(mask >> i) & 1 for i in range(N)]

            # Check if this configuration guarantees victory
            max_bit = max(A).bit_length()
            D = [0] * max_bit

            def add(x: int, _D: list[int] = D, _max_bit: int = max_bit) -> bool:
                """Try to insert x into the xor-basis D."""
                for i in range(_max_bit - 1, -1, -1):
                    if (x >> i) & 1:
                        if _D[i]:
                            x ^= _D[i]
                        else:
                            _D[i] = x
                            return True
                return False

            valid = True
            for i, Ai in enumerate(A):
                if not removed[i]:
                    if not add(Ai):
                        valid = False
                        break

            if valid:
                # This is a valid solution, calculate sum
                current_sum = sum(A[i] for i in range(N) if removed[i])
                if current_sum < best_sum:
                    best_sum = current_sum
                    best_indices = [i + 1 for i in range(N) if removed[i]]

        if best_indices is None:
            raise RuntimeError("No valid solution found!")

        if best_sum != gold_answer:
            raise RuntimeError(
                f"Best sum {best_sum} does not match gold answer {gold_answer}"
            )

        # Return "0" if no heaps need to be removed (user can also provide empty line)
        if not best_indices:
            return "0"

        return " ".join(map(str, best_indices))

    def _prompt_generate(self) -> str:
        """Generate the prompt text for the problem."""
        if self._n is None:
            raise RuntimeError("No problem generated")
        return self.prompt_template.format(
            N=self._n,
            A=", ".join(
                f"the size of heap {i} is {Ai}"
                for i, Ai in enumerate(self._heaps, start=1)
            ),
        )

    def _process(self, answer: str | None) -> list[int] | None:
        """Process the answer string into a list of integers.

        "0" or empty string represents removing no heaps.
        """
        if answer is not None:
            answer = answer.strip()
            if answer == "":
                return []  # Empty answer is valid (remove no heaps)
            try:
                answer_array = list(map(int, answer.split()))
                # Handle "0" as empty list (remove no heaps)
                if answer_array == [0]:
                    return []
                return answer_array
            except ValueError:
                return None
        else:
            return None

    def _score_answer(self, answer: str) -> float:
        """Score the answer based on correctness.

        Returns:
            wrong_format: answer cannot be parsed
            invalid_solution: duplicate indices, out of range, or removes all heaps
            unsuccessful_solution: does not guarantee victory
            rewarding_weight * score: valid solution scored by strategy
        """
        processed_result = self._process(answer)
        if processed_result is not None:
            if not isinstance(processed_result, list):
                return 0.0

            # Check for duplicate indices
            if len(processed_result) != len(set(processed_result)):
                return 0.0

            # Check for valid indices
            if not all(1 <= index <= self._n for index in processed_result):
                return 0.0

            # Check cannot remove all heaps
            if len(processed_result) == self._n:
                return 0.0

            # Build removed array
            removed = [False] * self._n
            for index in processed_result:
                removed[index - 1] = True

            # Check if remaining heaps can guarantee victory using xor-basis
            max_bit = max(self._heaps).bit_length()
            D = [0] * max_bit

            def add(x: int) -> bool:
                """Try to insert x into the xor-basis D."""
                for i in range(max_bit - 1, -1, -1):
                    if (x >> i) & 1:
                        if D[i]:
                            x ^= D[i]
                        else:
                            D[i] = x
                            return True
                return False

            for i, Ai in enumerate(self._heaps):
                if not removed[i]:
                    if not add(Ai):
                        return 0.0

            # Valid solution - compute score
            answer_sum = sum(self._heaps[i - 1] for i in processed_result)
            gold = self._oracle_answer

            if not (0 <= gold <= answer_sum):
                raise ValueError(
                    f"Gold answer {gold} should be non-negative and not exceed answer {answer_sum}"
                )

            if self._rewards["rewarding_strategy"] == "(gold/answer)^beta":
                if answer_sum == 0:
                    if gold != 0:
                        raise ValueError("If answer is 0, gold must also be 0")
                    return self._rewards["rewarding_weight"] * 1.0
                return self._rewards["rewarding_weight"] * (
                    (gold / answer_sum) ** self._rewards["rewarding_beta"]
                )
            elif self._rewards["rewarding_strategy"] == "gold=answer":
                return self._rewards["rewarding_weight"] * (gold == answer_sum)
            else:
                raise NotImplementedError(
                    f"Unknown rewarding strategy: {self._rewards['rewarding_strategy']}"
                )
        else:
            return 0.0

    def render(self) -> Image.Image | list[Image.Image] | None:
        """Render the Nim game as an image.

        Shows:
        - Multiple piles with stone counts
        - Visual representation of each pile
        - Game state showing available moves
        - Legend explaining Nim game rules
        """
        if self._n is None or self._heaps is None:
            raise RuntimeError("No problem generated")

        pile_width = self._pile_width
        pile_height = self._pile_height
        padding = self._padding

        num_piles = self._n

        # Calculate dimensions
        title_height = 80
        legend_height = 60
        piles_area_height = pile_height + 60

        total_pile_width = num_piles * pile_width + (num_piles - 1) * 20
        width = padding * 2 + total_pile_width
        height = padding * 3 + title_height + piles_area_height + legend_height

        img = Image.new("RGB", (width, height), (250, 250, 250))
        draw = ImageDraw.Draw(img)

        # Load font
        font_path = None
        font_path_candidate = self.assets_dir / "DejaVuSans.ttf"
        if font_path_candidate.exists():
            font_path = str(font_path_candidate)

        if font_path:
            font_title = ImageFont.truetype(font_path, 32)
            font_large = ImageFont.truetype(font_path, 28)
            font_medium = ImageFont.truetype(font_path, 18)
            font_small = ImageFont.truetype(font_path, 14)
        else:
            font_title = ImageFont.load_default()
            font_large = ImageFont.load_default()
            font_medium = ImageFont.load_default()
            font_small = ImageFont.load_default()

        # Draw title
        title = "New Nim Game"
        title_bbox = draw.textbbox((0, 0), title, font=font_title)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = (width - title_width) // 2
        draw.text((title_x, padding), title, fill=(30, 30, 30), font=font_title)

        subtitle = f"{self._n} heaps of matches"
        subtitle_bbox = draw.textbbox((0, 0), subtitle, font=font_medium)
        subtitle_width = subtitle_bbox[2] - subtitle_bbox[0]
        subtitle_x = (width - subtitle_width) // 2
        draw.text(
            (subtitle_x, padding + 45), subtitle, fill=(100, 100, 100), font=font_medium
        )

        # Draw piles
        piles_y = padding * 2 + title_height
        max_matches = max(self._heaps)
        if max_matches == 0:
            max_matches = 1

        pile_spacing = 20
        pile_start_x = padding + (total_pile_width - num_piles * pile_width) // 2

        for i, matches in enumerate(self._heaps):
            pile_x = pile_start_x + i * (pile_width + pile_spacing)

            # Draw pile base
            base_y = piles_y + pile_height
            base_height = 10
            draw.rectangle(
                [pile_x, base_y, pile_x + pile_width, base_y + base_height],
                fill=(80, 60, 40),
                outline=(50, 40, 30),
                width=2,
            )

            # Draw pile with height proportional to matches
            if matches > 0:
                pile_fill_height = int((matches / max_matches) * (pile_height - 40))
                pile_top_y = base_y - pile_fill_height

                # Gradient effect for pile
                for y in range(pile_fill_height):
                    ratio = y / pile_fill_height
                    # Brown gradient
                    r = int(160 - ratio * 30)
                    g = int(120 - ratio * 30)
                    b = int(80 - ratio * 30)
                    draw.rectangle(
                        [
                            pile_x + 10,
                            pile_top_y + y,
                            pile_x + pile_width - 10,
                            pile_top_y + y + 1,
                        ],
                        fill=(r, g, b),
                    )

                # Pile outline
                draw.rectangle(
                    [
                        pile_x + 10,
                        pile_top_y,
                        pile_x + pile_width - 10,
                        base_y,
                    ],
                    outline=(100, 70, 40),
                    width=2,
                )

                # Draw matches count on pile
                count_text = str(matches)
                count_bbox = draw.textbbox((0, 0), count_text, font=font_large)
                count_width = count_bbox[2] - count_bbox[0]
                count_height = count_bbox[3] - count_bbox[1]
                count_x = pile_x + (pile_width - count_width) // 2
                count_y = pile_top_y - count_height - 15

                # Draw text with shadow
                draw.text(
                    (count_x + 2, count_y + 2),
                    count_text,
                    fill=(0, 0, 0),
                    font=font_large,
                )
                draw.text(
                    (count_x, count_y),
                    count_text,
                    fill=(255, 255, 255),
                    font=font_large,
                )

            # Draw heap index
            index_text = f"Heap {i + 1}"
            index_bbox = draw.textbbox((0, 0), index_text, font=font_small)
            index_width = index_bbox[2] - index_bbox[0]
            index_x = pile_x + (pile_width - index_width) // 2
            draw.text(
                (index_x, base_y + base_height + 5),
                index_text,
                fill=(60, 60, 60),
                font=font_small,
            )

        # Draw legend
        legend_y = piles_y + piles_area_height + padding
        legend_x = padding + 20

        # Simple footer
        legend_text = "Modified Nim: Two-phase first round, then standard Nim. Details in description."
        draw.text((legend_x, legend_y), legend_text, fill=(80, 80, 80), font=font_small)

        return img

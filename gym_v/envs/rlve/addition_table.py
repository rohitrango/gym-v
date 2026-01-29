"""Addition table puzzle environment for gym-v (self-contained)."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation
from gym_v.logger import get_logger

logger = get_logger()


class RLVEAdditionTableEnv(Env):
    """RLVE Addition Table puzzle as a single-turn environment.

    This environment challenges the agent to find the correct base N and digit
    mappings for a system where distinct letters represent digits in base-N,
    given a set of addition equations.
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""You are given an unknown base-N number system (N is an integer ≥ 3), and {N} distinct digits {ALL_LETTERS} in that system. The digits satisfy the following equations in base-N:

{EQUATIONS}

Note:
- {ALL_LETTERS} are distinct digits in the range [0, N−1].
- Expressions like ba represent base-N numbers formed by **concatenation**. For example, if a=1 and b=2, then ba = "21" in base-N.

Your task is to find the correct base N (in decimal), and the values of {ALL_LETTERS} (also in decimal) that satisfy all the equations.

Output Format:
Your final answer should be a single line containing N, {ALL_LETTERS} (all in decimal), separated by **spaces**.
Example: `{N_plus_1} {EXAMPLE_1}` (do **NOT** include the backticks or quotes); this means N={N_plus_1}, {EXAMPLE_2}."""

    def __init__(
        self,
        min_n: int = 3,
        max_n: int = 10,
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._min_n = min_n
        self._max_n = max_n
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._N: int | None = None
        self._digit2letter: list[str] | None = None
        self._letter2digit: dict[str, int] | None = None
        self._prompt: str | None = None
        self._oracle_answer: str | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        if self._N:
            n_hint = str(self._N)
            num_letters = self._N
        else:
            n_hint = "N"
            num_letters = "N"

        return dedent(
            f"""
            Addition table puzzle rules:
            1) You are given {num_letters} distinct letters representing digits in an unknown base-{n_hint} number system.
            2) Each letter maps to a unique digit from 0 to {n_hint}-1.
            3) The addition table shows sums of all pairs of letters (a+a, a+b, a+c, etc.).
            4) Some sums may produce multi-digit results in the given base.

            In the image:
            - An addition table is shown with letters as row and column headers
            - Each cell contains the result of adding the row letter and column letter in base-{n_hint}
            - The table is symmetric (a+b = b+a)
            - Your task is to determine the base N and the decimal value of each letter

            Output format: A single line with N followed by the decimal values of all letters in alphabetical order, space-separated.
            Example format: {n_hint} 0 1 2 3... (where the numbers are the decimal values of a, b, c, d...)
            """
        ).strip()

    def _get_state_text(self) -> str:
        """Return text representation of the addition table image."""
        if self._N is None or self._digit2letter is None or self._letter2digit is None:
            return ""
        N = self._N
        letters = [chr(97 + i) for i in range(N)]

        # Header row
        rows = ["+ " + " ".join(letters)]

        # Data rows
        for r in range(N):
            row_letter = letters[r]
            cells = [row_letter]
            for c in range(N):
                sum_val = (
                    self._letter2digit[letters[r]] + self._letter2digit[letters[c]]
                )
                cells.append(self._convert_to_expression(sum_val))
            rows.append(" ".join(cells))

        return "\n".join(rows)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        super().reset(seed=seed)

        # Handle N from options or random selection
        if options and "N" in options:
            self._N = options["N"]
        else:
            self._N = int(self.np_random.integers(self._min_n, self._max_n + 1))

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
            text=state_text,
            metadata={
                "text_prompt": f"{state_text}\n\n{self.description}",
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
        """Generate a random addition table puzzle."""
        assert self._N is not None
        N = self._N
        assert N in range(3, 26 + 1), "N should be in the range [3, 26]"

        # Create letter mapping (shuffle digits to letters)
        self._digit2letter = [chr(i) for i in range(97, 97 + N)]
        digit_indices = list(range(N))
        self.np_random.shuffle(digit_indices)
        self._digit2letter = [self._digit2letter[i] for i in digit_indices]

        self._letter2digit = {
            letter: digit for digit, letter in enumerate(self._digit2letter)
        }
        self._oracle_answer = "{} {}".format(
            N, " ".join([str(self._letter2digit[chr(i)]) for i in range(97, 97 + N)])
        )

    def _convert_to_expression(self, n: int) -> str:
        """Convert a decimal number to base-N expression using letters."""
        assert self._N is not None and self._digit2letter is not None
        N = self._N

        if n == 0:
            return self._digit2letter[0]
        else:
            expression = ""
            while n > 0:
                digit = n % N
                expression = self._digit2letter[digit] + expression
                n //= N
            return expression

    def _prompt_generate(self) -> str:
        """Generate the text prompt for the puzzle."""
        if self._N is None or self._digit2letter is None or self._letter2digit is None:
            raise RuntimeError("No puzzle generated")

        N = self._N
        ALL_LETTERS = ", ".join([chr(i) for i in range(97, 97 + N)])

        EQUATIONS = []
        for a_ascii in range(97, 97 + N):
            for b_ascii in range(a_ascii, 97 + N):
                a = chr(a_ascii)
                b = chr(b_ascii)
                EQUATIONS.append(
                    f"{a} + {b} = {self._convert_to_expression(self._letter2digit[a] + self._letter2digit[b])}"
                )
        EQUATIONS = "\n".join(EQUATIONS)

        return self.prompt_template.format(
            ALL_LETTERS=ALL_LETTERS,
            EQUATIONS=EQUATIONS,
            N=N,
            N_plus_1=N + 1,
            EXAMPLE_1=" ".join([str(_) for _ in range(N)]),
            EXAMPLE_2=", ".join([f"{chr(i)}={i - 97}" for i in range(97, 97 + N)]),
        )

    def _process(self, answer: str | None) -> dict[str, Any] | None:
        """Process the answer string into structured format."""
        if answer is None:
            return None
        answer = answer.strip()
        try:
            answer_array = list(map(int, answer.split()))
            if self._N is None:
                return {}
            if len(answer_array) != self._N + 1:
                return {}
            N = answer_array[0]
            digits = answer_array[1:]
            return {"N": N, "digits": digits}
        except ValueError:
            return {}

    def _score_answer(self, answer: str) -> float:
        """Score the answer based on correctness.

        Returns:
            1.0 if the answer is correct, 0.0 otherwise.
        """
        processed_result = self._process(answer)
        if processed_result is None or not processed_result:
            return 0.0

        N = processed_result["N"]
        if N != self._N:
            return 0.0

        predict_digits = processed_result["digits"]
        if len(predict_digits) != N:
            return 0.0

        if self._letter2digit is None:
            return 0.0

        gold_digits = [self._letter2digit[chr(i)] for i in range(97, 97 + N)]

        # Exact match required
        if all(a == b for a, b in zip(gold_digits, predict_digits, strict=False)):
            return 1.0
        return 0.0

    def render(self) -> Image.Image | list[Image.Image] | None:
        """Render the addition table puzzle as a beautiful visual grid."""
        if self._N is None or self._digit2letter is None or self._letter2digit is None:
            raise RuntimeError("No puzzle generated")

        N = self._N
        cell_px = 70
        header_px = 50
        padding = 40

        # Calculate dimensions
        width = padding * 2 + header_px + N * cell_px
        height = padding * 2 + header_px + N * cell_px

        img = Image.new("RGB", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Load font
        font_path = None
        font_path_candidate = self.assets_dir / "DejaVuSans.ttf"
        if font_path_candidate.exists():
            font_path = str(font_path_candidate)

        if font_path:
            large_font = ImageFont.truetype(font_path, int(cell_px * 0.35))
            header_font = ImageFont.truetype(font_path, int(header_px * 0.6))
        else:
            large_font = ImageFont.load_default()
            header_font = ImageFont.load_default()

        # Draw title
        title = f"Addition Table (Base-{N})"
        bbox = draw.textbbox((0, 0), title, font=header_font)
        tw = bbox[2] - bbox[0]
        draw.text(
            ((width - tw) // 2, padding // 3),
            title,
            fill=(50, 50, 150),
            font=header_font,
        )

        # Grid origin
        grid_x = padding + header_px
        grid_y = padding + header_px

        # Draw column headers (top)
        for c in range(N):
            letter = chr(97 + c)
            bbox = draw.textbbox((0, 0), letter, font=header_font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            cx = grid_x + c * cell_px + cell_px // 2
            cy = padding + header_px // 2
            draw.text(
                (cx - tw // 2, cy - th // 2),
                letter,
                fill=(100, 50, 150),
                font=header_font,
            )

        # Draw row headers (left)
        for r in range(N):
            letter = chr(97 + r)
            bbox = draw.textbbox((0, 0), letter, font=header_font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            cx = padding + header_px // 2
            cy = grid_y + r * cell_px + cell_px // 2
            draw.text(
                (cx - tw // 2, cy - th // 2),
                letter,
                fill=(100, 50, 150),
                font=header_font,
            )

        # Draw grid lines
        for r in range(N + 1):
            y = grid_y + r * cell_px
            draw.line(
                (grid_x, y, grid_x + N * cell_px, y),
                fill=(80, 80, 80),
                width=2,
            )

        for c in range(N + 1):
            x = grid_x + c * cell_px
            draw.line(
                (x, grid_y, x, grid_y + N * cell_px),
                fill=(80, 80, 80),
                width=2,
            )

        # Fill in addition results
        for r in range(N):
            for c in range(N):
                row_letter = chr(97 + r)
                col_letter = chr(97 + c)
                sum_value = (
                    self._letter2digit[row_letter] + self._letter2digit[col_letter]
                )
                result = self._convert_to_expression(sum_value)

                # Add light background for diagonal
                if r == c:
                    rect_x1 = grid_x + c * cell_px + 1
                    rect_y1 = grid_y + r * cell_px + 1
                    rect_x2 = grid_x + (c + 1) * cell_px - 1
                    rect_y2 = grid_y + (r + 1) * cell_px - 1
                    draw.rectangle(
                        [rect_x1, rect_y1, rect_x2, rect_y2],
                        fill=(240, 245, 255),
                    )

                bbox = draw.textbbox((0, 0), result, font=large_font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                cx = grid_x + c * cell_px + cell_px // 2
                cy = grid_y + r * cell_px + cell_px // 2
                draw.text(
                    (cx - tw // 2, cy - th // 2),
                    result,
                    fill=(20, 20, 20),
                    font=large_font,
                )

        # Draw + symbol in corner
        plus_text = "+"
        bbox = draw.textbbox((0, 0), plus_text, font=header_font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(
            (padding + header_px // 2 - tw // 2, padding + header_px // 2 - th // 2),
            plus_text,
            fill=(150, 150, 150),
            font=header_font,
        )

        return img

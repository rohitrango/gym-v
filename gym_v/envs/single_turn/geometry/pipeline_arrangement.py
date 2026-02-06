"""Pipeline Arrangement environment for gym-v (self-contained)."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation
from gym_v.logger import get_logger

logger = get_logger()


class PipelineArrangementEnv(Env):
    # Meta: source=RLVE, category=geometry, turn=single
    """RLVE Pipeline Arrangement as a single-turn environment.

    Given N products that each need processing on two machines (A then B),
    find the optimal ordering to minimize total completion time. Machine A
    processes products sequentially, and machine B processes them in the
    order they finish on A.
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""You need to process {N} products labeled from `0` to `{N_minus_1}`. Each product must go through **two machines**, A and B, **in order**.

The processing times for each product on machines A and B are given as:
{A_and_B}

Please determine a permutation (i.e., an ordering) of all products. Each product is processed one by one in the chosen order:
- First on machine A.
- Then, after finishing on A, it waits (if needed) and is processed by machine B; meanwhile, machine A can continue processing subsequent products without any delay.
- Machine B processes one product at a time in the order they complete machine A.

Try your best to **minimize the time** when the **last product finishes** on machine B.

**Output Format:** Your final answer should be a single line containing the indices of the products in the chosen order (i.e., the permutation), separated by spaces."""

    def __init__(
        self,
        max_n: int = 8,
        cell_height: int = 60,
        cell_width: int = 100,
        padding: int = 40,
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._max_n = max_n
        self._cell_height = cell_height
        self._cell_width = cell_width
        self._padding = padding
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._n: int | None = None
        self._processing_a: list[int] | None = None
        self._processing_b: list[int] | None = None
        self._prompt: str | None = None
        self._oracle_answer: str | None = None
        self._gold_answer: int | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        if self._n:
            size_hint = f"{self._n} products"
        else:
            size_hint = "N products"

        return dedent(
            f"""
            Pipeline Arrangement Problem:

            Given {size_hint}, each requiring processing on two machines (A and B) in sequence,
            find the optimal ordering to minimize total completion time.

            Rules:
            1) Each product must be processed on machine A first, then machine B
            2) Machine A processes products one at a time in the chosen order
            3) Machine B processes products in the order they finish on A
            4) Machine B waits if the previous product hasn't finished on A
            5) Goal: minimize the time when the last product finishes on B

            In the image:
            - A flow diagram shows the two-stage pipeline from top to bottom
            - Machine A (blue) is the first processing stage
            - Machine B (orange) is the second processing stage
            - Products flow from Machine A to Machine B
            - Each product box shows its index and processing times [A_time, B_time]
            - The table displays all processing time requirements

            Output Format: Your final answer should be a single line containing the
            indices of the products in the chosen order (i.e., the permutation),
            separated by spaces.
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
        """Generate a pipeline arrangement problem instance.

        Ports generation logic from RLVE using self.np_random.
        """
        N = int(self.np_random.integers(2, self._max_n + 1))

        A = [int(self.np_random.integers(1, N + 1)) for _ in range(N)]
        B = [int(self.np_random.integers(1, N + 1)) for _ in range(N)]

        tasks = []
        for i in range(N):
            if A[i] < B[i]:
                tasks.append((A[i], 0, i))
            else:
                tasks.append((B[i], 1, i))

        tasks.sort(key=lambda x: x[0])

        order = [None] * N
        left, right = 0, N - 1
        for _time, belong, idx in tasks:
            if belong == 0:
                order[left] = idx
                left += 1
            else:
                order[right] = idx
                right -= 1

        self._n = N
        self._processing_a = A
        self._processing_b = B
        self._oracle_answer = " ".join(map(str, order))
        self._gold_answer = self._get_finishing_time(order)

    def _prompt_generate(self) -> str:
        """Generate the prompt text for the problem."""
        if self._n is None:
            raise RuntimeError("No problem generated")
        return self.prompt_template.format(
            N=self._n,
            N_minus_1=self._n - 1,
            A_and_B="\n".join(
                f"A[{i}]={self._processing_a[i]}, B[{i}]={self._processing_b[i]}"
                for i in range(self._n)
            ),
        )

    def _get_finishing_time(self, order: list[int]) -> int:
        """Calculate total completion time for a given order."""
        tA = tB = 0
        for idx in order:
            tA += self._processing_a[idx]
            if tB < tA:
                tB = tA
            tB += self._processing_b[idx]
        return tB

    def _process(self, answer: str | None) -> list[int] | None:
        """Process the answer string into a list of integers."""
        if answer is not None:
            answer = answer.strip()
            try:
                answer_array = list(map(int, answer.split()))
                return answer_array
            except ValueError:
                return None
        else:
            return None

    def _score_answer(self, answer: str) -> float:
        """Score the answer using (gold/answer)^beta strategy.

        Returns:
            -1.0: wrong format (cannot parse)
            -0.5: invalid solution (not a valid permutation)
            rewarding_weight * ((gold / answer) ** beta): valid solution
        """
        processed_result = self._process(answer)
        if processed_result is not None:
            if len(processed_result) != self._n:
                return 0.0
            if len(set(processed_result)) != self._n:
                return 0.0
            if not all(0 <= i < self._n for i in processed_result):
                return 0.0
            answer_time = self._get_finishing_time(processed_result)
            gold = self._gold_answer

            beta = 5.0
            rewarding_weight = 1.0
            return rewarding_weight * ((gold / answer_time) ** beta)
        else:
            return 0.0

    def render(self) -> Image.Image | list[Image.Image] | None:
        """Render the pipeline arrangement problem as a flow diagram.

        Shows:
        - Two-stage pipeline (Machine A -> Machine B)
        - Product boxes with processing times
        - Flow arrows showing the pipeline direction
        - Table of processing requirements
        """
        if self._n is None or self._processing_a is None:
            raise RuntimeError("No problem generated")

        padding = self._padding
        cell_height = self._cell_height
        cell_width = self._cell_width

        # Calculate dimensions
        n = self._n
        title_height = 120
        machine_label_width = 120
        products_width = n * cell_width
        table_height = 40 + (n + 1) * 30  # Header + rows
        arrow_space = 60

        width = padding * 2 + machine_label_width + products_width
        height = (
            padding * 3 + title_height + cell_height * 2 + arrow_space + table_height
        )

        img = Image.new("RGB", (width, height), (250, 250, 250))
        draw = ImageDraw.Draw(img)

        # Load fonts
        font_path = None
        font_path_candidate = self.assets_dir / "DejaVuSans.ttf"
        if font_path_candidate.exists():
            font_path = str(font_path_candidate)

        if font_path:
            font_title = ImageFont.truetype(font_path, 28)
            font_large = ImageFont.truetype(font_path, 18)
            font_medium = ImageFont.truetype(font_path, 14)
            font_small = ImageFont.truetype(font_path, 12)
        else:
            font_title = ImageFont.load_default()
            font_large = ImageFont.load_default()
            font_medium = ImageFont.load_default()
            font_small = ImageFont.load_default()

        # Draw title
        title = "Pipeline Arrangement Problem"
        title_bbox = draw.textbbox((0, 0), title, font=font_title)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = (width - title_width) // 2
        draw.text((title_x, padding), title, fill=(30, 30, 30), font=font_title)

        # Subtitle
        subtitle = f"Optimize processing order for {n} products through 2 machines"
        subtitle_bbox = draw.textbbox((0, 0), subtitle, font=font_small)
        subtitle_width = subtitle_bbox[2] - subtitle_bbox[0]
        subtitle_x = (width - subtitle_width) // 2
        draw.text(
            (subtitle_x, padding + 35), subtitle, fill=(100, 100, 100), font=font_small
        )

        # Starting Y for machines
        start_y = padding + title_height

        # Machine A (blue)
        machine_a_y = start_y
        label_x = padding
        products_x = padding + machine_label_width

        # Machine A label
        draw.rectangle(
            [
                label_x,
                machine_a_y,
                label_x + machine_label_width - 10,
                machine_a_y + cell_height,
            ],
            fill=(70, 130, 180),
            outline=(50, 100, 150),
            width=2,
        )
        label_text = "Machine A"
        label_bbox = draw.textbbox((0, 0), label_text, font=font_large)
        label_w, label_h = label_bbox[2] - label_bbox[0], label_bbox[3] - label_bbox[1]
        draw.text(
            (
                label_x + (machine_label_width - 10 - label_w) // 2,
                machine_a_y + (cell_height - label_h) // 2,
            ),
            label_text,
            fill=(255, 255, 255),
            font=font_large,
        )

        # Machine A product boxes
        for i in range(n):
            x = products_x + i * cell_width
            y = machine_a_y

            # Product box
            draw.rectangle(
                [x, y, x + cell_width - 4, y + cell_height],
                fill=(200, 220, 240),
                outline=(70, 130, 180),
                width=2,
            )

            # Product ID
            product_text = f"P{i}"
            product_bbox = draw.textbbox((0, 0), product_text, font=font_medium)
            pw, _ph = (
                product_bbox[2] - product_bbox[0],
                product_bbox[3] - product_bbox[1],
            )
            draw.text(
                (x + (cell_width - 4 - pw) // 2, y + 8),
                product_text,
                fill=(30, 30, 30),
                font=font_medium,
            )

            # Processing time
            time_text = f"A: {self._processing_a[i]}"
            time_bbox = draw.textbbox((0, 0), time_text, font=font_small)
            tw, th = time_bbox[2] - time_bbox[0], time_bbox[3] - time_bbox[1]
            draw.text(
                (x + (cell_width - 4 - tw) // 2, y + cell_height - th - 8),
                time_text,
                fill=(50, 100, 150),
                font=font_small,
            )

        # Arrow space between machines
        machine_a_y + cell_height + arrow_space // 2

        # Draw flow arrows
        for i in range(n):
            x = products_x + i * cell_width + (cell_width - 4) // 2
            y_start = machine_a_y + cell_height
            y_end = machine_a_y + cell_height + arrow_space

            # Vertical arrow
            draw.line([(x, y_start + 5), (x, y_end - 5)], fill=(120, 120, 120), width=2)
            # Arrow head
            draw.polygon(
                [(x, y_end - 5), (x - 5, y_end - 12), (x + 5, y_end - 12)],
                fill=(120, 120, 120),
            )

        # Machine B (orange)
        machine_b_y = machine_a_y + cell_height + arrow_space

        # Machine B label
        draw.rectangle(
            [
                label_x,
                machine_b_y,
                label_x + machine_label_width - 10,
                machine_b_y + cell_height,
            ],
            fill=(230, 140, 70),
            outline=(200, 110, 50),
            width=2,
        )
        label_text = "Machine B"
        label_bbox = draw.textbbox((0, 0), label_text, font=font_large)
        label_w, label_h = label_bbox[2] - label_bbox[0], label_bbox[3] - label_bbox[1]
        draw.text(
            (
                label_x + (machine_label_width - 10 - label_w) // 2,
                machine_b_y + (cell_height - label_h) // 2,
            ),
            label_text,
            fill=(255, 255, 255),
            font=font_large,
        )

        # Machine B product boxes
        for i in range(n):
            x = products_x + i * cell_width
            y = machine_b_y

            # Product box
            draw.rectangle(
                [x, y, x + cell_width - 4, y + cell_height],
                fill=(255, 220, 200),
                outline=(230, 140, 70),
                width=2,
            )

            # Product ID
            product_text = f"P{i}"
            product_bbox = draw.textbbox((0, 0), product_text, font=font_medium)
            pw, _ph = (
                product_bbox[2] - product_bbox[0],
                product_bbox[3] - product_bbox[1],
            )
            draw.text(
                (x + (cell_width - 4 - pw) // 2, y + 8),
                product_text,
                fill=(30, 30, 30),
                font=font_medium,
            )

            # Processing time
            time_text = f"B: {self._processing_b[i]}"
            time_bbox = draw.textbbox((0, 0), time_text, font=font_small)
            tw, th = time_bbox[2] - time_bbox[0], time_bbox[3] - time_bbox[1]
            draw.text(
                (x + (cell_width - 4 - tw) // 2, y + cell_height - th - 8),
                time_text,
                fill=(200, 110, 50),
                font=font_small,
            )

        # Draw processing table
        table_y = machine_b_y + cell_height + padding
        table_x = padding + 20

        # Table title
        table_title = "Processing Time Requirements:"
        draw.text((table_x, table_y), table_title, fill=(30, 30, 30), font=font_medium)
        table_y += 25

        # Table header
        col_width = 80
        header_y = table_y
        draw.text((table_x, header_y), "Product", fill=(50, 50, 50), font=font_small)
        draw.text(
            (table_x + col_width, header_y),
            "Machine A",
            fill=(50, 50, 50),
            font=font_small,
        )
        draw.text(
            (table_x + col_width * 2, header_y),
            "Machine B",
            fill=(50, 50, 50),
            font=font_small,
        )
        table_y += 20

        # Table rows
        for i in range(n):
            row_y = table_y + i * 22
            draw.text((table_x, row_y), f"P{i}", fill=(80, 80, 80), font=font_small)
            draw.text(
                (table_x + col_width, row_y),
                str(self._processing_a[i]),
                fill=(70, 130, 180),
                font=font_small,
            )
            draw.text(
                (table_x + col_width * 2, row_y),
                str(self._processing_b[i]),
                fill=(230, 140, 70),
                font=font_small,
            )

        return img

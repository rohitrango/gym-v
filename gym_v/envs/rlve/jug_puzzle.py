"""Jug puzzle environment for gym-v (self-contained)."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation
from gym_v.logger import get_logger

logger = get_logger()


class RLVEJugPuzzleEnv(Env):
    """RLVE Jug Puzzle as a single-turn environment.

    Given N jugs with specified capacities (initially empty), fill one jug
    with exactly a target volume using Fill, Empty, and Pour operations.
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""You are given {N} jugs (initially empty) with the following capacities:
{jug_capacities}

Please fill a jug (you pick the one) with exactly {target_volume} liters of water. You may perform the following actions:
- `Fill i` — Fill jug `i` to its full capacity.
- `Empty i` — Empty all water from jug `i`.
- `Pour i j` — Pour water from jug `i` to jug `j` until either jug `i` is empty or jug `j` is full.

**Output Format:** Each action should be written on its own line in the format shown above (without backticks or quotes). Output one action per line, in the order they should be performed."""

    def __init__(
        self,
        max_capacity_multiple: int = 10,
        operation_probabilities: list[float] | None = None,
        wrong_format: float = -1.0,
        invalid_solution: float = -0.5,
        wrong_solution: float = 0.0,
        correct_solution: float = 1.0,
        jug_width: int = 80,
        jug_height: int = 200,
        padding: int = 40,
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._max_capacity_multiple = max_capacity_multiple

        if operation_probabilities is None:
            operation_probabilities = [0.1, 0.1, 0.8]

        if len(operation_probabilities) != 3:
            raise ValueError(
                "operation_probabilities should have exactly 3 elements for Fill, Empty, and Pour operations"
            )
        if sum(operation_probabilities) <= 0:
            raise ValueError("operation_probabilities should sum to a positive value")

        self._operation_probabilities = operation_probabilities
        self._rewards = {
            "wrong_format": wrong_format,
            "invalid_solution": invalid_solution,
            "wrong_solution": wrong_solution,
            "correct_solution": correct_solution,
        }

        self._jug_width = jug_width
        self._jug_height = jug_height
        self._padding = padding
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._n: int | None = None
        self._jug_capacities: list[int] | None = None
        self._target_volume: int | None = None
        self._oracle_answer: str | None = None
        self._prompt: str | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        if self._n and self._jug_capacities:
            n_jugs = self._n
            capacities_str = ", ".join(str(c) for c in self._jug_capacities)
            target = self._target_volume if self._target_volume else "X"
        else:
            n_jugs = "N"
            capacities_str = "C1, C2, ..., CN"
            target = "X"

        return dedent(
            f"""
            Jug Puzzle Problem:

            Given {n_jugs} jugs with capacities [{capacities_str}] (all initially empty),
            fill one jug with exactly {target} liters of water.

            Available operations:
            - Fill i: Fill jug i to its full capacity
            - Empty i: Empty all water from jug i
            - Pour i j: Pour water from jug i to jug j until either jug i is empty or jug j is full

            In the visualization:
            - Each jug is shown as a cylinder with its capacity labeled at the top
            - Jugs are initially empty (shown with light blue color)
            - The target volume is displayed prominently
            - Jug indices are labeled at the bottom (0-indexed)

            Output Format: Each action should be written on its own line in the format shown above (without backticks or quotes). Output one action per line, in the order they should be performed.
            """
        ).strip()

    def _get_state_text(self) -> str:
        """Return the text representation of the jug problem."""
        if (
            self._n is None
            or self._jug_capacities is None
            or self._target_volume is None
        ):
            raise RuntimeError("No problem generated")

        capacities_str = ", ".join(
            f"Jug {i}: {cap}L" for i, cap in enumerate(self._jug_capacities)
        )
        return f"Jugs: [{capacities_str}], Target: {self._target_volume}L"

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        super().reset(seed=seed)

        # Get N from options or use default
        if options and "N" in options:
            n = options["N"]
        else:
            n = 3  # Default value

        if options and "steps" in options:
            steps = options["steps"]
        else:
            steps = 5  # Default value

        self._generate(n=n, steps=steps)
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

    def _generate(self, n: int, steps: int) -> None:
        """Generate a jug puzzle problem instance.

        Ports generation logic from RLVE using self.np_random.
        """
        if n < 2:
            raise ValueError("N should be greater than or equal to 2")
        if steps < 2:
            raise ValueError("steps should be greater than or equal to 2")

        N = n
        capacities = [
            int(self.np_random.integers(2, N * self._max_capacity_multiple + 1))
            for _ in range(N)
        ]
        differences = set(
            capacity_i - capacity_j
            for capacity_j in capacities
            for capacity_i in capacities
            if capacity_i != capacity_j
        )

        jug = int(self.np_random.integers(0, N))
        reference_answer = f"Fill {jug}"
        target_volume = capacities[jug]

        volumes = [0] * N
        actions = ""
        existing_volumes = set()

        for step in range(steps):
            while True:
                operation = self.np_random.choice(
                    ["Fill", "Empty", "Pour"],
                    p=self._operation_probabilities,
                )
                if operation == "Fill":
                    jug = int(self.np_random.integers(0, N))
                    if volumes[jug] < capacities[jug]:
                        actions += f"Fill {jug}\n"
                        volumes[jug] = capacities[jug]
                        break
                elif operation == "Empty":
                    jug = int(self.np_random.integers(0, N))
                    if volumes[jug] > 0:
                        actions += f"Empty {jug}\n"
                        volumes[jug] = 0
                        break
                elif operation == "Pour":
                    jug_i = int(self.np_random.integers(0, N))
                    jug_j = int(self.np_random.integers(0, N))
                    if (
                        jug_i != jug_j
                        and volumes[jug_i] > 0
                        and volumes[jug_j] < capacities[jug_j]
                    ):
                        actions += f"Pour {jug_i} {jug_j}\n"
                        pour_amount = min(
                            volumes[jug_i], capacities[jug_j] - volumes[jug_j]
                        )
                        volumes[jug_i] -= pour_amount
                        volumes[jug_j] += pour_amount
                        break

            target_volumes = (
                set(volume for volume in volumes if volume > 0)
                - existing_volumes
                - differences
                - set(capacities)
            )
            if target_volumes:
                reference_answer = actions
                target_volume = self.np_random.choice(list(target_volumes))
                existing_volumes |= target_volumes

        self._n = N
        self._jug_capacities = capacities
        self._target_volume = int(target_volume)
        self._oracle_answer = reference_answer

    def _prompt_generate(self) -> str:
        """Generate the prompt text for the problem."""
        if self._n is None:
            raise RuntimeError("No problem generated")
        return self.prompt_template.format(
            N=self._n,
            target_volume=self._target_volume,
            jug_capacities="\n".join(
                f"Jug {i}'s capacity: {capacity} liters"
                for i, capacity in enumerate(self._jug_capacities)
            ),
        )

    def _process(self, answer: str | None) -> list | None:
        """Process the answer string into a list of actions."""
        if answer is not None:
            answer = answer.strip()
            if not answer:
                return None
            actions = []
            for line in answer.splitlines():
                line = line.strip()
                if line:
                    actions.append(line.split())
                    action = actions[-1]
                    if not action:
                        return None
                    if action[0] in ("Fill", "Empty"):
                        if len(action) != 2:
                            return None
                        try:
                            action[1] = int(action[1])
                        except ValueError:
                            return None
                    elif action[0] == "Pour":
                        if len(action) != 3:
                            return None
                        try:
                            action[1] = int(action[1])
                            action[2] = int(action[2])
                        except ValueError:
                            return None
                    else:
                        return None
            return actions
        else:
            return None

    def _score_answer(self, answer: str) -> float:
        """Score the answer based on correctness.

        Returns:
            -1.0: wrong format
            -0.5: invalid solution (out of bounds operations)
             0.0: wrong solution (doesn't achieve target)
            +1.0: correct solution
        """
        processed_result = self._process(answer)
        if processed_result is not None:
            volumes = [0] * self._n
            for action in processed_result:
                if action[0] == "Fill":
                    jug = action[1]
                    if not (0 <= jug < self._n):
                        return 0.0
                    volumes[jug] = self._jug_capacities[jug]
                elif action[0] == "Empty":
                    jug = action[1]
                    if not (0 <= jug < self._n):
                        return 0.0
                    volumes[jug] = 0
                elif action[0] == "Pour":
                    jug_i, jug_j = action[1], action[2]
                    if not (
                        0 <= jug_i < self._n and 0 <= jug_j < self._n and jug_i != jug_j
                    ):
                        return 0.0
                    pour_amount = min(
                        volumes[jug_i],
                        self._jug_capacities[jug_j] - volumes[jug_j],
                    )
                    volumes[jug_i] -= pour_amount
                    volumes[jug_j] += pour_amount
                else:
                    raise AssertionError("Should be unreachable")

            if self._target_volume in volumes:
                return self._rewards["correct_solution"]
            else:
                return self._rewards["wrong_solution"]
        else:
            return 0.0

    def render(self) -> Image.Image | list[Image.Image] | None:
        """Render the jug puzzle as an image.

        Shows:
        - Cylinders representing jugs with capacity labels
        - Jug indices at the bottom
        - Target volume displayed prominently
        - All jugs initially empty
        """
        if self._n is None or self._jug_capacities is None:
            raise RuntimeError("No problem generated")

        jug_width = self._jug_width
        jug_height = self._jug_height
        padding = self._padding

        n_jugs = self._n
        max_capacity = max(self._jug_capacities)

        # Load font first to calculate text dimensions
        font_path = None
        font_path_candidate = self.assets_dir / "DejaVuSans.ttf"
        if font_path_candidate.exists():
            font_path = str(font_path_candidate)

        if font_path:
            font_title = ImageFont.truetype(font_path, 28)
            font_large = ImageFont.truetype(font_path, 18)
            font_medium = ImageFont.truetype(font_path, 16)
            font_small = ImageFont.truetype(font_path, 14)
        else:
            font_title = ImageFont.load_default()
            font_large = ImageFont.load_default()
            font_medium = ImageFont.load_default()
            font_small = ImageFont.load_default()

        # Calculate dimensions
        title_height = 120

        # Calculate content width (jugs)
        content_width = n_jugs * jug_width + (n_jugs - 1) * padding

        # Calculate title width
        title = f"Fill one jug with exactly {self._target_volume} liters"
        # Create dummy draw to measure text
        dummy_img = Image.new("RGB", (1, 1))
        dummy_draw = ImageDraw.Draw(dummy_img)
        bbox = dummy_draw.textbbox((0, 0), title, font=font_title)
        title_width = bbox[2] - bbox[0]

        # Ensure total width accommodates both content and title
        total_width = max(padding * 2 + content_width, padding * 2 + title_width)
        total_height = padding * 3 + title_height + jug_height

        img = Image.new("RGB", (total_width, total_height), (250, 250, 250))
        draw = ImageDraw.Draw(img)

        # Draw title with target volume
        title_x = (total_width - title_width) // 2
        draw.text((title_x, padding), title, fill=(220, 50, 50), font=font_title)

        # Draw subtitle
        subtitle = "(all jugs initially empty)"
        bbox = draw.textbbox((0, 0), subtitle, font=font_small)
        subtitle_width = bbox[2] - bbox[0]
        subtitle_x = (total_width - subtitle_width) // 2
        draw.text(
            (subtitle_x, padding + 35), subtitle, fill=(100, 100, 100), font=font_small
        )

        # Draw each jug - centered horizontally
        jug_y = padding * 2 + title_height
        start_x = (total_width - content_width) // 2

        for i in range(n_jugs):
            capacity = self._jug_capacities[i]
            jug_x = start_x + i * (jug_width + padding)
            jug_center_x = jug_x + jug_width // 2

            # Draw jug as cylinder (initially empty)
            top_y = jug_y
            bottom_y = jug_y + jug_height

            # Draw cylinder body with gradient effect
            body_color = (230, 245, 255)  # Very light blue for empty jug
            draw.rectangle(
                [jug_x + 8, top_y + 10, jug_x + jug_width - 8, bottom_y],
                fill=body_color,
                outline=None,
            )

            # Draw left shadow (darker)
            for offset in range(4):
                alpha = 1.0 - (offset / 4.0) * 0.3
                shadow_color = (
                    int(230 * alpha),
                    int(245 * alpha),
                    int(255 * alpha),
                )
                draw.line(
                    [jug_x + 8 + offset, top_y + 10, jug_x + 8 + offset, bottom_y],
                    fill=shadow_color,
                    width=1,
                )

            # Draw right highlight (lighter)
            for offset in range(4):
                alpha = 1.0 + (offset / 4.0) * 0.1
                highlight_color = (
                    min(255, int(230 * alpha)),
                    min(255, int(245 * alpha)),
                    255,
                )
                draw.line(
                    [
                        jug_x + jug_width - 8 - offset,
                        top_y + 10,
                        jug_x + jug_width - 8 - offset,
                        bottom_y,
                    ],
                    fill=highlight_color,
                    width=1,
                )

            # Draw top ellipse (opening of jug)
            draw.ellipse(
                [jug_x + 8, top_y, jug_x + jug_width - 8, top_y + 20],
                fill=body_color,
                outline=(80, 130, 180),
                width=2,
            )

            # Draw bottom ellipse (base of jug)
            draw.ellipse(
                [jug_x + 8, bottom_y - 10, jug_x + jug_width - 8, bottom_y + 10],
                fill=(200, 220, 240),
                outline=(80, 130, 180),
                width=2,
            )

            # Draw main outline
            draw.line(
                [jug_x + 8, top_y + 10, jug_x + 8, bottom_y],
                fill=(80, 130, 180),
                width=2,
            )
            draw.line(
                [jug_x + jug_width - 8, top_y + 10, jug_x + jug_width - 8, bottom_y],
                fill=(80, 130, 180),
                width=2,
            )

            # Draw capacity label at the top of jug
            capacity_text = f"{capacity}L"
            bbox = draw.textbbox((0, 0), capacity_text, font=font_large)
            tw = bbox[2] - bbox[0]
            draw.text(
                (jug_center_x - tw // 2, top_y - 32),
                capacity_text,
                fill=(30, 80, 140),
                font=font_large,
            )

            # Draw jug index at the bottom
            index_text = f"Jug {i}"
            bbox = draw.textbbox((0, 0), index_text, font=font_medium)
            tw = bbox[2] - bbox[0]
            draw.text(
                (jug_center_x - tw // 2, bottom_y + 20),
                index_text,
                fill=(60, 60, 60),
                font=font_medium,
            )

        return img

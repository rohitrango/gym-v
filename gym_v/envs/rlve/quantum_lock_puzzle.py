"""Quantum Lock Puzzle environment for gym-v (self-contained)."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation
from gym_v.logger import get_logger

logger = get_logger()


class RLVEQuantumLockPuzzleEnv(Env):
    """RLVE Quantum Lock Puzzle as a single-turn environment.

    Given a state machine with quantum-like behavior where pressing buttons
    toggles between states (X = 0/1) and modifies a variable Y according to
    different rules depending on the current state. Find a sequence of button
    presses to reach a target Y value.
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""There is a 0/1 variable X, which is initially 0. You also have a variable Y, which starts at {Y_start}. You can press the buttons in any order, and you may press the same button multiple times. There are {N} buttons in total. Each time you press **any** button, X toggles: it becomes 1 - X.

When X is 0 and you press a button, Y changes according to the following rules:
{X0_rules}

When X is 1 and you press a button, Y changes according to the following rules:
{X1_rules}

Please find a sequence of button presses that will make Y equal to {Y_target}.

**Output Format:** Your final answer should be a single line containing the sequence of button presses in order, separated by spaces. For example, `0 1 0 2` means you pressed button 0, then button 1, then button 0 again, and finally button 2. Do **NOT** include backticks or quotes in your output."""

    def __init__(
        self,
        operation_weights: list[float] | None = None,
        wrong_format: float = -1.0,
        invalid_solution: float = -0.5,
        wrong_solution: float = 0.0,
        correct_solution: float = 1.0,
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if operation_weights is None:
            operation_weights = [0.4, 0.4, 0.2]
        self._operation_weights = operation_weights
        self._rewards = {
            "wrong_format": wrong_format,
            "invalid_solution": invalid_solution,
            "wrong_solution": wrong_solution,
            "correct_solution": correct_solution,
        }
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._n: int | None = None
        self._y_start: int | None = None
        self._y_target: int | None = None
        self._buttons: list[list[list[str | int]]] | None = None
        self._prompt: str | None = None
        self._reference_answer: str | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        n_hint = f"{self._n} buttons" if self._n else "N buttons"
        y_start_hint = f"Y={self._y_start}" if self._y_start is not None else "Y=Y_start"
        y_target_hint = f"Y={self._y_target}" if self._y_target is not None else "Y=Y_target"

        return dedent(
            f"""
            Quantum Lock Puzzle:

            Given a quantum-like state machine with:
            - Variable X (initially 0) that toggles with every button press
            - Variable Y (starts at {y_start_hint}) that transforms based on current state
            - {n_hint} with different operations for each state

            Rules:
            1) X toggles (0 → 1 or 1 → 0) every time ANY button is pressed
            2) When X=0, buttons apply one set of operations to Y
            3) When X=1, buttons apply a different set of operations to Y
            4) Operations: +, -, or * with various values
            5) Find sequence to reach target {y_target_hint}

            In the visualization:
            - State machine diagram showing X=0 and X=1 states
            - Each button shown as a transition with both state-dependent operations
            - Arrows show state transitions (X toggles with each press)
            - Button operations displayed as "Button i: Y ← operation" for each state
            - Start state (X=0, Y=Y_start) and goal (Y=Y_target) clearly marked
            - Color coding: X=0 state in blue, X=1 state in orange

            Output format: Space-separated button indices (e.g., "0 1 0 2").
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
                "rlve_reference_answer": str(self._reference_answer),
            },
        )
        info = {
            "reference_answer": str(self._reference_answer),
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
                "rlve_reference_answer": str(self._reference_answer),
            },
        )
        info = {
            "reference_answer": str(self._reference_answer),
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

    def operate(self, y: int, rule: list[str | int]) -> int:
        """Apply operation rule to Y value."""
        operation, value = rule
        if operation == "+":
            return y + value
        elif operation == "-":
            return y - value
        elif operation == "*":
            return y * value
        else:
            raise NotImplementedError(f"Unknown operation: {operation}")

    def _generate(self) -> None:
        """Generate a quantum lock puzzle instance.

        Ports generation logic from RLVE using self.np_random.
        """
        # Get N from options if available, otherwise default
        N = 3  # Default value
        if hasattr(self, "_env_options") and self._env_options is not None:
            N = self._env_options.get("N", 3)

        if N < 2:
            raise ValueError("N should be greater than or equal to 2")

        Y = int(self.np_random.integers(-N, N + 1))
        self._y_start = Y

        buttons = []
        for button in range(N):
            button_rules = []
            for _ in range(2):  # Two rules per button (for X=0 and X=1)
                operation = self.np_random.choice(
                    ["+", "-", "*"], p=self._operation_weights
                )
                if operation in ("+", "-"):
                    value = int(self.np_random.integers(1, N + 1))
                elif operation == "*":
                    value = int(self.np_random.integers(2, 4))
                else:
                    raise NotImplementedError
                button_rules.append([operation, value])
            buttons.append(button_rules)
        self._buttons = buttons

        # Get steps from options if available, otherwise default to N
        steps = N
        if hasattr(self, "_env_options") and self._env_options is not None:
            steps = self._env_options.get("steps", N)

        if steps < 2:
            raise ValueError("steps should be greater than or equal to 2")
        steps += int(self.np_random.integers(0, 2))

        X = 0
        pressed_buttons = []
        existing_Y = set([Y])
        reference_answer = None
        y_target = Y

        for step in range(steps):
            button = int(self.np_random.integers(0, N))
            pressed_buttons.append(button)
            Y = self.operate(Y, buttons[button][X])
            X = 1 - X
            if Y not in existing_Y:
                existing_Y.add(Y)
                reference_answer = pressed_buttons.copy()
                y_target = Y

        if reference_answer is None:
            self._reference_answer = ""
            self._y_target = self._y_start
        else:
            self._reference_answer = " ".join(map(str, reference_answer))
            self._y_target = y_target

        self._n = N

    def _prompt_generate(self) -> str:
        """Generate the prompt text for the problem."""
        if self._n is None:
            raise RuntimeError("No problem generated")
        return self.prompt_template.format(
            N=self._n,
            Y_start=self._y_start,
            Y_target=self._y_target,
            X0_rules="\n".join(
                "When you press button {}, Y becomes Y {} {}".format(
                    i, button[0][0], button[0][1]
                )
                for i, button in enumerate(self._buttons)
            ),
            X1_rules="\n".join(
                "When you press button {}, Y becomes Y {} {}".format(
                    i, button[1][0], button[1][1]
                )
                for i, button in enumerate(self._buttons)
            ),
        )

    def _process(self, answer: str | None) -> list[int] | None:
        """Process the answer string into a list of button indices."""
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
            -0.5: invalid solution (button out of range)
             0.0: wrong solution (doesn't reach target)
            +1.0: correct solution
        """
        processed_result = self._process(answer)
        if processed_result is not None:
            if not isinstance(processed_result, list):
                return self._rewards["wrong_format"]

            X, Y = 0, self._y_start
            for button in processed_result:
                if not (0 <= button < self._n):
                    return self._rewards["invalid_solution"]
                Y = self.operate(Y, self._buttons[button][X])
                X = 1 - X

            if Y == self._y_target:
                return self._rewards["correct_solution"]
            else:
                return self._rewards["wrong_solution"]
        else:
            return self._rewards["wrong_format"]

    def render(self) -> Image.Image | list[Image.Image] | None:
        """Render the quantum lock puzzle as a state machine diagram.

        Shows:
        - Two quantum states (X=0 and X=1) as large circles
        - Transitions between states with button operation labels
        - Start state (X=0, Y=Y_start) clearly marked
        - Goal state indication (Y=Y_target)
        - All button operations for both states
        """
        if self._n is None or self._buttons is None:
            raise RuntimeError("No problem generated")

        # Image dimensions
        width = 1000
        height = 800
        padding = 40

        img = Image.new("RGB", (width, height), (250, 250, 250))
        draw = ImageDraw.Draw(img)

        # Load font
        font_path = None
        font_path_candidate = self.assets_dir / "DejaVuSans.ttf"
        if font_path_candidate.exists():
            font_path = str(font_path_candidate)

        if font_path:
            font_title = ImageFont.truetype(font_path, 32)
            font_large = ImageFont.truetype(font_path, 24)
            font_medium = ImageFont.truetype(font_path, 18)
            font_small = ImageFont.truetype(font_path, 14)
        else:
            font_title = ImageFont.load_default()
            font_large = ImageFont.load_default()
            font_medium = ImageFont.load_default()
            font_small = ImageFont.load_default()

        # Draw title
        title = "Quantum Lock Puzzle - State Machine"
        title_bbox = draw.textbbox((0, 0), title, font=font_title)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = (width - title_width) // 2
        draw.text((title_x, padding), title, fill=(30, 30, 30), font=font_title)

        # State circle parameters
        state_radius = 100
        state_x0 = width // 3
        state_x1 = 2 * width // 3
        state_y = height // 3

        # Draw X=0 state (blue)
        draw.ellipse(
            [
                state_x0 - state_radius,
                state_y - state_radius,
                state_x0 + state_radius,
                state_y + state_radius,
            ],
            fill=(180, 210, 255),
            outline=(50, 100, 200),
            width=4,
        )

        # Draw X=1 state (orange)
        draw.ellipse(
            [
                state_x1 - state_radius,
                state_y - state_radius,
                state_x1 + state_radius,
                state_y + state_radius,
            ],
            fill=(255, 200, 150),
            outline=(200, 100, 50),
            width=4,
        )

        # State labels
        x0_label = "X = 0"
        x1_label = "X = 1"
        bbox = draw.textbbox((0, 0), x0_label, font=font_large)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text(
            (state_x0 - tw // 2, state_y - th // 2),
            x0_label,
            fill=(30, 30, 30),
            font=font_large,
        )

        bbox = draw.textbbox((0, 0), x1_label, font=font_large)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text(
            (state_x1 - tw // 2, state_y - th // 2),
            x1_label,
            fill=(30, 30, 30),
            font=font_large,
        )

        # Draw start marker
        start_label = f"START: Y = {self._y_start}"
        bbox = draw.textbbox((0, 0), start_label, font=font_medium)
        tw = bbox[2] - bbox[0]
        draw.text(
            (state_x0 - tw // 2, state_y - state_radius - 35),
            start_label,
            fill=(0, 120, 0),
            font=font_medium,
        )

        # Draw goal marker
        goal_label = f"GOAL: Y = {self._y_target}"
        bbox = draw.textbbox((0, 0), goal_label, font=font_medium)
        tw = bbox[2] - bbox[0]
        draw.text(
            (width // 2 - tw // 2, height - padding - 20),
            goal_label,
            fill=(180, 0, 0),
            font=font_medium,
        )

        # Draw transition arrows with button operations
        # Arrows go from X=0 to X=1 and from X=1 to X=0

        # Calculate arrow positions (curved arrows)
        arrow_y_offset_up = -40
        arrow_y_offset_down = 40

        # Arrow from X=0 to X=1 (upper arc)
        arrow_start_x0_to_x1 = state_x0 + state_radius - 20
        arrow_start_y0_to_x1 = state_y - 60
        arrow_end_x0_to_x1 = state_x1 - state_radius + 20
        arrow_end_y0_to_x1 = state_y - 60

        # Arrow from X=1 to X=0 (lower arc)
        arrow_start_x1_to_x0 = state_x1 - state_radius + 20
        arrow_start_y1_to_x0 = state_y + 60
        arrow_end_x1_to_x0 = state_x0 + state_radius - 20
        arrow_end_y1_to_x0 = state_y + 60

        # Draw curved arrows (simplified as straight with arc indication)
        # Upper arrow (X=0 → X=1)
        draw.line(
            [
                arrow_start_x0_to_x1,
                arrow_start_y0_to_x1,
                arrow_end_x0_to_x1,
                arrow_end_y0_to_x1,
            ],
            fill=(100, 100, 100),
            width=3,
        )
        # Arrowhead
        draw.polygon(
            [
                (arrow_end_x0_to_x1, arrow_end_y0_to_x1),
                (arrow_end_x0_to_x1 - 10, arrow_end_y0_to_x1 - 8),
                (arrow_end_x0_to_x1 - 10, arrow_end_y0_to_x1 + 8),
            ],
            fill=(100, 100, 100),
        )

        # Lower arrow (X=1 → X=0)
        draw.line(
            [
                arrow_start_x1_to_x0,
                arrow_start_y1_to_x0,
                arrow_end_x1_to_x0,
                arrow_end_y1_to_x0,
            ],
            fill=(100, 100, 100),
            width=3,
        )
        # Arrowhead
        draw.polygon(
            [
                (arrow_end_x1_to_x0, arrow_end_y1_to_x0),
                (arrow_end_x1_to_x0 + 10, arrow_end_y1_to_x0 - 8),
                (arrow_end_x1_to_x0 + 10, arrow_end_y1_to_x0 + 8),
            ],
            fill=(100, 100, 100),
        )

        # Draw button operations
        button_area_y = state_y + state_radius + 80
        button_area_height = height - button_area_y - padding - 40

        # Split button operations into two columns
        col_width = width // 2
        button_spacing = 25

        # Title for button operations
        ops_title = "Button Operations (Each press toggles X):"
        bbox = draw.textbbox((0, 0), ops_title, font=font_medium)
        tw = bbox[2] - bbox[0]
        draw.text(
            (width // 2 - tw // 2, button_area_y - 30),
            ops_title,
            fill=(30, 30, 30),
            font=font_medium,
        )

        # Draw button operations
        for i in range(self._n):
            btn_y = button_area_y + (i * button_spacing)

            # When X=0 (left side)
            op0, val0 = self._buttons[i][0]
            text0 = f"Button {i}: Y ← Y {op0} {val0}"
            draw.text(
                (padding + 20, btn_y),
                text0,
                fill=(50, 100, 200),
                font=font_small,
            )

            # When X=1 (right side)
            op1, val1 = self._buttons[i][1]
            text1 = f"Button {i}: Y ← Y {op1} {val1}"
            draw.text(
                (col_width + 20, btn_y),
                text1,
                fill=(200, 100, 50),
                font=font_small,
            )

        # Add column headers
        header_x0 = "When X = 0:"
        header_x1 = "When X = 1:"
        draw.text(
            (padding + 20, button_area_y - 10),
            header_x0,
            fill=(50, 100, 200),
            font=font_medium,
        )
        draw.text(
            (col_width + 20, button_area_y - 10),
            header_x1,
            fill=(200, 100, 50),
            font=font_medium,
        )

        return img

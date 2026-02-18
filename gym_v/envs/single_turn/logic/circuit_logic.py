"""Circuit Logic single-turn environment backed by reasoning-gym."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from reasoning_gym.factory import create_dataset

from gym_v import Env, Observation, get_logger

logger = get_logger()


class CircuitLogicEnv(Env):
    # Meta: source=ReasoningGym, category=logic, turn=single
    """Circuit Logic puzzle using reasoning-gym's dataset.

    The player evaluates a logic circuit given input assignments.

    Args:
        dataset_kwargs: Configuration parameters for the reasoning-gym dataset
        padding: Padding around the diagram in pixels for rendering
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    def __init__(
        self,
        dataset_kwargs: dict[str, Any] | None = None,
        padding: int = 24,
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._dataset_kwargs = dataset_kwargs or {}
        self._padding = padding
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._seed: int | None = None
        self._dataset = None
        self._entry: dict[str, Any] | None = None
        self._entry_idx: int | None = None
        self._metadata: dict[str, Any] | None = None
        self._diagram: str | None = None
        self._assignments: dict[str, int] | None = None
        self._oracle_answer: str | None = None

    @property
    def description(self) -> str:
        """Return description for Circuit Logic puzzle.

        Original reasoning-gym question format:
        ```
        Below is a randomly generated logic circuit.

        A: ─────────────────┐
        B: ───────────────┐ │
        C: ─────────────┐ │ │
        ...
            │ │ │ │ │ │ ├────│&&───┐
            │ │ │ │ │ │ │ ├─>o─│&&   │
        ...

        Legend for gates:
        &&: AND
        ↑↑: NAND
        ⊕⊕: XOR
        >o: Negate
        ++: OR

        Given the following input assignments:
          A = 1
          B = 0
          ...

        What is the output of the circuit?
        ```

        Original reasoning-gym answer format:
        ```
        1
        ```
        (Single bit: 0 or 1)
        """
        return dedent("""
            The image (PCB-style circuit board) is a randomly generated logic circuit.
            What is the output of the circuit?
            Output format: 0 or 1.
        """).strip()

    def _make_dataset(self, *, seed: int | None):
        kwargs = self._dataset_kwargs.copy()
        if "seed" not in kwargs:
            kwargs["seed"] = seed if seed is not None else int(self.np_random.integers(0, 2**31))
        if "size" not in kwargs:
            kwargs["size"] = 500

        return create_dataset("circuit_logic", **kwargs)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        super().reset(seed=seed)
        self._seed = seed

        self._dataset = self._make_dataset(seed=self._seed)
        self._entry_idx = int(self.np_random.integers(0, len(self._dataset)))
        self._entry = self._dataset[self._entry_idx]

        self._oracle_answer = self._entry["answer"]
        self._metadata = self._entry.get("metadata", {})
        self._assignments = self._metadata.get("assignments", {})

        # Extract diagram from question (it's not in metadata)
        self._diagram = self._extract_diagram_from_question(self._entry["question"])

        logger.info("Reset ReasoningGym Circuit Logic.")

        # Include both diagram and assignments in text
        text = self._diagram
        if self._assignments:
            text += "\n\nInputs: "
            text += ", ".join(f"{k}={v}" for k, v in sorted(self._assignments.items()))

        obs = Observation(
            image=self.render(),
            text=None,
            metadata={
                "state_text": text,
                **self._metadata,
                "text_prompt": self._entry.get("question", ""),
            },
        )
        info = {
            "reasoning_gym_seed": self._seed,
            "reasoning_gym_index": self._entry_idx,
            "oracle_answer": self._oracle_answer,
        }
        return {agent_id: obs for agent_id in self._agent_ids}, {
            agent_id: info for agent_id in self._agent_ids
        }

    def _extract_diagram_from_question(self, question: str) -> str:
        """Extract the ASCII circuit diagram from the question.

        The diagram is between the intro text and the legend.
        """
        lines = question.split("\n")
        diagram_lines = []
        in_diagram = False

        for line in lines:
            # Start capturing when we see input wire labels like "A: ──"
            if not in_diagram and ": " in line and "─" in line:
                in_diagram = True

            if in_diagram:
                # Stop at legend or input assignments
                if line.strip().startswith("Legend") or line.strip().startswith(
                    "Given"
                ):
                    break
                diagram_lines.append(line)

        return "\n".join(diagram_lines).rstrip()

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
        answer = action[agent_id]
        try:
            reward = self._dataset.score_answer(answer=answer, entry=self._entry)
        except Exception as e:
            logger.warning(f"score_answer failed for {type(self).__name__}: {e}, answer={str(answer)[:200]}")
            reward = 0.0

        obs = Observation(
            image=self.render(),
            text=None,
            metadata={
                **self._metadata,
                "text_prompt": self._entry.get("question", ""),
            },
        )
        info = {
            "reasoning_gym_seed": self._seed,
            "reasoning_gym_index": self._entry_idx,
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

    def render(self) -> Image.Image | list[Image.Image] | None:
        return self._render_circuit(
            self._diagram, self._assignments, padding=self._padding
        )

    def _render_circuit(
        self,
        diagram: str,
        assignments: dict[str, int] | None,
        padding: int = 32,
    ) -> Image.Image:
        """Render circuit diagram with PCB-style aesthetics."""
        if not diagram:
            return Image.new("RGB", (200, 200), (20, 60, 40))

        # PCB color scheme
        pcb_green = (20, 80, 50)
        pcb_dark_green = (15, 60, 40)
        copper_color = (180, 130, 70)
        copper_bright = (220, 170, 100)
        solder_color = (200, 200, 200)
        text_color = (240, 240, 220)
        gate_color = (40, 40, 40)
        input_high = (50, 200, 100)  # Green for 1
        input_low = (200, 80, 80)  # Red for 0
        label_color = (255, 255, 200)

        font_path = self.assets_dir / "DejaVuSansMono.ttf"
        if not font_path.exists():
            font_path = self.assets_dir / "DejaVuSans.ttf"

        lines = diagram.split("\n")
        max_line_len = max(len(line) for line in lines) if lines else 0

        # Larger font for better readability
        font_size = 16
        if font_path.exists():
            font = ImageFont.truetype(str(font_path), font_size)
            small_font = ImageFont.truetype(str(font_path), font_size - 2)
            title_font = ImageFont.truetype(str(font_path), font_size + 4)
        else:
            logger.warning(f"Font file not found: {font_path}, using default font")
            font = ImageFont.load_default()
            small_font = font
            title_font = font

        char_w = font_size * 0.62
        char_h = font_size * 1.25

        # Header and footer space
        header_h = 50
        footer_h = 50
        legend_h = 70

        # Calculate minimum width needed for bottom panels
        num_inputs = len(assignments) if assignments else 0
        input_panel_width = 80 + num_inputs * 50 + 20  # INPUTS: + LEDs
        legend_panel_width = 520  # GATES: + 5 gates + LED legend
        min_panel_width = max(input_panel_width, legend_panel_width)

        diagram_width = int(char_w * max_line_len + 60)
        width = max(
            int(padding * 2 + diagram_width), int(padding * 2 + min_panel_width)
        )
        height = int(padding * 2 + char_h * len(lines) + header_h + footer_h + legend_h)

        img = Image.new("RGB", (width, height), pcb_green)
        draw = ImageDraw.Draw(img)

        # Draw PCB texture (grid pattern)
        grid_spacing = 20
        for x in range(0, width, grid_spacing):
            draw.line([(x, 0), (x, height)], fill=pcb_dark_green, width=1)
        for y in range(0, height, grid_spacing):
            draw.line([(0, y), (width, y)], fill=pcb_dark_green, width=1)

        # Draw mounting holes in corners
        hole_radius = 8
        hole_positions = [
            (padding // 2, padding // 2),
            (width - padding // 2, padding // 2),
            (padding // 2, height - padding // 2),
            (width - padding // 2, height - padding // 2),
        ]
        for hx, hy in hole_positions:
            # Copper ring
            draw.ellipse(
                [
                    hx - hole_radius - 4,
                    hy - hole_radius - 4,
                    hx + hole_radius + 4,
                    hy + hole_radius + 4,
                ],
                fill=copper_color,
            )
            # Hole
            draw.ellipse(
                [
                    hx - hole_radius,
                    hy - hole_radius,
                    hx + hole_radius,
                    hy + hole_radius,
                ],
                fill=(30, 30, 30),
            )

        # Draw header
        title = "LOGIC CIRCUIT"
        bbox = draw.textbbox((0, 0), title, font=title_font)
        tw = bbox[2] - bbox[0]
        draw.text(
            (width // 2 - tw // 2, padding),
            title,
            fill=label_color,
            font=title_font,
        )

        # Draw circuit diagram with enhanced styling
        diagram_y = padding + header_h
        for line_idx, line in enumerate(lines):
            x = padding + 20
            y = diagram_y + int(line_idx * char_h)

            # Process each character for special rendering
            for char_idx, char in enumerate(line):
                cx = x + int(char_idx * char_w)

                # Color wires (─ │ ┐ ┘ └ ┌ ├ ┤ ┬ ┴ ┼)
                if char in "─│┐┘└┌├┤┬┴┼":
                    draw.text((cx, y), char, fill=copper_bright, font=font)
                # Input labels (A:, B:, etc.)
                elif char_idx < 3 and char.isalpha() and char.isupper():
                    # Draw a small LED indicator
                    led_x = cx + int(char_w * 0.5)
                    led_y = y + int(char_h * 0.4)
                    led_r = 4
                    if assignments and char in assignments:
                        led_color = input_high if assignments[char] == 1 else input_low
                    else:
                        led_color = solder_color
                    draw.ellipse(
                        [led_x - led_r, led_y - led_r, led_x + led_r, led_y + led_r],
                        fill=led_color,
                    )
                    draw.text((cx, y), char, fill=label_color, font=font)
                # Gate symbols
                elif char in "&+⊕↑":
                    draw.text((cx, y), char, fill=(255, 200, 100), font=font)
                # Negation
                elif char == "o" and char_idx > 0 and line[char_idx - 1] == ">":
                    draw.text((cx, y), char, fill=(255, 100, 100), font=font)
                elif char == ">":
                    draw.text((cx, y), char, fill=(255, 100, 100), font=font)
                # OUT label
                elif "OUT" in line[max(0, char_idx - 2) : char_idx + 3]:
                    draw.text((cx, y), char, fill=text_color, font=font)
                else:
                    draw.text((cx, y), char, fill=text_color, font=font)

        # Draw input assignments panel at bottom
        panel_y = height - footer_h - legend_h - padding // 2
        panel_h = footer_h
        panel_margin = padding

        # Panel background
        draw.rectangle(
            [panel_margin, panel_y, width - panel_margin, panel_y + panel_h],
            fill=(30, 30, 30),
            outline=copper_color,
            width=2,
        )

        # Input values - use smaller font and tighter spacing
        tiny_font = small_font
        if assignments:
            input_x = panel_margin + 10
            input_y = panel_y + 8
            draw.text((input_x, input_y), "INPUTS:", fill=solder_color, font=tiny_font)

            # Calculate spacing based on available width
            avail_width = width - 2 * panel_margin - 80
            num_items = len(assignments)
            spacing = min(48, avail_width // max(num_items, 1))

            # Draw each input as an LED indicator
            led_x = input_x + 70
            for key, val in sorted(assignments.items()):
                # LED
                led_color = input_high if val == 1 else input_low
                draw.ellipse(
                    [led_x, input_y + 3, led_x + 10, input_y + 13],
                    fill=led_color,
                    outline=(80, 80, 80),
                )
                # Label
                draw.text(
                    (led_x + 13, input_y),
                    f"{key}={val}",
                    fill=label_color,
                    font=tiny_font,
                )
                led_x += spacing

        # Draw legend at bottom
        legend_y = height - legend_h - padding // 2
        draw.rectangle(
            [panel_margin, legend_y, width - panel_margin, legend_y + legend_h - 10],
            fill=(25, 50, 35),
            outline=(60, 100, 70),
            width=1,
        )

        # First row: Gates
        lx = panel_margin + 10
        ly = legend_y + 6
        draw.text((lx, ly), "GATES:", fill=solder_color, font=tiny_font)

        gate_items = [
            ("&&", "AND"),
            ("++", "OR"),
            ("⊕⊕", "XOR"),
            ("↑↑", "NAND"),
            (">o", "NOT"),
        ]

        # Calculate gate spacing
        gate_start_x = lx + 55
        avail_gate_width = width - 2 * panel_margin - 70
        gate_spacing = min(75, avail_gate_width // len(gate_items))

        gx = gate_start_x
        for symbol, name in gate_items:
            gate_color = (255, 100, 100) if symbol == ">o" else (255, 200, 100)
            draw.text((gx, ly), symbol, fill=gate_color, font=tiny_font)
            draw.text((gx + 22, ly), name, fill=text_color, font=tiny_font)
            gx += gate_spacing

        # Second row: LED legend
        ly2 = legend_y + 28
        draw.text((panel_margin + 10, ly2), "LED:", fill=solder_color, font=tiny_font)
        # Green LED
        draw.ellipse(
            [panel_margin + 45, ly2 + 2, panel_margin + 55, ly2 + 12], fill=input_high
        )
        draw.text((panel_margin + 60, ly2), "=1(HIGH)", fill=text_color, font=tiny_font)
        # Red LED
        draw.ellipse(
            [panel_margin + 135, ly2 + 2, panel_margin + 145, ly2 + 12], fill=input_low
        )
        draw.text((panel_margin + 150, ly2), "=0(LOW)", fill=text_color, font=tiny_font)

        # Draw border
        draw.rectangle(
            [2, 2, width - 3, height - 3],
            outline=copper_color,
            width=3,
        )

        return img

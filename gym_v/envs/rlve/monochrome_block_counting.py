"""Monochrome block counting environment for gym-v (self-contained)."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation
from gym_v.logger import get_logger

logger = get_logger()


class RLVEMonochromeBlockCountingEnv(Env):
    """RLVE Monochrome block counting as a single-turn environment.

    This environment asks the agent to count the number of distinct ways to build
    a tower of blocks with the maximum number of layers, subject to constraints
    on the number of black and white blocks available.
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""You are building a **tower of blocks** with the following rules:
- The i-th layer (from top to bottom) must contain exactly i blocks (i is from 1 to N if the tower has N layers).
- All blocks in the same layer must be of the **same color**: either black or white.
- You may use **at most {A} black blocks** and **at most {B} white blocks** in total.
- You should build a tower with the **maximum possible number of layers (N)** under these constraints.

Please compute the total number of distinct ways to build such a tower with the **maximum number of layers**.

**Output Format:** Your final answer should be a single integer — the total number of valid tower configurations that achieve the maximum number of layers."""

    def __init__(
        self,
        max_a_b: int = 10,
        num_players: int = 1,
        **kwargs: Any,
    ):
        """Initialize the monochrome block counting environment.

        Args:
            max_a_b: Maximum value for A and B (number of black/white blocks).
            num_players: Number of players (default 1).
            **kwargs: Additional arguments for the base Env class.
        """
        super().__init__(**kwargs)
        self._max_a_b = max_a_b
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._a: int | None = None
        self._b: int | None = None
        self._prompt: str | None = None
        self._oracle_answer: int | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        return dedent(
            """
            Monochrome Block Counting:

            Build a tower where the i-th layer (from top) has exactly i blocks.
            Each layer must be monochrome (all black or all white).
            Given A black blocks and B white blocks, find the maximum number of layers N
            and count the distinct ways to achieve it.

            Rules:
            - Layer i (1 ≤ i ≤ N) contains exactly i blocks of the same color
            - Total black blocks used ≤ A
            - Total white blocks used ≤ B
            - Maximize N (number of layers)
            - Count distinct configurations with maximum N

            The image shows:
            - A visual representation of the tower building problem
            - Available black blocks (A) and white blocks (B)
            - Example tower structure

            Output format: A single integer (the count of valid configurations).
            """
        ).strip()

    def _get_state_text(self) -> str:
        """Return the text representation of the current state."""
        return self._prompt or ""

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed=seed)

        self._generate()
        self._prompt = self._prompt_generate()
        self._last_image = self.render()

        state_text = self._get_state_text()
        obs = Observation(
            image=self._last_image,
            text=None,
            metadata={"state_text": state_text, "text_prompt": self._prompt},
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
        """Process the agent's answer and compute reward."""
        agent_id = next(iter(self._agent_ids))
        action_str = action[agent_id]
        reward = float(self._score_answer(action_str))
        state_text = self._get_state_text()
        obs = Observation(
            image=self._last_image,
            text=None,
            metadata={"state_text": state_text, "text_prompt": self._prompt},
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
        """Generate a random problem instance.

        Generates values A and B (number of black and white blocks available),
        then computes the maximum number of layers T and counts the number of
        distinct ways to build a tower with T layers.
        """
        max_a_b = self._max_a_b
        if max_a_b < 1:
            raise ValueError("max_a_b must be >= 1")

        self._a = int(self.np_random.integers(1, max_a_b + 1))
        self._b = int(self.np_random.integers(1, max_a_b + 1))

        A = self._a
        B = self._b

        # Find maximum number of layers T
        # Total blocks needed for T layers = 1 + 2 + ... + T = T*(T+1)/2
        T = 0
        while (T + 1) * (T + 2) // 2 <= A + B:
            T += 1

        # Dynamic programming to count ways
        # F[j] = number of ways to use exactly j black blocks for first i layers
        F = [0] * (A + 1)
        F[0] = 1
        for i in range(1, T + 1):
            # Process layers from 1 to T
            # For each layer i, we can either use i black blocks or i white blocks
            for j in range(A, i - 1, -1):
                F[j] += F[j - i]

        # Count valid configurations
        # We need T*(T+1)/2 total blocks
        # If we use j black blocks, we need T*(T+1)/2 - j white blocks
        # Valid if j <= A and T*(T+1)/2 - j <= B
        total_blocks = T * (T + 1) // 2
        min_black = max(total_blocks - B, 0)
        max_black = min(A, total_blocks)

        self._oracle_answer = sum(F[i] for i in range(min_black, max_black + 1))

    def _prompt_generate(self) -> str:
        """Generate the prompt text for this problem instance."""
        if self._a is None or self._b is None:
            raise RuntimeError("Problem not generated")
        return self.prompt_template.format(A=self._a, B=self._b)

    def _process(self, answer: str | None) -> int | None:
        """Parse the answer string into an integer.

        Args:
            answer: The answer string to parse.

        Returns:
            Parsed integer or None if invalid.
        """
        if answer is None:
            return None
        answer = answer.strip()
        try:
            int_answer = int(answer)
            return int_answer
        except ValueError:
            return None

    def _score_answer(self, answer: str) -> float:
        """Score the answer using (min/max)^beta rewarding strategy.

        Args:
            answer: The answer string to score.

        Returns:
            Score in range [0, 1] or -1.0 for wrong format.
        """
        processed_result = self._process(answer)
        if processed_result is None:
            return 0.0
        if processed_result <= 0:
            return 0.0
        a = self._oracle_answer
        b = processed_result
        beta = 10.0
        return (min(a, b) / max(a, b)) ** beta

    def render(self) -> Image.Image | list[Image.Image] | None:
        """Render a visual representation of the tower building problem.

        Creates an image showing:
        - The available blocks (A black, B white)
        - A schematic of the tower structure
        - Visual representation of layers
        """
        if self._a is None or self._b is None:
            raise RuntimeError("Problem not generated")

        # Image dimensions
        width = 600
        height = 500
        img = Image.new("RGB", (width, height), (250, 250, 250))
        draw = ImageDraw.Draw(img)

        # Load font
        font_path = None
        font_path_candidate = self.assets_dir / "DejaVuSans.ttf"
        if font_path_candidate.exists():
            font_path = str(font_path_candidate)

        if font_path:
            title_font = ImageFont.truetype(font_path, 28)
            large_font = ImageFont.truetype(font_path, 24)
            medium_font = ImageFont.truetype(font_path, 18)
            small_font = ImageFont.truetype(font_path, 14)
        else:
            title_font = ImageFont.load_default()
            large_font = ImageFont.load_default()
            medium_font = ImageFont.load_default()
            small_font = ImageFont.load_default()

        # Title
        title = "Tower Building Problem"
        bbox = draw.textbbox((0, 0), title, font=title_font)
        tw = bbox[2] - bbox[0]
        draw.text(
            (width // 2 - tw // 2, 20),
            title,
            fill=(20, 20, 80),
            font=title_font,
        )

        # Available blocks section
        y_offset = 80

        # Black blocks
        text = f"Black blocks: {self._a}"
        draw.text((50, y_offset), text, fill=(30, 30, 30), font=large_font)

        # Draw black block representation
        block_size = 30
        x_start = 250
        blocks_to_show = min(self._a, 10)
        for i in range(blocks_to_show):
            x = x_start + i * (block_size + 5)
            draw.rectangle(
                [x, y_offset, x + block_size, y_offset + block_size],
                fill=(40, 40, 40),
                outline=(10, 10, 10),
                width=2,
            )
        if self._a > 10:
            draw.text(
                (x_start + blocks_to_show * (block_size + 5) + 10, y_offset + 5),
                "...",
                fill=(30, 30, 30),
                font=large_font,
            )

        # White blocks
        y_offset += 50
        text = f"White blocks: {self._b}"
        draw.text((50, y_offset), text, fill=(30, 30, 30), font=large_font)

        # Draw white block representation
        blocks_to_show = min(self._b, 10)
        for i in range(blocks_to_show):
            x = x_start + i * (block_size + 5)
            draw.rectangle(
                [x, y_offset, x + block_size, y_offset + block_size],
                fill=(240, 240, 240),
                outline=(10, 10, 10),
                width=2,
            )
        if self._b > 10:
            draw.text(
                (x_start + blocks_to_show * (block_size + 5) + 10, y_offset + 5),
                "...",
                fill=(30, 30, 30),
                font=large_font,
            )

        # Divider line
        y_offset += 60
        draw.line((50, y_offset, width - 50, y_offset), fill=(150, 150, 150), width=2)

        # Tower structure explanation
        y_offset += 30
        text = "Tower Structure:"
        draw.text((50, y_offset), text, fill=(20, 20, 80), font=large_font)

        y_offset += 40
        text = "Layer i contains exactly i blocks (same color)"
        draw.text((70, y_offset), text, fill=(50, 50, 50), font=medium_font)

        # Example tower visualization
        y_offset += 50
        text = "Example Tower:"
        draw.text((50, y_offset), text, fill=(20, 20, 80), font=medium_font)

        # Draw a small example tower
        tower_x = 250
        tower_y = y_offset
        small_block = 20

        # Determine example tower size (small for visualization)
        max_layers = 4
        T_example = 0
        while (T_example + 1) * (T_example + 2) // 2 <= self._a + self._b:
            T_example += 1
        T_example = min(T_example, max_layers)

        # Draw layers from top to bottom
        colors_example = [(40, 40, 40), (240, 240, 240), (40, 40, 40), (240, 240, 240)]
        for layer in range(1, T_example + 1):
            num_blocks = layer
            layer_width = num_blocks * small_block
            start_x = tower_x + (max_layers * small_block - layer_width) // 2
            y_pos = tower_y + (layer - 1) * small_block

            color = colors_example[(layer - 1) % len(colors_example)]

            for block in range(num_blocks):
                x = start_x + block * small_block
                draw.rectangle(
                    [x, y_pos, x + small_block - 2, y_pos + small_block - 2],
                    fill=color,
                    outline=(10, 10, 10),
                    width=1,
                )

            # Label layer
            label = f"Layer {layer}"
            draw.text(
                (tower_x + max_layers * small_block + 20, y_pos),
                label,
                fill=(50, 50, 50),
                font=small_font,
            )

        # Footer
        footer_y = height - 50
        text = "Find the maximum layers and count valid configurations"
        bbox = draw.textbbox((0, 0), text, font=medium_font)
        tw = bbox[2] - bbox[0]
        draw.text(
            (width // 2 - tw // 2, footer_y),
            text,
            fill=(80, 80, 80),
            font=medium_font,
        )

        return img

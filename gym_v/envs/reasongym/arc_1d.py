"""ARC 1D single-turn environment backed by reasoning-gym."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from reasoning_gym.factory import create_dataset

from gym_v import Env, Observation, get_logger

logger = get_logger()

# ARC color palette (0-9)
ARC_COLORS = [
    (0, 0, 0),  # 0: black
    (0, 116, 217),  # 1: blue
    (255, 65, 54),  # 2: red
    (46, 204, 64),  # 3: green
    (255, 220, 0),  # 4: yellow
    (170, 170, 170),  # 5: gray
    (240, 18, 190),  # 6: magenta
    (255, 133, 27),  # 7: orange
    (127, 219, 255),  # 8: cyan
    (135, 12, 37),  # 9: maroon
]


class ReasoningGymArc1DEnv(Env):
    """ARC 1D puzzle using reasoning-gym's dataset.

    The player finds a rule from examples and applies it to a test input.

    Args:
        dataset_kwargs: Configuration parameters for the reasoning-gym dataset
        cell_px: Size of each cell in pixels for rendering
        padding: Padding around the grids in pixels for rendering
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    def __init__(
        self,
        dataset_kwargs: dict[str, Any] | None = None,
        cell_px: int = 24,
        padding: int = 20,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._dataset_kwargs = dataset_kwargs or {}
        self._cell_px = cell_px
        self._padding = padding

        self._seed: int | None = None
        self._dataset = None
        self._entry: dict[str, Any] | None = None
        self._entry_idx: int | None = None
        self._metadata: dict[str, Any] | None = None
        self._train_examples: list[dict] | None = None
        self._test_example: dict | None = None
        self._oracle_answer: str | None = None

    @property
    def description(self) -> str:
        """Return description for ARC 1D puzzle.

        Original reasoning-gym question format:
        ```
        Find the common rule that maps an input grid to an output grid,
        given the examples below.

        Example 1:
        Input:  0 0 0 2 9 2 3 4 4 0
        Output: 2 9 2 3 4 4 0 0 0 0

        Example 2:
        Input:  0 0 0 0 4 4 2 1 1 0
        Output: 0 4 4 2 1 1 0 0 0 0

        Example 3:
        Input:  0 0 0 7 9 4 9 1 0 0
        Output: 7 9 4 9 1 0 0 0 0 0

        Below is a test input grid. Predict the corresponding output grid
        by applying the rule you found.

        Input:
        0 0 0 0 0 1 5 0 0 0
        ```

        Original reasoning-gym answer format:
        ```
        0 0 1 5 0 0 0 0 0 0
        ```
        (Space-separated integers)
        """
        num_examples = len(self._train_examples) if self._train_examples else 0

        return dedent(f"""
            Find the rule that maps input to output, then apply it to the test input.

            In the image (ARC 1D - Pattern Recognition):
            - {num_examples} training examples (Example 1, 2, 3) showing INPUT → OUTPUT
            - Each example has colored cells representing numbers (0-9)
            - TEST input at bottom (red label) with "?" for unknown output

            Your task: Analyze the pattern in the examples and predict the test output.

            Output format: Space-separated integers.
            Example: 0 0 1 5 0 0 0 0 0 0
        """).strip()

    def _make_dataset(self, *, seed: int | None):
        kwargs = self._dataset_kwargs.copy()
        if seed is not None and "seed" not in kwargs:
            kwargs["seed"] = seed
        if "size" not in kwargs:
            kwargs["size"] = 500

        return create_dataset("arc_1d", **kwargs)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)
        self._seed = seed

        self._dataset = self._make_dataset(seed=self._seed)
        self._entry_idx = int(self.np_random.integers(0, len(self._dataset)))
        self._entry = self._dataset[self._entry_idx]

        self._oracle_answer = self._entry["answer"]
        self._metadata = self._entry.get("metadata", {})
        self._train_examples = self._metadata.get("train_examples", [])
        self._test_example = self._metadata.get("test_example", {})

        logger.info("Reset ReasoningGym ARC 1D.")

        # Format text representation
        text = self._format_examples()

        obs = Observation(
            image=self.render(),
            text=text,
            metadata=self._metadata,
        )
        info = {
            "reasoning_gym_seed": self._seed,
            "reasoning_gym_index": self._entry_idx,
            "oracle_answer": self._oracle_answer,
        }
        return obs, info

    def _format_examples(self) -> str:
        """Format examples as text."""
        lines = []
        for i, ex in enumerate(self._train_examples or []):
            inp = " ".join(str(x) for x in ex.get("input", []))
            out = " ".join(str(x) for x in ex.get("output", []))
            lines.append(f"Example {i+1}:")
            lines.append(f"  Input:  {inp}")
            lines.append(f"  Output: {out}")
        if self._test_example:
            inp = " ".join(str(x) for x in self._test_example.get("input", []))
            lines.append("Test:")
            lines.append(f"  Input:  {inp}")
        return "\n".join(lines)

    def inner_step(
        self, action: str
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        answer = action
        reward = self._dataset.score_answer(answer=answer, entry=self._entry)

        obs = Observation(
            image=self.render(),
            text=None,
            metadata=self._metadata,
        )
        info = {
            "reasoning_gym_seed": self._seed,
            "reasoning_gym_index": self._entry_idx,
            "oracle_answer": self._oracle_answer,
        }

        return obs, reward, True, False, info

    def render(self) -> Image.Image:
        return self._render_arc_1d(
            self._train_examples,
            self._test_example,
            cell_px=self._cell_px,
            padding=self._padding,
        )

    def _render_arc_1d(
        self,
        train_examples: list[dict],
        test_example: dict,
        cell_px: int = 28,
        padding: int = 24,
        bg: tuple[int, int, int] = (245, 248, 250),
        label_color: tuple[int, int, int] = (60, 60, 60),
    ) -> Image.Image:
        if not train_examples and not test_example:
            return Image.new("RGB", (200, 200), bg)

        font_path = self.assets_dir / "DejaVuSans.ttf"
        if font_path.exists():
            title_font = ImageFont.truetype(str(font_path), 18)
            font = ImageFont.truetype(str(font_path), 14)
            small_font = ImageFont.truetype(str(font_path), 12)
        else:
            logger.warning(f"Font file not found: {font_path}, using default font")
            font = ImageFont.load_default()
            small_font = font
            title_font = font

        # Calculate dimensions
        max_len = 0
        for ex in train_examples or []:
            max_len = max(max_len, len(ex.get("input", [])), len(ex.get("output", [])))
        if test_example:
            max_len = max(max_len, len(test_example.get("input", [])))

        # Layout parameters
        header_h = 60
        row_height = cell_px + 16
        label_width = int(cell_px * 3.5)
        arrow_width = int(cell_px * 2)
        cell_border = 2

        num_train = len(train_examples) if train_examples else 0
        total_rows = num_train + (1 if test_example else 0)

        width = (
            padding * 2
            + label_width
            + max_len * cell_px
            + arrow_width
            + max_len * cell_px
            + 40
        )
        height = padding * 2 + header_h + total_rows * row_height + row_height

        img = Image.new("RGB", (width, height), bg)
        draw = ImageDraw.Draw(img)

        # Draw title
        title = "ARC 1D - Pattern Recognition"
        bbox = draw.textbbox((0, 0), title, font=title_font)
        tw = bbox[2] - bbox[0]
        draw.text(
            (width // 2 - tw // 2, padding),
            title,
            fill=(40, 90, 140),
            font=title_font,
        )

        # Draw subtitle
        subtitle = "Find the rule from examples and apply to test"
        bbox = draw.textbbox((0, 0), subtitle, font=small_font)
        tw = bbox[2] - bbox[0]
        draw.text(
            (width // 2 - tw // 2, padding + 28),
            subtitle,
            fill=(100, 100, 100),
            font=small_font,
        )

        y = padding + header_h

        # Draw column headers
        header_y = y - 24
        draw.text(
            (padding + label_width + max_len * cell_px // 2 - 20, header_y),
            "INPUT",
            fill=(80, 80, 80),
            font=small_font,
        )
        draw.text(
            (
                padding
                + label_width
                + max_len * cell_px
                + arrow_width
                + max_len * cell_px // 2
                - 25,
                header_y,
            ),
            "OUTPUT",
            fill=(80, 80, 80),
            font=small_font,
        )

        # Draw training examples with better styling
        for i, ex in enumerate(train_examples or []):
            inp = ex.get("input", [])
            out = ex.get("output", [])

            # Example label with background
            label = f"Example {i+1}"
            label_bg_x = padding
            label_bg_w = label_width - 5
            draw.rectangle(
                [label_bg_x, y - 2, label_bg_x + label_bg_w, y + cell_px + 2],
                fill=(220, 230, 240),
                outline=(180, 190, 200),
                width=1,
            )
            draw.text(
                (padding + 8, y + cell_px // 2 - 6),
                label,
                fill=(40, 80, 120),
                font=small_font,
            )

            # Input cells with border
            x = padding + label_width
            input_x_start = x
            for j, val in enumerate(inp):
                color = ARC_COLORS[val % len(ARC_COLORS)]
                draw.rectangle(
                    [
                        x + j * cell_px,
                        y,
                        x + (j + 1) * cell_px - cell_border,
                        y + cell_px - cell_border,
                    ],
                    fill=color,
                    outline=(120, 120, 120),
                    width=1,
                )
                # Draw number on the cell
                num_text = str(val)
                text_color = (255, 255, 255) if val in [0, 1, 2, 9] else (0, 0, 0)
                bbox = draw.textbbox((0, 0), num_text, font=small_font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                text_x = x + j * cell_px + (cell_px - cell_border) // 2 - text_w // 2
                text_y = y + (cell_px - cell_border) // 2 - text_h // 2
                draw.text((text_x, text_y), num_text, fill=text_color, font=small_font)

            # Draw border around input
            draw.rectangle(
                [
                    input_x_start - 2,
                    y - 2,
                    input_x_start + max_len * cell_px + 2,
                    y + cell_px + 2,
                ],
                outline=(100, 120, 140),
                width=2,
            )

            # Arrow with better styling
            arrow_x = x + max_len * cell_px + arrow_width // 2
            draw.text(
                (arrow_x - 8, y + cell_px // 2 - 8),
                "→",
                fill=(100, 150, 100),
                font=font,
            )

            # Output cells with border
            x = padding + label_width + max_len * cell_px + arrow_width
            output_x_start = x
            for j, val in enumerate(out):
                color = ARC_COLORS[val % len(ARC_COLORS)]
                draw.rectangle(
                    [
                        x + j * cell_px,
                        y,
                        x + (j + 1) * cell_px - cell_border,
                        y + cell_px - cell_border,
                    ],
                    fill=color,
                    outline=(120, 120, 120),
                    width=1,
                )
                # Draw number on the cell
                num_text = str(val)
                text_color = (255, 255, 255) if val in [0, 1, 2, 9] else (0, 0, 0)
                bbox = draw.textbbox((0, 0), num_text, font=small_font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                text_x = x + j * cell_px + (cell_px - cell_border) // 2 - text_w // 2
                text_y = y + (cell_px - cell_border) // 2 - text_h // 2
                draw.text((text_x, text_y), num_text, fill=text_color, font=small_font)

            # Draw border around output
            draw.rectangle(
                [
                    output_x_start - 2,
                    y - 2,
                    output_x_start + max_len * cell_px + 2,
                    y + cell_px + 2,
                ],
                outline=(100, 140, 100),
                width=2,
            )

            y += row_height

        # Separator with "Apply Rule" text
        if train_examples and test_example:
            y += 10
            sep_y = y + 8
            draw.line(
                [(padding, sep_y), (width // 2 - 60, sep_y)],
                fill=(180, 180, 180),
                width=2,
            )
            draw.line(
                [(width // 2 + 60, sep_y), (width - padding, sep_y)],
                fill=(180, 180, 180),
                width=2,
            )

            apply_text = "Apply Rule"
            bbox = draw.textbbox((0, 0), apply_text, font=small_font)
            tw = bbox[2] - bbox[0]
            draw.text(
                (width // 2 - tw // 2, sep_y - 8),
                apply_text,
                fill=(200, 100, 50),
                font=small_font,
            )
            y += 26

        # Draw test example with emphasis
        if test_example:
            inp = test_example.get("input", [])

            # Test label with red background
            label = "TEST"
            label_bg_x = padding
            label_bg_w = label_width - 5
            draw.rectangle(
                [label_bg_x, y - 2, label_bg_x + label_bg_w, y + cell_px + 2],
                fill=(255, 220, 220),
                outline=(220, 100, 100),
                width=2,
            )
            draw.text(
                (padding + 8, y + cell_px // 2 - 6),
                label,
                fill=(180, 40, 40),
                font=font,
            )

            # Input cells
            x = padding + label_width
            input_x_start = x
            for j, val in enumerate(inp):
                color = ARC_COLORS[val % len(ARC_COLORS)]
                draw.rectangle(
                    [
                        x + j * cell_px,
                        y,
                        x + (j + 1) * cell_px - cell_border,
                        y + cell_px - cell_border,
                    ],
                    fill=color,
                    outline=(120, 120, 120),
                    width=1,
                )
                # Draw number on the cell
                num_text = str(val)
                text_color = (255, 255, 255) if val in [0, 1, 2, 9] else (0, 0, 0)
                bbox = draw.textbbox((0, 0), num_text, font=small_font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                text_x = x + j * cell_px + (cell_px - cell_border) // 2 - text_w // 2
                text_y = y + (cell_px - cell_border) // 2 - text_h // 2
                draw.text((text_x, text_y), num_text, fill=text_color, font=small_font)

            # Draw border around input
            draw.rectangle(
                [
                    input_x_start - 2,
                    y - 2,
                    input_x_start + max_len * cell_px + 2,
                    y + cell_px + 2,
                ],
                outline=(180, 100, 100),
                width=2,
            )

            # Arrow
            arrow_x = x + max_len * cell_px + arrow_width // 2
            draw.text(
                (arrow_x - 8, y + cell_px // 2 - 8),
                "→",
                fill=(150, 100, 100),
                font=font,
            )

            # Question mark box
            x = padding + label_width + max_len * cell_px + arrow_width
            qmark_w = max_len * cell_px
            draw.rectangle(
                [x - 2, y - 2, x + qmark_w + 2, y + cell_px + 2],
                fill=(255, 255, 240),
                outline=(200, 150, 100),
                width=2,
            )
            draw.text(
                (x + qmark_w // 2 - 10, y + cell_px // 2 - 10),
                "?",
                fill=(200, 100, 50),
                font=title_font,
            )

        # Draw outer border
        draw.rectangle(
            [2, 2, width - 3, height - 3],
            outline=(120, 140, 160),
            width=3,
        )

        return img

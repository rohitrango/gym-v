"""ARC-AGI single-turn environment backed by reasoning-gym."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from reasoning_gym.factory import create_dataset

from gym_v import Env, Observation, get_logger

from .arc_1d import ARC_COLORS

logger = get_logger()


class ArcAgiEnv(Env):
    # Meta: source=ReasoningGym, category=arc, turn=single
    """ARC-AGI puzzle using reasoning-gym's arc_agi dataset.

    2D grid pattern recognition: find the rule from examples and apply it to a
    test input.  Grids are up to 30x30 with 10 colours (0-9).

    Args:
        dataset_kwargs: Configuration parameters for the reasoning-gym dataset
        cell_px: Size of each cell in pixels for rendering
        padding: Padding around the grids in pixels for rendering
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    def __init__(
        self,
        dataset_kwargs: dict[str, Any] | None = None,
        cell_px: int = 16,
        padding: int = 16,
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._dataset_kwargs = dataset_kwargs or {}
        self._cell_px = cell_px
        self._padding = padding
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._seed: int | None = None
        self._dataset = None
        self._entry: dict[str, Any] | None = None
        self._entry_idx: int | None = None
        self._metadata: dict[str, Any] | None = None
        self._train_examples: list[dict] | None = None
        self._test_input: tuple | None = None
        self._test_output: tuple | None = None
        self._oracle_answer: str | None = None

    @property
    def description(self) -> str:
        num_examples = len(self._train_examples) if self._train_examples else 0
        rows = len(self._test_input) if self._test_input else "?"
        cols = len(self._test_input[0]) if self._test_input else "?"

        return dedent(f"""
            Find the rule that maps input grids to output grids, then apply it to the test input.

            In the image (ARC-AGI - 2D Pattern Recognition):
            - {num_examples} training examples showing INPUT → OUTPUT as coloured 2D grids
            - Each cell uses colours 0-9 (black, blue, red, green, yellow, grey, magenta, orange, cyan, maroon)
            - TEST input ({rows}x{cols}) at the bottom with "?" for the unknown output

            Your task: Analyse the pattern in the examples and predict the test output.

            Output format: Space-separated integers, one row per line.
            Example:
            0 0 1
            2 3 0
            0 0 1
        """).strip()

    def _make_dataset(self, *, seed: int | None):
        kwargs = self._dataset_kwargs.copy()
        if seed is not None and "seed" not in kwargs:
            kwargs["seed"] = seed
        if "size" not in kwargs:
            kwargs["size"] = 500
        return create_dataset("arc_agi", **kwargs)

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
        self._test_input = self._metadata.get("input")
        self._test_output = self._metadata.get("output")

        # Retrieve raw train examples from dataset's internal tasks
        task_id = self._metadata.get("task_id")
        if hasattr(self._dataset, "_tasks") and task_id in self._dataset._tasks:
            self._train_examples = self._dataset._tasks[task_id]["train"]
        else:
            self._train_examples = []

        logger.info("Reset ReasoningGym ARC-AGI.")

        text = self._format_examples()

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
        return {aid: obs for aid in self._agent_ids}, {
            aid: info for aid in self._agent_ids
        }

    def _format_examples(self) -> str:
        lines = []
        for i, ex in enumerate(self._train_examples or []):
            lines.append(f"Example {i + 1}:")
            lines.append("  Input:")
            for row in ex.get("input", []):
                lines.append("    " + " ".join(str(x) for x in row))
            lines.append("  Output:")
            for row in ex.get("output", []):
                lines.append("    " + " ".join(str(x) for x in row))
        if self._test_input:
            lines.append("Test Input:")
            for row in self._test_input:
                lines.append("    " + " ".join(str(x) for x in row))
        return "\n".join(lines)

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
        if not answer or not answer.strip():
            reward = 0.0
        else:
            reward = self._dataset.score_answer(answer=answer, entry=self._entry)

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
            {aid: obs for aid in self._agent_ids},
            {aid: reward for aid in self._agent_ids},
            {**{aid: terminated for aid in self._agent_ids}, "__all__": terminated},
            {**{aid: truncated for aid in self._agent_ids}, "__all__": truncated},
            {aid: info for aid in self._agent_ids},
        )

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self) -> Image.Image | None:
        return self._render_arc_2d(
            self._train_examples or [],
            self._test_input,
            cell_px=self._cell_px,
            padding=self._padding,
        )

    def _render_arc_2d(
        self,
        train_examples: list[dict],
        test_input: tuple | None,
        cell_px: int = 16,
        padding: int = 16,
        bg: tuple[int, int, int] = (245, 248, 250),
    ) -> Image.Image:
        if not train_examples and test_input is None:
            return Image.new("RGB", (200, 200), bg)

        font_path = self.assets_dir / "DejaVuSans.ttf"
        if font_path.exists():
            title_font = ImageFont.truetype(str(font_path), 16)
            font = ImageFont.truetype(str(font_path), 12)
            small_font = ImageFont.truetype(str(font_path), 10)
        else:
            font = ImageFont.load_default()
            small_font = font
            title_font = font

        # Collect all grids to compute max dimensions
        all_grids: list[list | tuple] = []
        for ex in train_examples:
            all_grids.append(ex.get("input", []))
            all_grids.append(ex.get("output", []))
        if test_input:
            all_grids.append(test_input)

        max_rows = max((len(g) for g in all_grids), default=1)
        max_cols = max((len(g[0]) if g else 0 for g in all_grids), default=1)

        # Layout: for each example row: label | input grid | arrow | output grid
        label_w = 80
        arrow_w = 36
        grid_w = max_cols * cell_px
        grid_h = max_rows * cell_px
        row_h = grid_h + 28

        header_h = 44
        num_rows = len(train_examples) + (1 if test_input else 0)
        sep_h = 20 if train_examples and test_input else 0

        width = padding * 2 + label_w + grid_w + arrow_w + grid_w + 10
        height = padding * 2 + header_h + num_rows * row_h + sep_h

        img = Image.new("RGB", (width, height), bg)
        draw = ImageDraw.Draw(img)

        # Title
        title = "ARC-AGI \u2014 2D Pattern Recognition"
        bbox = draw.textbbox((0, 0), title, font=title_font)
        tw = bbox[2] - bbox[0]
        draw.text(
            (width // 2 - tw // 2, padding), title, fill=(40, 90, 140), font=title_font
        )

        subtitle = "Find the rule from examples and apply to test"
        bbox = draw.textbbox((0, 0), subtitle, font=small_font)
        tw = bbox[2] - bbox[0]
        draw.text(
            (width // 2 - tw // 2, padding + 22),
            subtitle,
            fill=(100, 100, 100),
            font=small_font,
        )

        y = padding + header_h

        def _draw_grid(grid, x0, y0, border_color=(100, 120, 140)):
            rows = len(grid)
            cols = len(grid[0]) if grid else 0
            for r in range(rows):
                for c in range(cols):
                    val = grid[r][c]
                    color = ARC_COLORS[val % len(ARC_COLORS)]
                    draw.rectangle(
                        [
                            x0 + c * cell_px,
                            y0 + r * cell_px,
                            x0 + (c + 1) * cell_px - 1,
                            y0 + (r + 1) * cell_px - 1,
                        ],
                        fill=color,
                    )
            draw.rectangle(
                [x0 - 1, y0 - 1, x0 + cols * cell_px, y0 + rows * cell_px],
                outline=border_color,
                width=2,
            )

        # Training examples
        for i, ex in enumerate(train_examples):
            inp = ex.get("input", [])
            out = ex.get("output", [])

            label = f"Ex {i + 1}"
            draw.rectangle(
                [padding, y, padding + label_w - 6, y + grid_h],
                fill=(220, 230, 240),
                outline=(180, 190, 200),
                width=1,
            )
            draw.text(
                (padding + 6, y + grid_h // 2 - 6),
                label,
                fill=(40, 80, 120),
                font=small_font,
            )

            gx = padding + label_w
            _draw_grid(inp, gx, y)

            ax = gx + grid_w + arrow_w // 2 - 6
            draw.text(
                (ax, y + grid_h // 2 - 6), "\u2192", fill=(100, 150, 100), font=font
            )

            ox = padding + label_w + grid_w + arrow_w
            _draw_grid(out, ox, y, border_color=(100, 140, 100))

            y += row_h

        # Separator
        if train_examples and test_input:
            sep_y = y + sep_h // 2
            draw.line(
                [(padding, sep_y), (width - padding, sep_y)],
                fill=(180, 180, 180),
                width=1,
            )
            y += sep_h

        # Test input
        if test_input:
            label = "TEST"
            draw.rectangle(
                [padding, y, padding + label_w - 6, y + grid_h],
                fill=(255, 220, 220),
                outline=(220, 100, 100),
                width=2,
            )
            draw.text(
                (padding + 6, y + grid_h // 2 - 6), label, fill=(180, 40, 40), font=font
            )

            gx = padding + label_w
            _draw_grid(test_input, gx, y, border_color=(180, 100, 100))

            ax = gx + grid_w + arrow_w // 2 - 6
            draw.text(
                (ax, y + grid_h // 2 - 6), "\u2192", fill=(150, 100, 100), font=font
            )

            ox = padding + label_w + grid_w + arrow_w
            draw.rectangle(
                [ox - 1, y - 1, ox + grid_w, y + grid_h],
                fill=(255, 255, 240),
                outline=(200, 150, 100),
                width=2,
            )
            draw.text(
                (ox + grid_w // 2 - 8, y + grid_h // 2 - 10),
                "?",
                fill=(200, 100, 50),
                font=title_font,
            )

        draw.rectangle([1, 1, width - 2, height - 2], outline=(120, 140, 160), width=2)
        return img

"""Tower of Hanoi single-turn environment backed by reasoning-gym."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from reasoning_gym.factory import create_dataset

from gym_v import Env, Observation, get_logger

logger = get_logger()


class ReasoningGymTowerOfHanoiEnv(Env):
    """Tower of Hanoi puzzle using reasoning-gym's dataset.

    The player must move all disks from the start peg to the target peg,
    following the classic Tower of Hanoi rules.

    Args:
        dataset_kwargs: Configuration parameters for the reasoning-gym dataset
        peg_width: Width of each peg in pixels
        peg_height: Height of the peg area in pixels
        padding: Padding around the visualization in pixels
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    # Color palette for disks (up to 10 disks)
    DISK_COLORS = [
        (231, 76, 60),  # Red
        (241, 196, 15),  # Yellow
        (46, 204, 113),  # Green
        (52, 152, 219),  # Blue
        (155, 89, 182),  # Purple
        (230, 126, 34),  # Orange
        (26, 188, 156),  # Teal
        (241, 148, 138),  # Light Red
        (133, 193, 233),  # Light Blue
        (169, 204, 227),  # Pale Blue
    ]

    def __init__(
        self,
        dataset_kwargs: dict[str, Any] | None = None,
        peg_width: int = 150,
        peg_height: int = 250,
        padding: int = 40,
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._dataset_kwargs = dataset_kwargs or {}
        self._peg_width = peg_width
        self._peg_height = peg_height
        self._padding = padding
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._seed: int | None = None
        self._dataset = None
        self._entry: dict[str, Any] | None = None
        self._entry_idx: int | None = None
        self._metadata: dict[str, Any] | None = None
        self._oracle_answer: str | None = None

    @property
    def description(self) -> str:
        """Return description with current puzzle parameters.

        Original reasoning-gym question format:
        ```
        Solve the Tower of Hanoi problem with {num_disks} disks and {num_pegs} pegs.
        Move all disks from Peg {start_peg} to Peg {target_peg} following the rules:
        - Only one disk can be moved at a time.
        - A larger disk cannot be placed on top of a smaller disk.
        - All disks must be on a peg at all times.

        Provide the sequence of moves.

        Formatting guidelines:
        - Each instruction should be placed on a single line.
        - Each line should be formatted as 'Move disk X from Peg Y to Peg Z'
        - Do not include any other text or formatting.
        ```

        Original reasoning-gym answer format:
        ```
        Move disk 1 from Peg 1 to Peg 3
        Move disk 2 from Peg 1 to Peg 2
        Move disk 1 from Peg 3 to Peg 2
        Move disk 3 from Peg 1 to Peg 3
        Move disk 1 from Peg 2 to Peg 1
        Move disk 2 from Peg 2 to Peg 3
        Move disk 1 from Peg 1 to Peg 3
        ```
        (Each move on a new line, format: "Move disk X from Peg Y to Peg Z")
        """
        num_disks = self._metadata.get("num_disks", 3) if self._metadata else 3
        num_pegs = self._metadata.get("num_pegs", 3) if self._metadata else 3
        start_peg = self._metadata.get("start_peg", 1) if self._metadata else 1
        target_peg = self._metadata.get("target_peg", 3) if self._metadata else 3

        return dedent(f"""
            Solve the Tower of Hanoi problem with {num_disks} disks and {num_pegs} pegs.

            In the image:
            - Disks are shown as colored rectangles with numbers
            - All disks start on Peg {start_peg}
            - Each peg is labeled at the bottom

            Goal: Move all disks from Peg {start_peg} to Peg {target_peg}.

            Rules:
            - Only one disk can be moved at a time
            - A larger disk cannot be placed on top of a smaller disk
            - All disks must be on a peg at all times

            Output format: Each move on a single line as:
            Move disk X from Peg Y to Peg Z

            Example:
            Move disk 1 from Peg 1 to Peg 3
            Move disk 2 from Peg 1 to Peg 2
        """).strip()

    def _make_dataset(self, *, seed: int | None):
        kwargs = self._dataset_kwargs.copy()
        if seed is not None and "seed" not in kwargs:
            kwargs["seed"] = seed
        if "size" not in kwargs:
            kwargs["size"] = 500

        return create_dataset("tower_of_hanoi", **kwargs)

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

        logger.info("Reset ReasoningGym Tower of Hanoi.")

        # obs.text = None (原始 reasoning-gym question 没有显示初始状态)
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
        answer = action[agent_id]
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
        return self._render_hanoi(
            peg_width=self._peg_width,
            peg_height=self._peg_height,
            padding=self._padding,
        )

    def _render_hanoi(
        self,
        peg_width: int = 150,
        peg_height: int = 250,
        padding: int = 40,
        bg: tuple[int, int, int] = (250, 250, 250),
        peg_color: tuple[int, int, int] = (139, 90, 43),
        base_color: tuple[int, int, int] = (101, 67, 33),
    ) -> Image.Image:
        num_pegs = self._metadata.get("num_pegs", 3)
        num_disks = self._metadata.get("num_disks", 3)
        start_peg = self._metadata.get("start_peg", 1)

        # Calculate image dimensions
        total_width = padding * 2 + peg_width * num_pegs + padding * (num_pegs - 1)
        total_height = padding * 2 + peg_height + 60  # Extra for labels

        img = Image.new("RGB", (total_width, total_height), bg)
        draw = ImageDraw.Draw(img)

        font_path = self.assets_dir / "DejaVuSans.ttf"
        if font_path.exists():
            font = ImageFont.truetype(str(font_path), 16)
            label_font = ImageFont.truetype(str(font_path), 14)
        else:
            logger.warning(f"Font file not found: {font_path}, using default font")
            font = ImageFont.load_default()
            label_font = font

        # Build initial peg state (all disks on start peg)
        pegs_state: dict[int, list[int]] = {i: [] for i in range(1, num_pegs + 1)}
        for disk in range(num_disks, 0, -1):
            pegs_state[start_peg].append(disk)

        disk_height = min(30, (peg_height - 40) // max(num_disks, 1))
        max_disk_width = peg_width - 20
        min_disk_width = 30

        # Draw each peg
        for peg_idx in range(1, num_pegs + 1):
            peg_x = padding + (peg_idx - 1) * (peg_width + padding)
            peg_center_x = peg_x + peg_width // 2
            base_y = padding + peg_height

            # Draw base
            base_rect = [peg_x, base_y, peg_x + peg_width, base_y + 15]
            draw.rectangle(base_rect, fill=base_color, outline=(80, 50, 20), width=2)

            # Draw peg pole
            pole_width = 12
            pole_rect = [
                peg_center_x - pole_width // 2,
                padding + 20,
                peg_center_x + pole_width // 2,
                base_y,
            ]
            draw.rectangle(pole_rect, fill=peg_color, outline=(100, 60, 30), width=1)

            # Draw disks on this peg
            disks = pegs_state.get(peg_idx, [])
            for i, disk_size in enumerate(disks):
                # Calculate disk width based on size
                width_range = max_disk_width - min_disk_width
                disk_width = min_disk_width + int((disk_size / num_disks) * width_range)

                disk_x = peg_center_x - disk_width // 2
                disk_y = base_y - (i + 1) * disk_height

                color = self.DISK_COLORS[(disk_size - 1) % len(self.DISK_COLORS)]
                draw.rounded_rectangle(
                    [disk_x, disk_y, disk_x + disk_width, disk_y + disk_height - 2],
                    radius=5,
                    fill=color,
                    outline=(0, 0, 0),
                    width=1,
                )

                # Draw disk number
                disk_label = str(disk_size)
                bbox = draw.textbbox((0, 0), disk_label, font=label_font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                draw.text(
                    (peg_center_x - tw // 2, disk_y + (disk_height - th) // 2 - 1),
                    disk_label,
                    fill=(255, 255, 255),
                    font=label_font,
                )

            # Draw peg label
            label = f"Peg {peg_idx}"
            bbox = draw.textbbox((0, 0), label, font=font)
            tw = bbox[2] - bbox[0]
            draw.text(
                (peg_center_x - tw // 2, base_y + 25),
                label,
                fill=(60, 60, 60),
                font=font,
            )

        return img

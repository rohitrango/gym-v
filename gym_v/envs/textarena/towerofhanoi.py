"""TowerOfHanoi game using TextArena."""

from __future__ import annotations

import colorsys
from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont
import textarena as ta

from gym_v import Env, Observation, get_logger

logger = get_logger()


class TextArenaTowerOfHanoiEnv(Env):
    """Tower of Hanoi mathematical puzzle game using TextArena's TowerOfHanoi environment.

    The player moves disks between three towers (A, B, C) to transfer all disks
    from the source tower to the target tower. Only one disk can be moved at a time,
    and a larger disk cannot be placed on top of a smaller disk. The goal is to
    move all disks from tower A to tower C following these rules.

    Args:
        num_disks: Number of disks to move
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    def __init__(
        self,
        num_disks: int = 3,
        num_players: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._num_disks = num_disks
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._ta_env = ta.make(
            "TowerOfHanoi-v0-raw",
            num_disks=num_disks,
            max_turns=self._max_episode_steps - 1,
        )

    @property
    def description(self) -> str:
        return dedent(f"""
            You are playing Tower of Hanoi with {self._num_disks} disks.
            You have to move the disks from tower A to tower C.
            To move a disk, type the source tower and the target tower (e.g., '[A C]').
            Note that you can only move the top disk of a tower, and that a bigger disk cannot be placed on a smaller disk.
            At each turn, submit one move.
        """).strip()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        super().reset(seed=seed)

        self._ta_env.reset(num_players=self.num_players, seed=seed)

        logger.info("Reset TowerofHanoi.")

        obs = Observation(image=self.render(), text=self._get_observation_text())
        info = {}

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
        info = {}
        done, _ = self._ta_env.step(action_str)

        info["invalid_action"] = (
            self._ta_env.state.error_count > 0
            or self._ta_env.state.game_info[0]["invalid_move"]
        )

        if done:
            reward = self._ta_env.state.rewards[0]
            terminated = True
            truncated = False
        else:
            reward = 0
            terminated = False
            truncated = False

        obs = Observation(image=self.render(), text=self._get_observation_text())

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
        # Constants for rendering
        tower_width = 200
        tower_height = 300
        base_height = 25
        pole_width = 12
        margin = 60
        disk_height = 25

        # Calculate image dimensions
        img_width = 3 * tower_width + 4 * margin
        img_height = tower_height + base_height + 2 * margin

        # Create base image
        img = Image.new("RGB", (img_width, img_height), (240, 240, 240))
        draw = ImageDraw.Draw(img)

        # Get current tower state
        towers = self._ta_env.state.game_state["towers"]

        tower_names = ["A", "B", "C"]
        tower_labels = ["Source (A)", "Helper (B)", "Target (C)"]

        for i, (tower_name, tower_label) in enumerate(
            zip(tower_names, tower_labels, strict=False)
        ):
            # Calculate tower position
            tower_x = margin + i * (tower_width + margin)
            tower_center_x = tower_x + tower_width // 2

            # Draw base
            base_y = img_height - margin - base_height
            base_rect = [tower_x, base_y, tower_x + tower_width, base_y + base_height]
            draw.rectangle(
                base_rect, fill=(139, 69, 19), outline=(101, 67, 33), width=2
            )

            # Draw pole
            pole_x = tower_center_x - pole_width // 2
            pole_y = base_y - tower_height
            pole_rect = [pole_x, pole_y, pole_x + pole_width, base_y]
            draw.rectangle(
                pole_rect, fill=(101, 67, 33), outline=(139, 69, 19), width=1
            )

            # Draw tower label
            font_path = self.assets_dir / "DejaVuSans.ttf"
            if font_path.exists():
                font = ImageFont.truetype(str(font_path), 16)
            else:
                logger.warning(f"Font file not found: {font_path}, using default font")
                font = ImageFont.load_default()

            bbox = draw.textbbox((0, 0), tower_label, font=font)
            text_width = bbox[2] - bbox[0]
            text_x = tower_center_x - text_width // 2
            text_y = pole_y - 25

            draw.text((text_x, text_y), tower_label, fill=(50, 50, 50), font=font)

            # Draw disks on this tower
            tower_disks = towers[tower_name]
            for j, disk_size in enumerate(tower_disks):
                # Calculate disk position (from bottom up)
                disk_y = base_y - (j + 1) * disk_height

                # Calculate disk width based on size
                max_disk_width = tower_width - 30  # Leave some margin
                min_disk_width = 40  # Minimum disk width
                disk_width = max(
                    min_disk_width, max_disk_width * disk_size // self._num_disks
                )
                disk_x = tower_center_x - disk_width // 2

                # Get disk color
                disk_color = self._get_disk_color(disk_size)

                # Draw disk with rounded edges
                disk_rect = [
                    disk_x,
                    disk_y,
                    disk_x + disk_width,
                    disk_y + disk_height - 2,
                ]
                draw.rounded_rectangle(
                    disk_rect, radius=5, fill=disk_color, outline=(0, 0, 0), width=2
                )

        return img

    def _get_disk_color(self, disk_size: int) -> tuple[int, int, int]:
        # Use hue variation to create distinct colors
        hue = (disk_size - 1) / max(self._num_disks, 1)  # Normalize to 0-1 range
        saturation = 0.8
        value = 0.9

        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        return (int(r * 255), int(g * 255), int(b * 255))

    def _get_observation_text(self) -> str:
        _, ta_obs = self._ta_env.get_observation()
        obs_text = []

        for _, msg, type in ta_obs:
            if type in [
                ta.ObservationType.GAME_ADMIN,
                ta.ObservationType.GAME_ACTION_DESCRIPTION,
            ]:
                obs_text.append(msg)

        if "reason" in self._ta_env.state.game_info[0]:
            obs_text.append(self._ta_env.state.game_info[0]["reason"])

        obs_text = "\n".join(obs_text) if obs_text else None

        return obs_text

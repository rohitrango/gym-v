"""Sokoban game using TextArena."""

from __future__ import annotations

from functools import cached_property
from importlib import resources
from textwrap import dedent
from typing import Any

import numpy as np
from PIL import Image
import textarena as ta

from gym_v import Env, Observation, get_logger

logger = get_logger()


class SokobanEnv(Env):
    # Meta: source=TextArena, category=games, turn=multi
    # Overrides: interaction_mode=multi_turn
    """Sokoban puzzle game using TextArena's Sokoban environment.

    The player pushes boxes onto targets in a warehouse.
    The game is solved when all boxes are on targets.

    Args:
        dim_room: Room dimensions (rows, cols)
        num_boxes: Number of boxes to push
        tile_size: Size of each tile in pixels for rendering
    """

    assets_dir = resources.files("gym_v.envs") / "assets" / "sokoban"

    def __init__(
        self,
        dim_room: tuple[int, int] = (6, 6),
        num_boxes: int = 3,
        tile_size: int = 48,
        num_players: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._dim_room = dim_room
        self._num_boxes = num_boxes
        self._tile_size = tile_size
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._ta_env = ta.make(
            "Sokoban-v0-raw",
            dim_room=dim_room,
            num_boxes=num_boxes,
            max_turns=self._max_episode_steps - 1,
        )

        self._initial_room_state: np.ndarray | None = None

    @property
    def description(self) -> str:
        return dedent("""
            You are solving the Sokoban puzzle. You are the player and you need to push all boxes to targets.
            When you are right next to a box, you can push it by moving in the same direction.
            You cannot push a box through a wall, and you cannot pull a box.

            You can also use [w] for up, [a] for left, [s] for down, and [d] for right.
        """).strip()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        super().reset(seed=seed)

        self._ta_env.reset(num_players=self.num_players, seed=seed)
        self._initial_room_state = self._ta_env.room_state.copy()

        logger.info("Reset Sokoban.")

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
        # Handle single player logic for TextArena
        agent_id = next(iter(self._agent_ids))
        action_str = action[agent_id]
        done, _ = self._ta_env.step(action_str)

        info = {}
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
        room_state = self._ta_env.room_state

        rows, cols = room_state.shape
        img_width = cols * self._tile_size
        img_height = rows * self._tile_size

        # Create base image
        img = Image.new("RGB", (img_width, img_height), (200, 200, 200))

        for r in range(rows):
            for c in range(cols):
                x0, y0 = c * self._tile_size, r * self._tile_size
                val = int(room_state[r, c])
                if val == 5 and self._initial_room_state[r, c] == 2:
                    val = 6

                # TextArena uses different encoding - need to map to our constants
                # This mapping should match TextArena's Sokoban encoding
                sprite = self._get_sprite_for_cell(val, self._assets)

                if sprite:
                    img.paste(sprite, (x0, y0), sprite)
                else:
                    raise ValueError(f"Invalid cell value: {val}")

        return img

    @cached_property
    def _assets(self) -> dict[str, Image.Image]:
        assets = {}

        # Asset mapping: asset_key -> filename
        asset_files = {
            "wall": "wall.png",
            "floor": "floor.png",
            "dock": "dock.png",
            "box_docked": "box_docked.png",
            "box": "box.png",
            "worker": "worker.png",
            "worker_dock": "worker_dock.png",
        }

        for asset_key, filename in asset_files.items():
            asset_path = self.assets_dir / filename
            if asset_path.exists():
                img = Image.open(asset_path).convert("RGBA")
                # Resize to tile_size
                img = img.resize(
                    (self._tile_size, self._tile_size), Image.Resampling.LANCZOS
                )
                assets[asset_key] = img
            else:
                raise FileNotFoundError(f"Asset not found: {asset_path}")

        return assets

    def _get_sprite_for_cell(self, cell_value: int, assets: dict) -> Image.Image:
        # This mapping depends on TextArena's Sokoban implementation

        # Always draw floor first for non-wall cells
        base_sprite = None
        if cell_value != 0:
            base_sprite = assets.get("floor")

        # Determine the main sprite
        if cell_value == 0:  # Wall
            return assets.get("wall")
        elif cell_value == 1:  # Empty floor
            return assets.get("floor")
        elif cell_value == 2:  # Target/dock
            return assets.get("dock")
        elif cell_value == 3:  # Box on target
            return assets.get("box_docked")
        elif cell_value == 4:  # Box
            return assets.get("box")
        elif cell_value == 5:  # Player
            return assets.get("worker")
        elif cell_value == 6:  # Player on target
            return assets.get("worker_dock")

        return base_sprite

    def _get_observation_text(self) -> str:
        _, ta_obs = self._ta_env.get_observation()
        obs_text = []

        for _, msg, type in ta_obs:
            if type in [ta.ObservationType.GAME_ADMIN, ta.ObservationType.GAME_MESSAGE]:
                obs_text.append(msg)

        if "reason" in self._ta_env.state.game_info[0]:
            obs_text.append(self._ta_env.state.game_info[0]["reason"])

        obs_text = "\n".join(obs_text) if obs_text else None

        return obs_text

"""FrozenLake game using TextArena."""

from __future__ import annotations

from functools import cached_property
from importlib import resources
from textwrap import dedent
from typing import Any

import textarena as ta
from PIL import Image

from gym_v import Env, Observation, get_logger

logger = get_logger()


class TextArenaFrozenLakeEnv(Env):
    """Frozen Lake navigation game using TextArena's FrozenLake environment.

    The player navigates across a frozen lake from start to goal while avoiding holes.

    Args:
        size: Size of the square grid (size x size)
        num_holes: Number of holes randomly placed on the lake
        randomize_start_goal: If True, randomizes start and goal positions each episode
        tile_size: Size of each tile in pixels for rendering
    """

    assets_dir = resources.files("gym_v.envs.textarena") / "assets" / "frozenlake"

    def __init__(
        self,
        size: int = 4,
        num_holes: int = 3,
        randomize_start_goal: bool = False,
        tile_size: int = 64,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._size = size
        self._num_holes = num_holes
        self._randomize_start_goal = randomize_start_goal
        self._tile_size = tile_size

        self._ta_env = ta.make(
            "FrozenLake-v0-raw",
            size=size,
            num_holes=num_holes,
            randomize_start_goal=randomize_start_goal,
        )

        self._last_direction = "DOWN"

    @property
    def description(self) -> str:
        return dedent("""
            Welcome to Frozen Lake!

            Available actions: up, down, left, right (or w, a, s, d)
            Type your action as: [up], [down], [left], [right] or [w], [a], [s], [d]

            Objective: Navigate from the start (top-left) to the goal (bottom-right) without falling into any holes!
        """).strip()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)

        self._ta_env.reset(num_players=1, seed=seed)
        self._ta_env.state.max_turns = self._max_episode_steps - 1
        self._last_direction = "DOWN"

        logger.info("Reset FrozenLake.")

        obs = Observation(image=self.render(), text=self._get_observation_text())
        info = {}

        return obs, info

    def inner_step(
        self, action: str
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        info = {}
        done, _ = self._ta_env.step(action)

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

        return obs, reward, terminated, truncated, info

    def render(self) -> Image.Image:
        grid = self._ta_env.state.game_state["grid"]
        player_pos = self._ta_env.state.game_state["player_pos"]

        size = self._tile_size * self._size
        img = Image.new("RGB", (size, size), (255, 255, 255))

        # Load and cache assets
        assets = self._assets
        pr, pc = player_pos

        for r in range(self._size):
            for c in range(self._size):
                x = c * self._tile_size
                y = r * self._tile_size

                cell_content = grid[r][c]

                # Draw base ice surface
                if "ice" in assets:
                    img.paste(assets["ice"], (x, y))

                # Draw cell content
                if cell_content == "H":
                    if "hole" in assets:
                        img.paste(assets["hole"], (x, y), assets["hole"])
                elif cell_content == "G":
                    if "goal" in assets:
                        img.paste(assets["goal"], (x, y), assets["goal"])

                # Draw player if at this position
                if (r, c) == (pr, pc):
                    # Use appropriate elf sprite based on last direction
                    elf_sprite_map = {
                        "UP": "elf_up",
                        "DOWN": "elf_down",
                        "LEFT": "elf_left",
                        "RIGHT": "elf_right",
                    }
                    elf_sprite = elf_sprite_map.get(self._last_direction, "elf_down")
                    if elf_sprite in assets:
                        img.paste(assets[elf_sprite], (x, y), assets[elf_sprite])

        return img

    @cached_property
    def _assets(self) -> dict[str, Image.Image]:
        assets = {}

        # Asset mapping: asset_key -> filename
        asset_files = {
            "ice": "ice.png",
            "hole": "hole.png",
            "goal": "goal.png",
            "elf_down": "elf_down.png",
            "elf_up": "elf_up.png",
            "elf_left": "elf_left.png",
            "elf_right": "elf_right.png",
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
                logger.debug(f"Loaded asset: {asset_key} from {filename}")
            else:
                raise FileNotFoundError(f"Asset not found: {asset_path}")

        return assets

    def _get_observation_text(self) -> str:
        _, ta_obs = self._ta_env.get_observation()
        obs_text = []

        for _, msg, type in ta_obs:
            if type in [
                ta.ObservationType.GAME_ADMIN,
                ta.ObservationType.GAME_ACTION_DESCRIPTION,
                ta.ObservationType.GAME_MESSAGE,
            ]:
                obs_text.append(msg)

        if "reason" in self._ta_env.state.game_info[0]:
            obs_text.append(self._ta_env.state.game_info[0]["reason"])

        obs_text = "\n".join(obs_text) if obs_text else None

        return obs_text

"""FrozenLake single-turn environment using TextArena."""

from __future__ import annotations

import re
from functools import cached_property
from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image
import textarena as ta

from gym_v import Env, Observation, get_logger

logger = get_logger()

VALID_ACTIONS = {"up", "down", "left", "right", "w", "a", "s", "d"}


class FrozenLakeSingleTurnEnv(Env):
    # Meta: source=TextArena, category=games, turn=single
    """FrozenLake single-turn environment.

    The agent must navigate from start to goal in a single response by
    submitting a sequence of bracketed actions, e.g. [right] [right] [down].
    The episode terminates after the sequence is applied. Reward is taken
    from the environment if the goal is reached, otherwise 0.

    Args:
        size: Size of the square grid (size x size).
        num_holes: Number of holes randomly placed on the lake.
        randomize_start_goal: If True, randomizes start and goal positions.
        tile_size: Size of each tile in pixels for rendering.
    """

    assets_dir = resources.files("gym_v.envs") / "assets" / "frozenlake"

    def __init__(
        self,
        size: int = 4,
        num_holes: int = 3,
        randomize_start_goal: bool = False,
        tile_size: int = 64,
        num_players: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._size = size
        self._num_holes = num_holes
        self._randomize_start_goal = randomize_start_goal
        self._tile_size = tile_size
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

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

            Objective: Navigate from the start (top-left) to the goal (bottom-right) without falling into holes.

            Plan your entire route and submit all moves at once as a sequence of bracketed actions.
            Use spaces or commas to separate them. Examples:
                [right] [right] [down] [down]
                [right], [down], [right], [down]

            Available actions: [up] [down] [left] [right]
            Aliases: [w] = up, [s] = down, [a] = left, [d] = right

            If you fall in a hole or reach the goal mid-sequence, the episode ends immediately.
            Reward is 1 if you reach the goal, 0 otherwise.
        """).strip()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        super().reset(seed=seed)

        self._ta_env.reset(num_players=self.num_players, seed=seed)
        self._ta_env.state.max_turns = 10000  # no turn limit; sequence length is the limit
        self._last_direction = "DOWN"

        logger.info("Reset FrozenLake single-turn.")

        text_prompt = self.description
        obs = Observation(
            image=self.render(),
            text=text_prompt,
            metadata={
                "text_prompt": text_prompt,
                "state_text": self._get_observation_text(),
            },
        )
        return {agent_id: obs for agent_id in self._agent_ids}, {
            agent_id: {} for agent_id in self._agent_ids
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

        bracketed_actions = self._parse_action_sequence(action_str)

        reward = 0.0
        done = False
        info = {}

        for bracketed in bracketed_actions:
            done, _ = self._ta_env.step(bracketed)
            # Update sprite direction
            inner = bracketed.strip("[] ").lower()
            _dir_map = {"up": "UP", "w": "UP", "down": "DOWN", "s": "DOWN",
                        "left": "LEFT", "a": "LEFT", "right": "RIGHT", "d": "RIGHT"}
            if inner in _dir_map:
                self._last_direction = _dir_map[inner]
            if done:
                reward = float(self._ta_env.state.rewards[0])
                break

        terminated = True
        truncated = False

        obs = Observation(
            image=self.render(),
            text=self._get_observation_text(),
            metadata={"text_prompt": self.description},
        )

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
        grid = self._ta_env.state.game_state["grid"]
        player_pos = self._ta_env.state.game_state["player_pos"]

        size = self._tile_size * self._size
        img = Image.new("RGB", (size, size), (255, 255, 255))

        assets = self._assets
        pr, pc = player_pos

        for r in range(self._size):
            for c in range(self._size):
                x = c * self._tile_size
                y = r * self._tile_size

                cell_content = grid[r][c]

                if "ice" in assets:
                    img.paste(assets["ice"], (x, y))

                if cell_content == "H":
                    if "hole" in assets:
                        img.paste(assets["hole"], (x, y), assets["hole"])
                elif cell_content == "G":
                    if "goal" in assets:
                        img.paste(assets["goal"], (x, y), assets["goal"])

                if (r, c) == (pr, pc):
                    elf_sprite_map = {
                        "UP": "elf_up", "DOWN": "elf_down",
                        "LEFT": "elf_left", "RIGHT": "elf_right",
                    }
                    elf_sprite = elf_sprite_map.get(self._last_direction, "elf_down")
                    if elf_sprite in assets:
                        img.paste(assets[elf_sprite], (x, y), assets[elf_sprite])

        return img

    @cached_property
    def _assets(self) -> dict[str, Image.Image]:
        assets = {}
        asset_files = {
            "ice": "ice.png", "hole": "hole.png", "goal": "goal.png",
            "elf_down": "elf_down.png", "elf_up": "elf_up.png",
            "elf_left": "elf_left.png", "elf_right": "elf_right.png",
        }
        for asset_key, filename in asset_files.items():
            asset_path = self.assets_dir / filename
            if asset_path.exists():
                img = Image.open(asset_path).convert("RGBA")
                img = img.resize((self._tile_size, self._tile_size), Image.Resampling.LANCZOS)
                assets[asset_key] = img
            else:
                raise FileNotFoundError(f"Asset not found: {asset_path}")
        return assets

    def _parse_action_sequence(self, action_str: str) -> list[str]:
        """Extract all [action] tokens, filtering to valid actions only."""
        matches = re.findall(r'\[([^\]]+)\]', action_str)
        result = []
        for token in matches:
            token = token.strip().lower()
            if token in VALID_ACTIONS:
                result.append(f"[{token}]")
        return result

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
        return "\n".join(obs_text) if obs_text else None

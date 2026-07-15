"""DoorKey environment using Minigrid."""

from __future__ import annotations

from textwrap import dedent
from typing import Any

import numpy as np
import gymnasium as gym
import minigrid  # noqa: F401 – registers MiniGrid envs
from PIL import Image

from gym_v import Env, Observation, get_logger

logger = get_logger()


class DoorKeyEnv(Env):
    # Meta: source=Minigrid, category=spatial, turn=multi
    """DoorKey environment using Minigrid.

    The agent must pick up a key and open a door to reach the goal.

    Args:
        size: Size of the grid (size x size)
        tile_size: Size of each tile in pixels for rendering
    """

    def __init__(
        self,
        size: int = 6,
        tile_size: int = 32,
        num_players: int = 1,
        max_episode_steps: int | None = None,
        room_types: list[str] = ["DoorKey"],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._size = size
        self._tile_size = tile_size
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        if max_episode_steps is None:
            max_episode_steps = 10 * size * size
        self._max_episode_steps = max_episode_steps

        # self._minigrid_env = gym.make(
        #     f"MiniGrid-DoorKey-{size}x{size}-v0",
        #     render_mode="rgb_array",
        #     agent_pov=False,
        #     max_steps=max_episode_steps,
        #     tile_size=tile_size,
        # )
        self._room_types = room_types
        self._minigrid_envs = {
            room_type: gym.make(
                f"MiniGrid-{room_type}-{size}x{size}-v0",
                render_mode="rgb_array",
                agent_pov=False,
                max_steps=max_episode_steps,
                tile_size=tile_size,
            )
            for room_type in room_types
        }

        self._action_map = {
            "left": 0,
            "right": 1,
            "forward": 2,
            "pickup": 3,
            "drop": 4,
            "toggle": 5,
        }

        self._abbrev_map = {
            "l": "left",
            "r": "right",
            "f": "forward",
            "p": "pickup",
            "d": "drop",
            "t": "toggle",
        }

    @property
    def description(self) -> str:
        return dedent("""
            You are in a room with a locked door. Your goal is to:
            1. Find and pick up the key
            2. Go to the door and toggle it to unlock
            3. Walk through the door
            4. Reach the green goal square

            IMPORTANT: You can only see a 7x7 grid in front of you (first-person view).
            Walls and closed doors block your vision. You cannot see behind you or outside your field of view.
            You need to explore by turning and moving to find the key and the door.

            Available actions (full name or abbreviation):
            - forward (f): Move forward one cell in the direction you're facing
            - left (l): Turn left 90 degrees
            - right (r): Turn right 90 degrees
            - pickup (p): Pick up the object in front of you (use to pick up the key)
            - toggle (t): Interact with object in front (use on door to unlock/open it when holding key)
            - drop (d): Drop the object you're carrying (rarely needed)

            Output format: Plan your full sequence of moves as a comma-separated list.
            You can use full names or single-letter abbreviations, or mix them.
            Unknown tokens are ignored. Examples:
            forward,forward,right,pickup,left,forward,toggle,forward
            f,f,r,p,l,f,t,f
            f,f,right,p,l,forward,t,f
        """).strip()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        super().reset(seed=seed)

        room_type = self.np_random.choice(self._room_types)

        self._minigrid_env = self._minigrid_envs[room_type]
        self._minigrid_env.reset(seed=seed)

        logger.info("Reset Minigrid DoorKey environment.")

        state_text = self._get_observation_text()
        obs = Observation(
            image=self.render(),
            text=None,
            metadata={
                "text_prompt": self.description,
                "state_text": state_text,
            },
        )
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

        actions = self._parse_action_sequence(action_str)

        terminated = False
        truncated = False
        reward = 0.0
        info = {}

        for act in actions:
            _, step_reward, terminated, truncated, info = self._minigrid_env.step(
                self._action_map[act]
            )
            if terminated or truncated:
                reward = float(step_reward)
                break
        else:
            unwrapped = self._minigrid_env.unwrapped
            if unwrapped.carrying is not None and unwrapped.carrying.type == "key":
                reward = 0.5

        # Single-turn: episode always ends after the submitted sequence
        terminated = True

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

    def render(self) -> Image.Image | None:
        img_array = self._minigrid_env.render()
        return Image.fromarray(img_array)

    def _parse_action_sequence(self, action_str: str) -> list[str]:
        """Parse a comma-separated action sequence, resolving abbreviations and ignoring unknowns."""
        actions = []
        for token in action_str.split(","):
            token = token.strip().lower()
            if token in self._action_map:
                actions.append(token)
            elif token in self._abbrev_map:
                actions.append(self._abbrev_map[token])
        return actions

    def _get_observation_text(self) -> str:
        unwrapped_env = self._minigrid_env.unwrapped
        obs = unwrapped_env.gen_obs()
        mission = obs.get("mission", "")
        direction = obs.get("direction", 0)
        direction_names = ["right", "down", "left", "up"]
        direction_str = direction_names[direction] if 0 <= direction < 4 else "unknown"

        carrying = "nothing"
        if unwrapped_env.carrying is not None:
            carrying = f"{unwrapped_env.carrying.color} {unwrapped_env.carrying.type}"

        return (
            f"{mission}\nYou are facing {direction_str}.\nYou are carrying: {carrying}."
        )

    def close(self):
        self._minigrid_env.close()

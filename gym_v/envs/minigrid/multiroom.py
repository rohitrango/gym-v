"""MultiRoom environment using Minigrid."""

from __future__ import annotations

from textwrap import dedent
from typing import Any

import gymnasium as gym
import minigrid
from PIL import Image

from gym_v import Env, Observation, get_logger

logger = get_logger()


class MinigridMultiRoomEnv(Env):
    """MultiRoom environment using Minigrid.

    The agent must navigate through multiple rooms to reach the goal.

    Args:
        min_num_rooms: Minimum number of rooms
        max_num_rooms: Maximum number of rooms
        max_room_size: Maximum size of each room
        tile_size: Size of each tile in pixels for rendering
    """

    def __init__(
        self,
        min_num_rooms: int = 6,
        max_num_rooms: int = 6,
        max_room_size: int = 10,
        tile_size: int = 32,
        num_players: int = 1,
        max_episode_steps: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._min_num_rooms = min_num_rooms
        self._max_num_rooms = max_num_rooms
        self._max_room_size = max_room_size
        self._tile_size = tile_size
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        if max_episode_steps is None:
            max_episode_steps = 100
        self._max_episode_steps = max_episode_steps

        self._minigrid_env = gym.make(
            "MiniGrid-MultiRoom-N6-v0",
            render_mode="rgb_array",
            max_steps=max_episode_steps,
            tile_size=tile_size,
        )

        self._action_map = {
            "left": 0,
            "right": 1,
            "forward": 2,
            "pickup": 3,
            "drop": 4,
            "toggle": 5,
            "done": 6,
        }

    @property
    def description(self) -> str:
        return dedent("""
            You are in a multi-room environment. Your goal is to navigate through multiple rooms to reach the goal.
            Some rooms may be connected by doors that you need to open or unlock.

            Available actions:
            - left: Turn left
            - right: Turn right
            - forward: Move forward
            - pickup: Pick up an object (like a key)
            - drop: Drop the object you're carrying
            - toggle: Toggle/activate an object (like opening a door)
            - done: End the episode
        """).strip()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        super().reset(seed=seed)

        self._minigrid_env.reset(seed=seed)

        logger.info("Reset Minigrid MultiRoom environment.")

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

        if action_str not in self._action_map:
            raise ValueError(
                f"Invalid action '{action_str}'. Valid actions: {list(self._action_map.keys())}"
            )

        action_int = self._action_map[action_str]
        _, reward, terminated, truncated, info = self._minigrid_env.step(action_int)

        obs = Observation(image=self.render(), text=self._get_observation_text())

        return (
            {agent_id: obs for agent_id in self._agent_ids},
            {agent_id: float(reward) for agent_id in self._agent_ids},
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

        return f"{mission}\nYou are facing {direction_str}.\nYou are carrying: {carrying}."

    def close(self):
        self._minigrid_env.close()

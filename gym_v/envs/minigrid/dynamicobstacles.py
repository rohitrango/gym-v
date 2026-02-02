"""Dynamic Obstacles environment using Minigrid."""

from __future__ import annotations

from textwrap import dedent
from typing import Any

import gymnasium as gym
from PIL import Image

from gym_v import Env, Observation, get_logger

logger = get_logger()


class MinigridDynamicObstaclesEnv(Env):
    """Dynamic Obstacles environment using Minigrid.

    The agent must navigate to the goal while avoiding moving obstacles.

    Args:
        size: Size of the grid (size x size)
        n_obstacles: Number of moving obstacles
        tile_size: Size of each tile in pixels for rendering
    """

    def __init__(
        self,
        size: int = 8,
        n_obstacles: int = 4,
        tile_size: int = 32,
        num_players: int = 1,
        max_episode_steps: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._size = size
        self._n_obstacles = n_obstacles
        self._tile_size = tile_size
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        if max_episode_steps is None:
            max_episode_steps = 4 * size * size
        self._max_episode_steps = max_episode_steps

        self._minigrid_env = gym.make(
            "MiniGrid-Dynamic-Obstacles-8x8-v0",
            render_mode="rgb_array",
            agent_pov=True,
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
            You are in a grid with moving obstacles. Your goal is to reach the green goal square while avoiding the moving obstacles.
            Be careful! If you collide with an obstacle, you will fail.

            IMPORTANT: You can only see a 7x7 grid in front of you (first-person view).
            You cannot see behind you or outside your field of view.
            Watch out for moving obstacles in your field of view - colliding with them will fail the task.

            Available actions:
            - left: Turn left 90 degrees (stay in place, only change facing direction)
            - right: Turn right 90 degrees (stay in place, only change facing direction)
            - forward: Move forward one cell in the direction you're facing (avoid obstacles!)
            - toggle: Interact with object in front (not needed in this environment)
            - done: Declare task complete (optional)

            Output format: Simply output the action name.
            Examples:
            forward
            right
            left
        """).strip()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        super().reset(seed=seed)

        self._minigrid_env.reset(seed=seed)

        logger.info("Reset Minigrid Dynamic Obstacles environment.")

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

        return f"{mission}\nYou are facing {direction_str}."

    def close(self):
        self._minigrid_env.close()

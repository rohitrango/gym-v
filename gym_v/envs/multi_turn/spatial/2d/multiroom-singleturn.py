"""Multi-room single-turn environment using Minigrid."""

from __future__ import annotations

from textwrap import dedent
from typing import Any

import gymnasium as gym
import minigrid  # noqa: F401 – registers MiniGrid envs
from PIL import Image

from gym_v import Env, Observation, get_logger

logger = get_logger()

CONFIGS = {
    (2, 4): "MiniGrid-MultiRoom-N2-S4-v0",
    (4, 5): "MiniGrid-MultiRoom-N4-S5-v0",
}


class MultiRoomSingleTurnEnv(Env):
    # Meta: source=Minigrid, category=spatial, turn=single
    """Multi-room single-turn environment using Minigrid.

    The agent must navigate through a series of rooms connected by doors to
    reach the goal. Doors must be toggled open to pass through. The agent
    submits a full planned sequence of moves in one turn.

    Args:
        num_rooms: Number of rooms. Valid values: 2, 4.
        room_size: Size of each room. Valid values: 4 (for N=2), 5 (for N=4).
        tile_size: Size of each tile in pixels for rendering.
    """

    def __init__(
        self,
        num_rooms: int = 2,
        room_size: int = 4,
        tile_size: int = 32,
        num_players: int = 1,
        max_episode_steps: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._num_rooms = num_rooms
        self._room_size = room_size
        self._tile_size = tile_size
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        key = (num_rooms, room_size)
        if key not in CONFIGS:
            raise ValueError(
                f"Unsupported (num_rooms={num_rooms}, room_size={room_size}). "
                f"Valid combinations: {list(CONFIGS.keys())}"
            )
        env_id = CONFIGS[key]

        if max_episode_steps is None:
            max_episode_steps = 20 * num_rooms * room_size * room_size
        self._max_episode_steps = max_episode_steps

        self._minigrid_env = gym.make(
            env_id,
            render_mode="rgb_array",
            agent_pov=False,
            max_steps=max_episode_steps,
            tile_size=tile_size,
        )

        self._action_map = {
            "left": 0,
            "right": 1,
            "forward": 2,
            "toggle": 5,
        }

        self._abbrev_map = {
            "l": "left",
            "r": "right",
            "f": "forward",
            "t": "toggle",
        }

    @property
    def description(self) -> str:
        return dedent(f"""
            You are navigating through {self._num_rooms} connected rooms to reach the green goal square.
            Rooms are connected by doors that must be toggled open before you can pass through.

            Available actions (full name or abbreviation):
            - forward (f): Move forward one cell in the direction you're facing
            - left (l): Turn left 90 degrees
            - right (r): Turn right 90 degrees
            - toggle (t): Open the door in front of you (stand adjacent and face the door)

            Output format: Plan your full sequence of moves as a comma-separated list.
            You can use full names or single-letter abbreviations, or mix them.
            Unknown tokens are ignored. Examples:
            forward,forward,toggle,forward,right,forward,forward
            f,f,t,f,r,f,f
            f,f,toggle,f,r,forward,f
        """).strip()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        super().reset(seed=seed)

        self._minigrid_env.reset(seed=seed)

        logger.info("Reset Minigrid MultiRoom single-turn environment.")

        obs = Observation(
            image=self.render(),
            text=self.description,
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

        return f"{mission}\nYou are facing {direction_str}."

    def close(self):
        self._minigrid_env.close()

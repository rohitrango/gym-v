"""Base wrapper for MiniWorld environments."""

from __future__ import annotations

from abc import abstractmethod
from textwrap import dedent
from typing import Any

import gymnasium as gym
import miniworld  # noqa: F401 - Required to register MiniWorld environments
import numpy as np
from PIL import Image

from gym_v import Env, Observation, get_logger

logger = get_logger()


# Action mapping from string to MiniWorld integer actions
ACTION_MAP = {
    "turn_left": 0,
    "turn_right": 1,
    "move_forward": 2,
    "move_back": 3,
    "pickup": 4,
    "drop": 5,
    "toggle": 6,
    "done": 7,
}

# Reverse mapping
ACTION_MAP_REVERSE = {v: k for k, v in ACTION_MAP.items()}


class MiniWorldBaseEnv(Env):
    """Base wrapper for MiniWorld environments.

    Wraps MiniWorld's Gymnasium environments to provide:
    - String action space (e.g., "move_forward") instead of integers
    - High-resolution 512x512 RGB observations
    - Detailed text descriptions with task goals and current state
    - gym-v compatible interface for VLM training

    Subclasses must implement:
    - get_task_description(): Return task-specific instructions
    - get_available_actions(): Return list of valid action strings for this env
    """

    def __init__(
        self,
        env_id: str,
        obs_width: int = 512,
        obs_height: int = 512,
        num_players: int = 1,
        **miniworld_kwargs,
    ):
        """Initialize MiniWorld wrapper.

        Args:
            env_id: MiniWorld environment ID (e.g., "MiniWorld-Hallway-v0")
            obs_width: Width of rendered observation images
            obs_height: Height of rendered observation images
            num_players: Number of players (always 1 for MiniWorld)
            **miniworld_kwargs: Additional kwargs passed to MiniWorld env
        """
        super().__init__(**miniworld_kwargs)
        self._env_id = env_id
        self._obs_width = obs_width
        self._obs_height = obs_height
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        # Create the wrapped MiniWorld environment
        self._miniworld_env = gym.make(
            env_id,
            render_mode="rgb_array",
            obs_width=obs_width,
            obs_height=obs_height,
            **miniworld_kwargs,
        )

    @property
    def description(self) -> str:
        """Return environment description for VLM."""
        task_desc = self.get_task_description()
        available_actions = self.get_available_actions()

        # Format available actions list
        action_list = "\n".join(
            f"- {action}: {self._get_action_description(action)}"
            for action in available_actions
        )

        return dedent(f"""
            You are navigating a 3D environment from a first-person perspective.

            Task: {task_desc}

            Available Actions:
            {action_list}

            Please choose one action from the available actions above, e.g., {available_actions[0]}
        """).strip()

    @abstractmethod
    def get_task_description(self) -> str:
        """Return task-specific description.

        Example: "Navigate to the red box at the end of the hallway."
        """
        raise NotImplementedError

    @abstractmethod
    def get_available_actions(self) -> list[str]:
        """Return list of valid action strings for this environment.

        Example: ["turn_left", "turn_right", "move_forward"]
        """
        raise NotImplementedError

    def _get_action_description(self, action: str) -> str:
        """Get human-readable description of an action."""
        descriptions = {
            "turn_left": "Turn left by approximately 15 degrees",
            "turn_right": "Turn right by approximately 15 degrees",
            "move_forward": "Move forward in the current direction",
            "move_back": "Move backward in the current direction",
            "pickup": "Pick up an object in front of you",
            "drop": "Drop the object you are carrying",
            "toggle": "Activate or toggle an object in front of you",
            "done": "Signal that you have completed the task",
        }
        return descriptions.get(action, "Unknown action")

    def _get_state_text(self) -> str:
        """Get current state as text description."""
        agent = self._miniworld_env.unwrapped.agent
        pos = agent.pos
        dir_angle = np.degrees(agent.dir)

        # Get carrying status
        carrying = None
        if hasattr(agent, "carrying") and agent.carrying is not None:
            carrying = type(agent.carrying).__name__

        state_parts = [
            f"Position: (x={pos[0]:.1f}, y={pos[1]:.1f}, z={pos[2]:.1f})",
            f"Direction: {dir_angle:.0f}°",
        ]

        if carrying:
            state_parts.append(f"Carrying: {carrying}")
        else:
            state_parts.append("Carrying: None")

        return "\n".join(state_parts)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed, options=options)

        obs_data, info = self._miniworld_env.reset(seed=seed, options=options)

        logger.info(f"Reset {self._env_id}")

        # Some environments (e.g., Sign) return a dict observation
        # Extract the image array from dict or use directly
        if isinstance(obs_data, dict):
            obs_array = obs_data.get("obs", obs_data.get("image", obs_data))
        else:
            obs_array = obs_data

        # Convert numpy array to PIL Image
        obs_image = Image.fromarray(obs_array)

        # Create text observation with state
        obs_text = f"Current State:\n{self._get_state_text()}"

        obs = Observation(image=obs_image, text=obs_text)

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
        """Execute action in the environment."""
        agent_id = next(iter(self._agent_ids))
        action_str = action[agent_id].strip().lower()

        # Convert string action to integer
        if action_str not in ACTION_MAP:
            logger.warning(f"Invalid action: {action_str}")
            # Return current state with zero reward
            obs_data = self._miniworld_env.unwrapped.render_obs()
            # Handle dict observations (e.g., Sign environment)
            if isinstance(obs_data, dict):
                obs_array = obs_data.get("obs", obs_data.get("image", obs_data))
            else:
                obs_array = obs_data
            obs_image = Image.fromarray(obs_array)
            obs_text = f"Current State:\n{self._get_state_text()}"
            obs = Observation(image=obs_image, text=obs_text)

            info = {"invalid_action": True}
            return (
                {agent_id: obs for agent_id in self._agent_ids},
                {agent_id: 0.0 for agent_id in self._agent_ids},
                {
                    **{agent_id: False for agent_id in self._agent_ids},
                    "__all__": False,
                },
                {
                    **{agent_id: False for agent_id in self._agent_ids},
                    "__all__": False,
                },
                {agent_id: info for agent_id in self._agent_ids},
            )

        action_int = ACTION_MAP[action_str]

        # Execute action in MiniWorld
        obs_data, reward, terminated, truncated, info = self._miniworld_env.step(
            action_int
        )

        # Handle dict observations (e.g., Sign environment)
        if isinstance(obs_data, dict):
            obs_array = obs_data.get("obs", obs_data.get("image", obs_data))
        else:
            obs_array = obs_data

        # Convert observation
        obs_image = Image.fromarray(obs_array)
        obs_text = f"Current State:\n{self._get_state_text()}"
        obs = Observation(image=obs_image, text=obs_text)

        info["invalid_action"] = False

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

    def render(self) -> Image.Image | list[Image.Image] | None:
        """Render current state as image."""
        obs_array = self._miniworld_env.unwrapped.render_obs()
        return Image.fromarray(obs_array)

    def close(self):
        """Close the environment."""
        self._miniworld_env.close()

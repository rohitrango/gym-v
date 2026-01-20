"""Stable-Retro environment wrapper for gym-v."""

from __future__ import annotations

from textwrap import dedent
from typing import Any

import numpy as np
from PIL import Image
import stable_retro

from gym_v import Env, Observation, get_logger

logger = get_logger()


class RetroGymVEnv(Env):
    """Wrapper for stable-retro environments.

    This adapter wraps stable-retro's RetroEnv to provide a gym_v.Env interface.
    It converts string-based actions (e.g., "UP", "A", "B+UP") into button masks
    and converts numpy image observations to PIL Images.

    Args:
        game: Name of the game (e.g., "Airstriker-Genesis")
        state: Game state to load (default: stable_retro.State.DEFAULT)
        scenario: Scenario file to use (default: None)
        players: Number of players (default: 1)
        num_players: Number of agents (default: 1)
        **kwargs: Additional arguments passed to stable_retro.make

    Example:
        >>> env = RetroGymVEnv(game="Airstriker-Genesis")
        >>> obs, info = env.reset()
        >>> obs, reward, terminated, truncated, info = env.step({"agent_0": "A"})
    """

    def __init__(
        self,
        game: str,
        state: Any | None = None,
        scenario: str | None = None,
        players: int = 1,
        num_players: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._game = game
        self._players = players
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        # Initialize the retro environment
        retro_kwargs = {
            "game": game,
            "players": players,
            "render_mode": "rgb_array",
        }
        if state is not None:
            retro_kwargs["state"] = state
        if scenario is not None:
            retro_kwargs["scenario"] = scenario

        # Use FILTERED actions for natural action mapping
        retro_kwargs["use_restricted_actions"] = stable_retro.Actions.FILTERED

        self._retro_env = stable_retro.make(**retro_kwargs)

        # Get button configuration from the retro environment
        self.buttons = self._retro_env.buttons
        self._num_buttons = self._retro_env.num_buttons
        self.available_actions = [b for b in self.buttons if b and b != "NULL"]

        # Build button name to index mapping
        self._button_to_idx: dict[str, int] = {}
        for idx, button in enumerate(self.buttons):
            if button and button != "NULL":
                self._button_to_idx[button.upper()] = idx

        logger.info(f"Initialized RetroGymVEnv for game: {game}")
        logger.info(f"Available buttons: {self.buttons}")

    @property
    def description(self) -> str:
        return dedent(f"""
            This is a retro game environment: {self._game}.

            Available buttons: {self.available_actions}

            ## Output Format
            You must output ONLY the action string, nothing else. No explanation, no reasoning, just the action.

            ## Valid Actions
            - Single button: A, B, C, UP, DOWN, LEFT, RIGHT, START
            - Combined buttons: Use "+" to press multiple buttons simultaneously

            ## Examples
            - Move right: RIGHT
            - Move up-right: UP+RIGHT
            - Jump: A
            - Jump right: A+RIGHT
            - Attack: B
            - Attack while moving: B+LEFT
            - Special move: A+B+DOWN
            - No action: NOOP

            ## Your Response
            Output only one action per step. Example valid responses:
            RIGHT
            A+UP
            B
            DOWN+LEFT
            NOOP
        """).strip()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        super().reset(seed=seed)

        # Reset the underlying retro environment
        obs_array, retro_info = self._retro_env.reset(seed=seed, options=options)

        # Convert numpy array to PIL Image
        image = Image.fromarray(obs_array)

        # Create observation text
        text = self._get_observation_text(retro_info)

        obs = Observation(image=image, text=text)
        info: dict[str, Any] = {"retro_info": retro_info}

        logger.info(f"Reset {self._game} environment.")

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

        # Convert string action to button mask
        button_mask = self._action_to_mask(action_str)

        # Step the retro environment
        obs_array, reward, terminated, truncated, retro_info = self._retro_env.step(
            button_mask
        )

        # Convert numpy array to PIL Image
        image = Image.fromarray(obs_array)

        # Create observation text
        text = self._get_observation_text(retro_info)

        obs = Observation(image=image, text=text)
        info: dict[str, Any] = {
            "retro_info": retro_info,
            "action_parsed": self._get_action_meaning(action_str),
        }

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
        """Render the current frame as a PIL Image."""
        rendered = self._retro_env.render()
        if rendered is not None and isinstance(rendered, np.ndarray):
            return Image.fromarray(rendered)
        return None

    def close(self):
        """Close the environment and release resources."""
        if hasattr(self, "_retro_env") and self._retro_env is not None:
            self._retro_env.close()

    def _action_to_mask(self, action_str: str) -> np.ndarray:
        """Convert a string action to a button mask.

        Supports:
        - Single buttons: "A", "UP", "LEFT"
        - Combined buttons: "A+UP", "B+LEFT+DOWN"
        - NOOP/NONE: No buttons pressed

        Args:
            action_str: String representation of the action

        Returns:
            numpy array of button states (0 or 1)
        """
        mask = np.zeros(self._num_buttons, dtype=np.uint8)

        # Normalize action string
        action_upper = action_str.upper().strip()

        # Handle NOOP/NONE
        if action_upper in ("NOOP", "NONE", ""):
            return mask

        # Split by "+" for combined actions
        buttons = [b.strip() for b in action_upper.split("+")]

        for button in buttons:
            if button in self._button_to_idx:
                idx = self._button_to_idx[button]
                mask[idx] = 1
            else:
                logger.warning(
                    f"Unknown button '{button}'. Available: {list(self._button_to_idx.keys())}"
                )

        return mask

    def _get_action_meaning(self, action_str: str) -> list[str]:
        """Get a list of button names being pressed."""
        action_upper = action_str.upper().strip()
        if action_upper in ("NOOP", "NONE", ""):
            return []
        return [
            b.strip()
            for b in action_upper.split("+")
            if b.strip() in self._button_to_idx
        ]

    def _get_observation_text(self, info: dict[str, Any]) -> str:
        """Generate text description from game state info.

        Args:
            info: Game state information from stable-retro

        Returns:
            Text description of current game state
        """
        parts = [f"Game: {self._game}"]

        # Include common game variables if available
        if "score" in info:
            parts.append(f"Score: {info['score']}")
        if "lives" in info:
            parts.append(f"Lives: {info['lives']}")
        if "health" in info:
            parts.append(f"Health: {info['health']}")
        if "level" in info:
            parts.append(f"Level: {info['level']}")

        # Add any other numeric variables
        for key, value in info.items():
            if key not in ("score", "lives", "health", "level") and isinstance(
                value, int | float
            ):
                parts.append(f"{key}: {value}")

        return " | ".join(parts)

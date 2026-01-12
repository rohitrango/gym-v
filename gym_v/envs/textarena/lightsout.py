"""LightsOut game using TextArena."""

from __future__ import annotations

from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw
import textarena as ta

from gym_v import Env, Observation, get_logger

logger = get_logger()


class TextArenaLightsOutEnv(Env):
    """Lights Out logic puzzle game using TextArena's LightsOut environment.

    The player toggles lights on a grid to turn all lights off. When a light is
    pressed, it toggles itself and its adjacent neighbors (up, down, left, right).
    The goal is to turn off all lights within the move limit.

    Args:
        size: Size of the square grid (size x size)
        cell_size: Size of each cell in pixels for rendering
    """

    def __init__(
        self,
        size: int = 5,
        cell_size: int = 80,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._size = size
        self._cell_size = cell_size

        self._ta_env = ta.make(
            "LightsOut-v0-raw",
            size=size,
            max_turns=self._max_episode_steps,
        )

    @property
    def description(self) -> str:
        return dedent(f"""
            Welcome to Lights Out! You have a {self._size}x{self._size} grid of lights.
            Your goal is to turn ALL lights OFF
            When you press a light, it toggles itself AND its adjacent neighbors (up/down/left/right).
            Type [row col] to press a light (0-indexed, so valid range is 0-{self._size-1}).
            You have up to {self._max_episode_steps} moves to solve the puzzle.
        """).strip()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)

        self._ta_env.reset(num_players=1, seed=seed)

        logger.info("Reset LightsOut.")

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

    def render(self) -> Image.Image | list[Image.Image] | None:
        grid = self._ta_env.state.game_state["grid"]

        img_width = self._size * self._cell_size
        img_height = self._size * self._cell_size

        # Create base image
        img = Image.new(
            "RGB", (img_width, img_height), (200, 200, 200)
        )  # Light gray background
        draw = ImageDraw.Draw(img)

        # Color scheme for Lights Out
        colors = {
            "light_on": (255, 255, 220),  # Light color for lights that are ON
            "light_off": (80, 80, 80),  # Dark color for lights that are OFF
            "border": (128, 128, 128),  # Gray borders
        }

        for r in range(self._size):
            for c in range(self._size):
                x0 = c * self._cell_size
                y0 = r * self._cell_size
                x1 = x0 + self._cell_size
                y1 = y0 + self._cell_size

                # Draw cell background based on light state
                if grid[r][c]:  # Light is ON - use light color
                    draw.rectangle([x0, y0, x1, y1], fill=colors["light_on"])
                else:  # Light is OFF - use dark color
                    draw.rectangle([x0, y0, x1, y1], fill=colors["light_off"])

                # Draw cell border
                draw.rectangle([x0, y0, x1, y1], outline=colors["border"], width=2)

        return img

    def _get_observation_text(self) -> str:
        _, ta_obs = self._ta_env.get_observation()
        obs_text = []

        for _, msg, type in ta_obs:
            if type == ta.ObservationType.GAME_ADMIN:
                obs_text.append(msg)

        if "reason" in self._ta_env.state.game_info[0]:
            obs_text.append(self._ta_env.state.game_info[0]["reason"])

        obs_text = "\n".join(obs_text) if obs_text else None

        return obs_text

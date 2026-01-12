"""FifteenPuzzle game using TextArena."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont
import textarena as ta

from gym_v import Env, Observation, get_logger

logger = get_logger()


class TextArenaFifteenPuzzleEnv(Env):
    """15-Puzzle sliding puzzle game using TextArena's FifteenPuzzle environment.

    The player slides numbered tiles on a 4x4 grid to arrange them in order.
    The goal is to arrange tiles 1-15 in ascending order with the empty space
    in the bottom-right corner.

    Args:
        tile_size: Size of each tile in pixels for rendering
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    def __init__(
        self,
        tile_size: int = 80,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._tile_size = tile_size

        self._ta_env = ta.make(
            "FifteenPuzzle-v0-raw",
            max_turns=self._max_episode_steps - 1,
        )

    @property
    def description(self) -> str:
        return dedent("""
            You are playing the 15-Puzzle game.
            The objective of the game is to arrange the numbered tiles in ascending order from 1 to 15, with the empty space located in the bottom-right corner.
            To make a move, you can slide a tile into the empty space by using one of the following commands:
            - 'up': Move the tile below the empty space up.
            - 'down': Move the tile above the empty space down.
            - 'left': Move the tile to the right of the empty space left.
            - 'right': Move the tile to the left of the empty space right.
            To submit your move, type the direction (e.g., 'up', 'down', 'left', or 'right') in square brackets, e.g. [up].
        """).strip()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)

        self._ta_env.reset(num_players=1, seed=seed)

        logger.info("Reset FifteenPuzzle.")

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
        board = self._ta_env.board

        board_width = 4 * self._tile_size
        board_height = 4 * self._tile_size

        # Create base image with light gray background
        img = Image.new("RGB", (board_width, board_height), (240, 240, 240))
        draw = ImageDraw.Draw(img)

        font_path = self.assets_dir / "DejaVuSans.ttf"
        if font_path.exists():
            font = ImageFont.truetype(str(font_path), self._tile_size // 3)
        else:
            logger.warning(f"Font file not found: {font_path}, using default font")
            font = ImageFont.load_default()

        for row in range(4):
            for col in range(4):
                x = col * self._tile_size
                y = row * self._tile_size

                cell_value = board[row][col]

                # Determine tile appearance
                if cell_value is None:
                    # Empty space - dark gray
                    tile_color = (150, 150, 150)
                    text_color = (255, 255, 255)
                    text = ""
                else:
                    # Numbered tile - white with number
                    tile_color = (255, 255, 255)
                    text_color = (0, 0, 0)
                    text = str(cell_value)

                # Draw tile
                draw.rectangle(
                    [x, y, x + self._tile_size - 1, y + self._tile_size - 1],
                    fill=tile_color,
                    outline=(100, 100, 100),
                    width=2,
                )

                # Draw number if present
                if text:
                    # Get text dimensions for centering
                    bbox = draw.textbbox((0, 0), text, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]

                    text_x = x + (self._tile_size - text_width) // 2
                    text_y = y + (self._tile_size - text_height) // 2

                    draw.text((text_x, text_y), text, fill=text_color, font=font)

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

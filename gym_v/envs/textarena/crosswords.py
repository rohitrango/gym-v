"""Crosswords game using TextArena."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont
import textarena as ta

from gym_v import Env, Observation, get_logger

logger = get_logger()


class TextArenaCrosswordsEnv(Env):
    """Crosswords puzzle game using TextArena's Crosswords environment.

    The player fills in letters to complete the crossword puzzle.
    Letters must match the hidden words to complete the game.

    Args:
        hardcore: If True, uses harder difficulty with more challenging words
        num_words: Number of words to include in the crossword puzzle
        cell_size: Size of each cell in pixels for rendering
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    def __init__(
        self,
        hardcore: bool = False,
        num_words: int = 5,
        cell_size: int = 48,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._hardcore = hardcore
        self._num_words = num_words
        self._cell_size = cell_size

        self._ta_env = ta.make(
            "Crosswords-v0-raw",
            hardcore=hardcore,
            num_words=num_words,
            max_turns=self._max_episode_steps - 1,
        )

    @property
    def description(self) -> str:
        return dedent("""
            You are playing Crosswords.
            Here is the current state of the Crosswords grid. Each row and column are numbered.

            You can only provide one response per turn. Hence, plan your approach and risk appetite. Only guesses in the format of [row column letter] will be fetched from your response, e.g. [0 0 d], [1 2 G].
        """).strip()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)

        self._ta_env.reset(num_players=1, seed=seed)

        logger.info("Reset Crosswords.")

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
        board = self._ta_env.state.game_state["board"]

        rows, cols = len(board), len(board[0])
        board_width = cols * self._cell_size
        board_height = rows * self._cell_size

        # Create base image with light gray background
        img = Image.new("RGB", (board_width, board_height), (240, 240, 240))
        draw = ImageDraw.Draw(img)

        font_path = self.assets_dir / "DejaVuSans.ttf"
        if font_path.exists():
            font = ImageFont.truetype(str(font_path), self._cell_size // 3)
        else:
            logger.warning(f"Font file not found: {font_path}, using default font")
            font = ImageFont.load_default()

        for row in range(rows):
            for col in range(cols):
                x = col * self._cell_size
                y = row * self._cell_size

                cell_value = board[row][col]

                # Determine tile color based on cell content
                if cell_value == ".":
                    # Blocked cell - dark gray
                    tile_color = (60, 60, 60)
                    text_color = (255, 255, 255)
                elif cell_value == "_":
                    # Empty cell - white
                    tile_color = (255, 255, 255)
                    text_color = (0, 0, 0)
                    cell_value = ""  # Don't draw the underscore
                else:
                    # Letter cell - light blue with letter
                    tile_color = (220, 235, 255)
                    text_color = (0, 0, 0)

                # Draw tile
                draw.rectangle(
                    [x, y, x + self._cell_size - 1, y + self._cell_size - 1],
                    fill=tile_color,
                    outline=(100, 100, 100),
                    width=2,
                )

                # Draw letter if present
                if cell_value and cell_value.isalpha():
                    # Get text dimensions for centering
                    bbox = draw.textbbox((0, 0), cell_value.upper(), font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]

                    text_x = x + (self._cell_size - text_width) // 2
                    text_y = y + (self._cell_size - text_height) // 2

                    draw.text(
                        (text_x, text_y), cell_value.upper(), fill=text_color, font=font
                    )

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

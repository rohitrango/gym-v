"""Wordle game using TextArena."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont
import textarena as ta

from gym_v import Env, Observation, get_logger

logger = get_logger()


class TextArenaWordleEnv(Env):
    """Crosswords puzzle game using TextArena's Crosswords environment."""

    assets_dir = resources.files("gym_v.envs") / "assets"

    def __init__(
        self,
        word_length: int = 5,
        num_guesses: int = 6,
        hardcore: bool = False,
        cell_size: int = 60,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._word_length = word_length
        self._num_guesses = num_guesses
        self._hardcore = hardcore
        self._cell_size = cell_size

        self._ta_env = ta.make(
            "Wordle-v0-raw",
            word_length=word_length,
            num_guesses=num_guesses,
            hardcore=hardcore,
        )

    @property
    def description(self) -> str:
        return dedent(f"""
            You are Playing Wordle.
            A secret {self._word_length}-letter word has been chosen. You have {self._num_guesses} attempts to guess it.
            For each guess, wrap your word in square brackets (e.g., '[apple]').
            Feedback for each letter will be given as follows:
              - G (green): correct letter in the correct position
              - Y (yellow): letter exists in the word but in the wrong position
              - X (wrong): letter is not in the word
            Enter your guess to begin.
        """).strip()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)

        self._ta_env.reset(num_players=1, seed=seed)

        logger.info("Reset Wordle.")

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
        elif self._current_episode_steps >= self._max_episode_steps:
            reward = (
                self._ta_env.state.rewards[0]
                if self._ta_env.state.rewards
                else self._ta_env._get_percentage_completion()
            )
            terminated = True
            truncated = False
        else:
            reward = 0
            terminated = False
            truncated = False

        obs = Observation(image=self.render(), text=self._get_observation_text())

        return obs, reward, terminated, truncated, info

    def render(self) -> Image.Image:
        # Constants for rendering
        cell_size = self._cell_size
        gap = 5
        margin = 30
        board_width = self._word_length * cell_size + (self._word_length - 1) * gap
        board_height = self._num_guesses * cell_size + (self._num_guesses - 1) * gap
        img_width = board_width + 2 * margin
        img_height = board_height + 2 * margin

        # Create image
        img = Image.new("RGB", (img_width, img_height), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Get game state
        game_state = self._ta_env.state.game_state
        guess_history = game_state["guess_history"]

        # Try to load font
        font_path = self.assets_dir / "DejaVuSans.ttf"
        if font_path.exists():
            letter_font = ImageFont.truetype(str(font_path), 24)
        else:
            logger.warning(f"Font file not found: {font_path}, using default font")
            letter_font = ImageFont.load_default()

        # Draw grid
        start_x = margin
        start_y = margin

        for row in range(self._num_guesses):
            for col in range(self._word_length):
                x = start_x + col * (cell_size + gap)
                y = start_y + row * (cell_size + gap)

                # Determine cell content and color
                letter = ""
                bg_color = (211, 214, 218)  # Default light gray
                text_color = (0, 0, 0)  # Black text

                if row < len(guess_history):
                    # Get guess and feedback for this row
                    guess_word, feedback = guess_history[row]

                    if col < len(guess_word):
                        letter = guess_word[col].upper()

                        # Get feedback color
                        if col < len(feedback):
                            bg_color = self._get_color_from_feedback(feedback[col])
                            text_color = (
                                255,
                                255,
                                255,
                            )  # White text on colored background

                # Draw cell background
                cell_rect = [x, y, x + cell_size, y + cell_size]
                draw.rectangle(
                    cell_rect, fill=bg_color, outline=(187, 187, 187), width=2
                )

                # Draw letter
                if letter:
                    letter_bbox = draw.textbbox((0, 0), letter, font=letter_font)
                    letter_width = letter_bbox[2] - letter_bbox[0]
                    letter_height = letter_bbox[3] - letter_bbox[1]

                    letter_x = x + (cell_size - letter_width) // 2
                    letter_y = (
                        y + (cell_size - letter_height) // 2 - letter_bbox[1] // 2
                    )

                    draw.text(
                        (letter_x, letter_y), letter, fill=text_color, font=letter_font
                    )

        return img

    def _get_color_from_feedback(self, feedback_char: str) -> tuple[int, int, int]:
        if feedback_char == "G":  # Green - correct position
            return (106, 170, 100)
        elif feedback_char == "Y":  # Yellow - wrong position
            return (201, 180, 88)
        elif feedback_char == "X":  # Gray - not in word
            return (120, 124, 126)
        else:  # Empty or unknown
            return (211, 214, 218)  # Light gray

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

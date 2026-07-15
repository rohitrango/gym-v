"""WordSearch game using TextArena."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont
import textarena as ta

from gym_v import Env, Observation, get_logger

logger = get_logger()


class WordSearchEnv(Env):
    # Meta: source=TextArena, category=games, turn=multi
    # Overrides: interaction_mode=multi_turn
    """Crosswords puzzle game using TextArena's Crosswords environment."""

    assets_dir = resources.files("gym_v.envs") / "assets"

    def __init__(
        self,
        hardcore: bool = False,
        cell_size: int = 60,
        num_players: int = 1,
        num_words: int = 5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._hardcore = hardcore
        self._num_words = num_words
        self._cell_size = cell_size
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._ta_env = ta.make(
            "WordSearch-v0-raw",
            hardcore=hardcore,
            max_turns=self._max_episode_steps - 1,
        )

    @property
    def description(self) -> str:
        return dedent("""
            You are participating in a Word Search challenge
            modeled as {'Hardcore' if self.hardcore else 'Basic'}. The objective is to find and highlight hidden words
            on the grid below. The rows and columns are numbered for your reference.

            Here is the current state of the Word Search board:
            ----------------------------------------
            Words you have already found are marked in square brackets [ ]. Each row and column is numbered for clarity.

            To locate a word, specify the row and column of its start and end letters. Note that words are either across or down.
            You may type your response and thoughts in any manner. But you may only submit one submission at a time. For your submissions, use the format '[start_row start_col end_row end_col]'.
            For instance, if you want to find the word 'HELLO' starting at row 1, column 1 and ending at row 1, column 5, enter '[1 1 1 5]'.

            Guidelines:
            - Each guess must be unique; you cannot repeat the same guess.
            - You have a total of 20 incorrect attempts remaining.
            - The history of your attempts will be recorded below.

            Make your guesses carefully and strategically. Good luck, Player {player_id}! Let's see how many words you can find!
        """).strip()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        super().reset(seed=seed)

        self._ta_env.reset(num_players=self.num_players, seed=seed)

        logger.info("Reset WordSearch.")

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
        info = {}
        done, _ = self._ta_env.step(action_str)

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

    def render(self) -> Image.Image | list[Image.Image] | None:
        # Get game state
        game_board = self._ta_env.game_board
        highlighted_positions = self._ta_env.highlighted_positions
        placed_words = self._ta_env.placed_words
        correct_words = self._ta_env.correct_words

        # Constants for rendering
        cell_size = self._cell_size
        margin = 20
        grid_height = len(game_board)
        grid_width = len(game_board[0]) if game_board else 0

        board_width = grid_width * cell_size
        board_height = grid_height * cell_size

        # Calculate required width for word list based on font and words
        font_path = self.assets_dir / "DejaVuSans.ttf"
        if font_path.exists():
            temp_word_font = ImageFont.truetype(str(font_path), 20)
            temp_title_font = ImageFont.truetype(str(font_path), 22)
        else:
            temp_word_font = ImageFont.load_default()
            temp_title_font = ImageFont.load_default()

        # Get all target words to calculate max width
        target_words = list(placed_words.keys())
        title_text = "Words to find:"

        # Calculate maximum text width
        temp_img = Image.new("RGB", (1, 1))
        temp_draw = ImageDraw.Draw(temp_img)

        title_bbox = temp_draw.textbbox((0, 0), title_text, font=temp_title_font)
        title_width = title_bbox[2] - title_bbox[0]

        max_word_width = 0
        for word in target_words:
            word_text = f"✓ {word.upper()}"  # Account for checkmark
            word_bbox = temp_draw.textbbox((0, 0), word_text, font=temp_word_font)
            word_width = word_bbox[2] - word_bbox[0]
            max_word_width = max(max_word_width, word_width)

        word_list_width = max(title_width, max_word_width) + 40  # Add padding

        img_width = board_width + 2 * margin + word_list_width
        img_height = board_height + 2 * margin

        # Create image
        img = Image.new("RGB", (img_width, img_height), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Try to load font
        font_path = self.assets_dir / "DejaVuSans.ttf"
        if font_path.exists():
            cell_font = ImageFont.truetype(str(font_path), 20)
            word_font = ImageFont.truetype(str(font_path), 20)
            title_font = ImageFont.truetype(str(font_path), 22)
        else:
            logger.warning(f"Font file not found: {font_path}, using default font")
            cell_font = ImageFont.load_default()
            word_font = ImageFont.load_default()
            title_font = ImageFont.load_default()

        # Draw grid
        start_x = margin
        start_y = margin

        for row in range(grid_height):
            for col in range(grid_width):
                x = start_x + col * cell_size
                y = start_y + row * cell_size

                # Determine if this cell is part of a found word
                is_highlighted = (row, col) in highlighted_positions

                # Cell colors
                if is_highlighted:
                    bg_color = (144, 238, 144)  # Light green for found letters
                    text_color = (0, 100, 0)  # Dark green text
                else:
                    bg_color = (250, 250, 250)  # Very light gray
                    text_color = (0, 0, 0)  # Black text

                # Draw cell background
                cell_rect = [x, y, x + cell_size, y + cell_size]
                draw.rectangle(
                    cell_rect, fill=bg_color, outline=(200, 200, 200), width=1
                )

                # Draw letter
                letter = game_board[row][col]
                letter_bbox = draw.textbbox((0, 0), letter, font=cell_font)
                letter_width = letter_bbox[2] - letter_bbox[0]
                letter_height = letter_bbox[3] - letter_bbox[1]

                letter_x = x + (cell_size - letter_width) // 2
                letter_y = y + (cell_size - letter_height) // 2 - letter_bbox[1] // 2

                draw.text((letter_x, letter_y), letter, fill=text_color, font=cell_font)

        # Draw word list on the right side
        word_list_x = board_width + 2 * margin + 10
        word_list_y = start_y

        draw.text(
            (word_list_x, word_list_y),
            "Words to find:",
            fill=(0, 0, 0),
            font=title_font,
        )

        # Get all target words from placed_words keys
        target_words = list(placed_words.keys())

        for i, word in enumerate(target_words):
            word_y = word_list_y + 35 + i * 30
            word_text = word.upper()

            # Check if word is found
            is_word_found = word in correct_words

            if is_word_found:
                color = (0, 120, 0)  # Dark green for found words
                word_text = f"✓ {word_text}"
            else:
                color = (50, 50, 50)  # Dark gray for unfound words

            draw.text((word_list_x, word_y), word_text, fill=color, font=word_font)

        return img

    def _get_observation_text(self) -> str:
        _, ta_obs = self._ta_env.get_observation()
        obs_text = []

        for _, msg, type in ta_obs:
            if type in [ta.ObservationType.GAME_ADMIN, ta.ObservationType.GAME_MESSAGE]:
                obs_text.append(msg)

        if "reason" in self._ta_env.state.game_info[0]:
            obs_text.append(self._ta_env.state.game_info[0]["reason"])

        obs_text = "\n".join(obs_text) if obs_text else None

        return obs_text

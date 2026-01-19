"""Crusade game using TextArena."""

from __future__ import annotations

from collections import defaultdict
import re
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw

from gym_v import Env, Observation, get_logger
import textarena as ta

logger = get_logger()


class TextArenaCrusade(Env):
    """
    Crusade game using TextArena's Crusade environment with visual rendering.

    All pieces move like chess knights. Capture opponent pieces to score points.
    """

    is_deterministic = True

    def __init__(
        self,
        tile_size: int = 60,
        num_players: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._tile_size = tile_size
        self._board_size = 8  # Standard 8x8 board
        self.num_players = num_players

        # TextArena Crusade environment
        self._ta_env = ta.make("Crusade-v0-raw")

        self._agent_ids = {"white", "black"}
        self.possible_players = ["white", "black"]

    def _get_current_player(self) -> str:
        """Get the current acting player."""
        current_ta_player_id = self._ta_env.state.current_player_id
        return self.possible_players[current_ta_player_id]

    @property
    def description(self) -> dict[str, str]:
        base_description = dedent("""
            You are Player {player_id} playing as {color} in a game of Crusade.
            Your goal is to capture opponent pieces and score points.

            Rules:
            - All pieces move like chess knights (L-shaped: 2 squares in one direction, 1 square perpendicular)
            - Each capture scores points
            - Game ends after 40 moves or when a player has no pieces/cannot move
            - Higher score wins

            Board coordinates are 0-7 for both rows and columns (0-indexed).
            {color_info}

            NOTE:
            1. Action format: "[from_square to_square]" using algebraic notation, e.g. "[a8 c7]".
               Columns are a-h, rows are 8-1 (row 8 at top, row 1 at bottom).
            2. Only legal moves are permitted.
        """).strip()

        return {
            "white": base_description.format(
                player_id="0",
                color="White",
                color_info="White pieces start in rows 6-7.",
            ),
            "black": base_description.format(
                player_id="1",
                color="Black",
                color_info="Black pieces start in rows 0-1.",
            ),
        }

    def _get_ta_observation(self, player_id: str) -> str:
        ta_player_id = self.possible_players.index(player_id)
        ta_obs = self._ta_env.state.observations[ta_player_id]
        self._ta_env.state.observations[ta_player_id] = []

        obs_text = []

        for _, msg, type in ta_obs:
            if type in [ta.ObservationType.GAME_ADMIN]:
                obs_text.append(msg)

        if "reason" in self._ta_env.state.game_info[ta_player_id]:
            obs_text.append(self._ta_env.state.game_info[ta_player_id]["reason"])

        obs_text = "\n".join(obs_text) if obs_text else None

        if obs_text:
            obs_text = re.sub(r"\bplayer\s0\b", "White", obs_text, flags=re.IGNORECASE)
            obs_text = re.sub(r"\bplayer\s1\b", "Black", obs_text, flags=re.IGNORECASE)

        return obs_text

    def _get_observation(self, all_players: bool = False) -> dict[str, Observation]:
        """Get current board observation for the acting player(s)"""
        image = self.render()

        obs_dict = {}

        players_to_observe = (
            self.possible_players if all_players else [self._get_current_player()]
        )

        for player_id in players_to_observe:
            text_obs = self._get_ta_observation(player_id)
            obs_dict[player_id] = Observation(image=image, text=text_obs)

        return obs_dict

    def inner_step(
        self, action: dict[str, str]
    ) -> tuple[
        dict[str, Observation],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, Any],
    ]:
        info = defaultdict(dict)
        reward = defaultdict(float)

        acting_player = self._get_current_player()

        if acting_player not in action:
            raise ValueError(f"Action required for {acting_player}")

        action_str = action[acting_player]
        ta_action = action_str

        done, _ = self._ta_env.step(ta_action)

        info[acting_player]["invalid_action"] = (
            self._ta_env.state.error_count > 0
            or self._ta_env.state.game_info[self.possible_players.index(acting_player)][
                "invalid_move"
            ]
        )

        terminated = {player_id: False for player_id in self.possible_players}
        truncated = {player_id: False for player_id in self.possible_players}
        terminated["__all__"] = False
        truncated["__all__"] = False

        if done:
            ta_rewards = self._ta_env.state.rewards
            reward["white"] = ta_rewards[0]
            reward["black"] = ta_rewards[1]

            if ta_rewards[0] > ta_rewards[1]:
                reason = "White wins!"
            elif ta_rewards[1] > ta_rewards[0]:
                reason = "Black wins!"
            else:
                reason = "Draw!"

            info["white"]["reward_reason"] = reason
            info["black"]["reward_reason"] = reason
            info["__all__"]["game_state"] = f"Game completed! {reason}"

            for player_id in self.possible_players:
                terminated[player_id] = True
            terminated["__all__"] = True

        else:
            info["__all__"]["game_state"] = "Running..."

        obs = self._get_observation(all_players=done)

        return obs, reward, terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        super().reset(seed=seed)

        self._ta_env.reset(num_players=2, seed=seed)
        obs = self._get_observation()
        info = defaultdict(dict)

        logger.info(f"Reset Crusade: {self._board_size}x{self._board_size} board")

        return obs, info

    def render(self) -> Image.Image:
        """Render the Crusade board with chess-style checkerboard and knight pieces."""
        board_state = self._ta_env.state.game_state["board"]

        board_size = self._tile_size * self._board_size
        cell_size = self._tile_size

        # Add border
        border_width = 15
        total_size = board_size + 2 * border_width

        # Create image with dark border
        img = Image.new(
            "RGB", (total_size, total_size), (40, 40, 40)
        )  # Dark gray border
        draw = ImageDraw.Draw(img)

        # Draw checkerboard pattern (chess style)
        light_square = (240, 217, 181)  # Light tan
        dark_square = (181, 136, 99)  # Dark brown

        for r in range(self._board_size):
            for c in range(self._board_size):
                x1 = c * cell_size + border_width
                y1 = r * cell_size + border_width
                x2 = x1 + cell_size
                y2 = y1 + cell_size

                # Checkerboard pattern
                color = light_square if (r + c) % 2 == 0 else dark_square
                draw.rectangle([x1, y1, x2, y2], fill=color)

        piece_radius = cell_size // 3

        for r in range(self._board_size):
            for c in range(self._board_size):
                cx = c * cell_size + cell_size // 2 + border_width
                cy = r * cell_size + cell_size // 2 + border_width

                piece = board_state[r][c]

                if piece == "W":  # White piece
                    # Draw outline
                    draw.ellipse(
                        [
                            cx - piece_radius - 2,
                            cy - piece_radius - 2,
                            cx + piece_radius + 2,
                            cy + piece_radius + 2,
                        ],
                        fill=(0, 0, 0),  # Black outline
                    )
                    # Draw white piece (hollow)
                    draw.ellipse(
                        [
                            cx - piece_radius,
                            cy - piece_radius,
                            cx + piece_radius,
                            cy + piece_radius,
                        ],
                        fill=(255, 255, 255),  # White
                    )

                elif piece == "B":  # Black piece
                    # Draw black piece (solid)
                    draw.ellipse(
                        [
                            cx - piece_radius,
                            cy - piece_radius,
                            cx + piece_radius,
                            cy + piece_radius,
                        ],
                        fill=(45, 45, 45),  # Dark gray
                    )

        return img

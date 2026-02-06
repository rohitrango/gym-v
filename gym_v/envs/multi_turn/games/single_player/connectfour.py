"""Connect Four game using TextArena."""

from __future__ import annotations

from collections import defaultdict
import re
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw
import textarena as ta

from gym_v import Env, Observation, get_logger

logger = get_logger()


class ConnectFourEnv(Env):
    # Meta: source=TextArena, category=games, turn=multi
    # Overrides: player_count=multi_player
    """
    Connect Four game using TextArena's ConnectFour environment with visual rendering.

    Two players take turns dropping pieces in columns to connect four.
    """

    is_deterministic = True

    def __init__(
        self,
        num_rows: int = 6,
        num_cols: int = 7,
        tile_size: int = 80,
        num_players: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._num_rows = num_rows
        self._num_cols = num_cols
        self._tile_size = tile_size
        self.num_players = num_players

        # TextArena ConnectFour environment
        self._ta_env = ta.make(
            "ConnectFour-v0-raw", num_rows=num_rows, num_cols=num_cols
        )

        self._agent_ids = {"red", "cyan"}
        self.possible_players = ["red", "cyan"]

    def _get_current_player(self) -> str:
        """Get the current acting player."""
        current_ta_player_id = self._ta_env.state.current_player_id
        return self.possible_players[current_ta_player_id]

    @property
    def description(self) -> dict[str, str]:
        base_description = dedent("""
            You are Player {player_id} playing with {color} pieces in a game of Connect Four.
            Your goal is to connect four of your pieces vertically, horizontally, or diagonally before your opponent does.

            On your turn, drop a piece in any column (0-{max_col}). The piece will fall to the lowest available row in that column.
            The first player to get four pieces in a row wins the game.

            NOTE:
            1. Action format: "[col <column_number>]" e.g. "[col 3]".
            2. Only legal moves are permitted.
        """).strip()

        return {
            "red": base_description.format(
                player_id="0", color="Red", max_col=self._num_cols - 1
            ),
            "cyan": base_description.format(
                player_id="1", color="Cyan", max_col=self._num_cols - 1
            ),
        }

    def _get_ta_observation(self, player_id: str) -> str:
        ta_player_id = self.possible_players.index(player_id)
        ta_obs = self._ta_env.state.observations[ta_player_id]
        self._ta_env.state.observations[ta_player_id] = []

        obs_text = []

        for _, msg, type in ta_obs:
            if type in [
                ta.ObservationType.GAME_ADMIN,
                ta.ObservationType.GAME_ACTION_DESCRIPTION,
            ]:
                obs_text.append(msg)

        if "reason" in self._ta_env.state.game_info[ta_player_id]:
            obs_text.append(self._ta_env.state.game_info[ta_player_id]["reason"])

        obs_text = "\n".join(obs_text) if obs_text else None

        if obs_text:
            obs_text = re.sub(r"\bplayer\s0\b", "Red", obs_text, flags=re.IGNORECASE)
            obs_text = re.sub(r"\bplayer\s1\b", "Cyan", obs_text, flags=re.IGNORECASE)

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
            reward["red"] = ta_rewards[0]
            reward["cyan"] = ta_rewards[1]

            if ta_rewards[0] > ta_rewards[1]:
                reason = "Red wins!"
            elif ta_rewards[1] > ta_rewards[0]:
                reason = "Cyan wins!"
            else:
                reason = "Draw!"

            info["red"]["reward_reason"] = reason
            info["cyan"]["reward_reason"] = reason
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

        logger.info(f"Reset ConnectFour: {self._num_rows}x{self._num_cols} board")

        return obs, info

    def render(self) -> Image.Image:
        """Render the Connect Four board with pieces."""
        board_state = self._ta_env.state.game_state["board"]

        width = self._tile_size * self._num_cols
        height = self._tile_size * self._num_rows
        cell_size = self._tile_size

        # Create image with dark slate background
        img = Image.new("RGB", (width, height), (71, 85, 105))  # Slate-600
        draw = ImageDraw.Draw(img)

        # Draw grid and holes for pieces
        piece_radius = cell_size // 2 - 8

        for r in range(self._num_rows):
            for c in range(self._num_cols):
                cx = c * cell_size + cell_size // 2
                cy = r * cell_size + cell_size // 2

                piece = board_state[r][c]

                if piece == "X":  # Red piece
                    draw.ellipse(
                        [
                            cx - piece_radius,
                            cy - piece_radius,
                            cx + piece_radius,
                            cy + piece_radius,
                        ],
                        fill=(255, 127, 127),  # Red
                        outline=None,
                    )
                elif piece == "O":  # Cyan piece
                    draw.ellipse(
                        [
                            cx - piece_radius,
                            cy - piece_radius,
                            cx + piece_radius,
                            cy + piece_radius,
                        ],
                        fill=(94, 234, 212),  # Cyan
                        outline=None,
                    )
                else:  # Empty slot (Dark hole)
                    draw.ellipse(
                        [
                            cx - piece_radius,
                            cy - piece_radius,
                            cx + piece_radius,
                            cy + piece_radius,
                        ],
                        fill=(30, 41, 59),  # Darker slate for holes
                        outline=None,
                    )

        return img

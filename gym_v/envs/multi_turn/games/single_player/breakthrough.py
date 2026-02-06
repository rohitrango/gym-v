"""Breakthrough game using TextArena."""

from __future__ import annotations

from collections import defaultdict
import re
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw
import textarena as ta

from gym_v import Env, Observation, get_logger

logger = get_logger()


class BreakthroughEnv(Env):
    # Meta: source=TextArena, category=games, turn=multi
    # Overrides: player_count=multi_player
    """
    Breakthrough game using TextArena's Breakthrough environment with visual rendering.

    Two players take turns moving pieces forward to reach the opponent's home row.
    """

    is_deterministic = True

    def __init__(
        self,
        board_size: int = 8,
        tile_size: int = 60,
        num_players: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._board_size = board_size
        self._tile_size = tile_size
        self.num_players = num_players
        assert self._board_size <= 26, "Board size must be no more than 26"

        # TextArena Breakthrough environment
        self._ta_env = ta.make("Breakthrough-v0-raw", board_size=board_size)

        self._agent_ids = {"white", "black"}
        self.possible_players = ["white", "black"]

    def _get_current_player(self) -> str:
        """Get the current acting player."""
        current_ta_player_id = self._ta_env.state.current_player_id
        return self.possible_players[current_ta_player_id]

    @property
    def description(self) -> dict[str, str]:
        base_description = dedent("""
            You are Player {player_id} playing as {color} in a game of Breakthrough.
            Your goal is to move one of your pieces to the opponent's home row (row {goal_row}).

            On your turn, you can move a single piece one step {direction}:
            - Move straight forward to an empty square (cannot capture)
            - Move diagonally forward ONLY to capture an opponent's piece

            **Important restrictions:**
            - You can NEVER move backward or sideways
            - Diagonal moves are ONLY allowed for capturing

            Board coordinates are 0-{max_coord} for both rows and columns (0-indexed).
            {color_info}

            NOTE:
            1. Action format: "[from_coord to_coord]" using algebraic notation, e.g. "[a2a3]".
            2. Ensure that all your moves comply with the rules. Frequent illegal actions will cause you to immediately lose and end the game.
        """).strip()

        return {
            "white": base_description.format(
                player_id="0",
                color="White",
                goal_row=self._board_size - 1,
                direction="up (increasing row)",
                max_coord=self._board_size - 1,
                color_info="White pieces start in rows 0-1.",
            ),
            "black": base_description.format(
                player_id="1",
                color="Black",
                goal_row=0,
                direction="down (decreasing row)",
                max_coord=self._board_size - 1,
                color_info=f"Black pieces start in rows {self._board_size - 2}-{self._board_size - 1}.",
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

        logger.info(f"Reset Breakthrough: {self._board_size}x{self._board_size} board")

        return obs, info

    def render(self) -> Image.Image:
        """Render the Breakthrough board with chess-style checkerboard."""
        board_state = self._ta_env.state.game_state["board"]

        size = self._tile_size * self._board_size
        cell_size = self._tile_size

        # Create image
        img = Image.new("RGB", (size, size), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Draw checkerboard pattern
        light_square = (240, 217, 181)  # Light brown
        dark_square = (181, 136, 99)  # Dark brown

        for r in range(self._board_size):
            for c in range(self._board_size):
                x1 = c * cell_size
                y1 = r * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size

                # Checkerboard pattern
                color = light_square if (r + c) % 2 == 0 else dark_square
                draw.rectangle([x1, y1, x2, y2], fill=color)

        # Draw pieces
        piece_radius = cell_size // 3

        for r in range(self._board_size):
            for c in range(self._board_size):
                cx = c * cell_size + cell_size // 2
                cy = r * cell_size + cell_size // 2

                piece = board_state[r][c]

                if piece == "W":  # White piece (hollow circle)
                    draw.ellipse(
                        [
                            cx - piece_radius,
                            cy - piece_radius,
                            cx + piece_radius,
                            cy + piece_radius,
                        ],
                        fill=(255, 255, 255),  # White fill
                        outline=(0, 0, 0),  # Black outline
                        width=3,
                    )
                elif piece == "B":  # Black piece (solid circle)
                    draw.ellipse(
                        [
                            cx - piece_radius,
                            cy - piece_radius,
                            cx + piece_radius,
                            cy + piece_radius,
                        ],
                        fill=(0, 0, 0),  # Black fill
                        outline=(0, 0, 0),
                        width=1,
                    )

        return img

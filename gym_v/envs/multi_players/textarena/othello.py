"""Othello game using TextArena."""

from __future__ import annotations

from collections import defaultdict
import re
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw

from gym_v import Env, Observation, get_logger
import textarena as ta

logger = get_logger()


class TextArenaOthello(Env):
    """
    Othello (Reversi) game using TextArena's Othello environment with visual rendering.

    Two players take turns placing pieces to flip opponent's pieces and control the board.
    """

    is_deterministic = True

    def __init__(
        self,
        board_size: int = 8,
        show_valid: bool = True,
        tile_size: int = 80,
        num_players: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._board_size = board_size
        self._show_valid = show_valid
        self._tile_size = tile_size
        self.num_players = num_players

        # TextArena Othello environment
        self._ta_env = ta.make(
            "Othello-v0-raw", board_size=board_size, show_valid=show_valid
        )

        self._agent_ids = {"black", "white"}
        self.possible_players = ["black", "white"]

    def _get_current_player(self) -> str:
        """Get the current acting player."""
        current_ta_player_id = self._ta_env.state.current_player_id
        return self.possible_players[current_ta_player_id]

    @property
    def description(self) -> dict[str, str]:
        base_description = dedent("""
            You are Player {player_id} playing {color} ({piece}) in a game of Othello.
            Your goal is to have more pieces of your color on the board by the end of the game.

            On your turn, place a piece such that it flanks one or more of your opponent's pieces—in any direction (horizontal, vertical, or diagonal)—between your new piece and another of your existing pieces.
            All flanked opponent pieces will be flipped to your color.
            {valid_moves_text}

            NOTE:
            1. Action format: "[row, col]" e.g. "[3, 2]".
            2. Only legal moves are permitted. If you continue to choose illegal actions, you will be immediately disqualified and the game will end.
        """).strip()

        valid_moves_text = (
            "Valid moves are highlighted for your convenience."
            if self._show_valid
            else ""
        )

        return {
            "black": base_description.format(
                player_id="0",
                color="Black",
                piece="●",
                valid_moves_text=valid_moves_text,
            ),
            "white": base_description.format(
                player_id="1",
                color="White",
                piece="○",
                valid_moves_text=valid_moves_text,
            ),
        }

    def _get_ta_observation(self, player_id: str) -> str:
        ta_player_id = self.possible_players.index(player_id)
        ta_obs = self._ta_env.state.observations[ta_player_id]
        # Clear observations after reading
        self._ta_env.state.observations[ta_player_id] = []

        obs_text = []

        for _, msg, type in ta_obs:
            if type in [
                ta.ObservationType.GAME_ADMIN,
                ta.ObservationType.GAME_ACTION_DESCRIPTION,
                ta.ObservationType.GAME_MESSAGE,
            ]:
                obs_text.append(msg)

        if "reason" in self._ta_env.state.game_info[ta_player_id]:
            obs_text.append(self._ta_env.state.game_info[ta_player_id]["reason"])

        obs_text = "\n".join(obs_text) if obs_text else None

        if obs_text:
            obs_text = re.sub(r"\bplayer\s0\b", "Black", obs_text, flags=re.IGNORECASE)
            obs_text = re.sub(r"\bplayer\s1\b", "White", obs_text, flags=re.IGNORECASE)

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

        # Get action for the acting player
        acting_player = self._get_current_player()

        # Check if acting player provided an action
        if acting_player not in action:
            raise ValueError(f"Action required for {acting_player}")

        action_str = action[acting_player]

        ta_action = action_str

        done, _ = self._ta_env.step(ta_action)

        # If the game is not done and there are no valid moves, pass this turn
        if not done and not self._ta_env.state.game_state["valid_moves"]:
            done, _ = self._ta_env.step("pass")

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
            reward["black"] = ta_rewards[0]
            reward["white"] = ta_rewards[1]

            # Get final scores for info
            game_state = self._ta_env.state.game_state
            black_count = game_state["black_count"]
            white_count = game_state["white_count"]

            if black_count > white_count:
                reason = f"Black wins {black_count}-{white_count}!"
            elif white_count > black_count:
                reason = f"White wins {white_count}-{black_count}!"
            else:
                reason = f"Draw {black_count}-{white_count}!"

            info["black"]["reward_reason"] = reason
            info["white"]["reward_reason"] = reason
            info["__all__"]["game_state"] = f"Game completed! {reason}"

            for player_id in self.possible_players:
                terminated[player_id] = True
            terminated["__all__"] = True

        else:
            # Game continues
            info["__all__"]["game_state"] = "Running..."

        obs = self._get_observation(all_players=done)

        return obs, reward, terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        super().reset(seed=seed)  # Initialize self._np_random

        # TextArena reset might not take seed in the same way or might need int
        self._ta_env.reset(num_players=2, seed=seed)
        obs = self._get_observation()
        info = defaultdict(dict)

        logger.info(f"Reset Othello: board size {self._board_size}")

        return obs, info

    def render(self) -> Image.Image:
        """Render the Othello board with pieces."""
        board_state = self._ta_env.board
        valid_moves = self._ta_env.state.game_state["valid_moves"]

        size = self._tile_size * self._board_size
        cell_size = self._tile_size

        # Create image with green background (classic board color)
        img = Image.new("RGB", (size, size), (34, 139, 34))  # Forest Green
        draw = ImageDraw.Draw(img)

        # Draw grid lines
        line_color = (0, 0, 0)
        line_width = 2
        for i in range(self._board_size + 1):
            # Vertical lines
            x = i * cell_size
            draw.line([(x, 0), (x, size)], fill=line_color, width=line_width)
            # Horizontal lines
            y = i * cell_size
            draw.line([(0, y), (size, y)], fill=line_color, width=line_width)

        # Draw pieces and valid moves
        piece_radius = cell_size // 2 - 4

        for r in range(self._board_size):
            for c in range(self._board_size):
                cx = c * cell_size + cell_size // 2
                cy = r * cell_size + cell_size // 2

                piece = (
                    board_state[r][c]
                    if r < len(board_state) and c < len(board_state[r])
                    else ""
                )
                if piece == "B":  # Black piece
                    draw.ellipse(
                        [
                            cx - piece_radius,
                            cy - piece_radius,
                            cx + piece_radius,
                            cy + piece_radius,
                        ],
                        fill=(0, 0, 0),
                        outline=(40, 40, 40),
                        width=2,
                    )
                elif piece == "W":  # White piece
                    draw.ellipse(
                        [
                            cx - piece_radius,
                            cy - piece_radius,
                            cx + piece_radius,
                            cy + piece_radius,
                        ],
                        fill=(255, 255, 255),
                        outline=(200, 200, 200),
                        width=2,
                    )
                elif self._show_valid and [r, c] in valid_moves:
                    # Valid move indicator (small dot)
                    dot_radius = 4
                    draw.ellipse(
                        [
                            cx - dot_radius,
                            cy - dot_radius,
                            cx + dot_radius,
                            cy + dot_radius,
                        ],
                        fill=(255, 255, 0),
                        outline=(200, 200, 0),
                        width=1,
                    )

        return img

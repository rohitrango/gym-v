"""Alquerque game using TextArena."""

from __future__ import annotations

from collections import defaultdict
import re
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw
import textarena as ta

from gym_v import Env, Observation, get_logger

logger = get_logger()


class AlquerqueEnv(Env):
    # Meta: source=TextArena, category=games, turn=multi
    # Overrides: player_count=multi_player
    """
    Alquerque game using TextArena's Alquerque environment with visual rendering.

    Two players take turns moving pieces forward or capturing opponent's pieces by jumping.
    """

    is_deterministic = True

    def __init__(
        self,
        tile_size: int = 80,
        num_players: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._tile_size = tile_size
        self._board_size = 5  # Alquerque is always 5x5
        self.num_players = num_players

        # TextArena Alquerque environment
        self._ta_env = ta.make("Alquerque-v0-raw")

        self._agent_ids = {"white", "black"}
        self.possible_players = ["white", "black"]

    def _get_current_player(self) -> str:
        """Get the current acting player."""
        current_ta_player_id = self._ta_env.state.current_player_id
        return self.possible_players[current_ta_player_id]

    @property
    def description(self) -> dict[str, str]:
        base_description = dedent("""
            You are Player {player_id} playing with {color} pieces in Alquerque game.
            Goal: Capture all opponent pieces or block them from moving.

            Board: 5×5 grid with horizontal, vertical, and diagonal connections.
            Each player has 12 pieces. White moves first. Center point starts empty.

            Actions (choose one per turn):

            1. Normal Move: Move one piece ONE STEP FORWARD along a vertical line
            to an adjacent empty point. Forward is toward the opponent's side.
            No backward, sideways, or diagonal moves.

            2. Capture Move:
            - Jump over an adjacent opponent piece to the immediate empty point beyond
                it along any connection line.
            - Captures can be in ANY direction (forward, backward, sideways, diagonal).
            - Multi-capture allowed: Continue jumping from the new position if possible,
                changing direction is allowed.
            - Must use the same piece for the entire capture sequence.

            Key Rules:
            - Capturing is NOT mandatory.
            - Cannot jump over your own pieces.
            - Normal moves cannot land on occupied points.

            Winning: Opponent has no pieces left or no legal moves.
            Draw if neither side can make progress.

            NOTE:
            1. Action format: "[from_coord to_coord]" using notation like "[a1 a2]".
               Columns are a-e, rows are 1-5 (row 1 at bottom, row 5 at top).
            2. All actions must be valid according to the rules.
        """).strip()

        return {
            "white": base_description.format(player_id="0", color="white"),
            "black": base_description.format(player_id="1", color="black"),
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

        logger.info("Reset Alquerque: 5x5 board")

        return obs, info

    def render(self) -> Image.Image:
        """Render the Alquerque board with pieces."""
        board_state = self._ta_env.state.game_state["board"]

        # Alquerque has 5x5 intersections, which means 4x4 cells
        num_cells = self._board_size - 1
        cell_size = self._tile_size
        board_size = cell_size * num_cells

        # Add padding around the board
        padding = cell_size // 2
        size = board_size + 2 * padding

        # Create image with green background
        img = Image.new("RGB", (size, size), (34, 139, 34))  # Forest Green
        draw = ImageDraw.Draw(img)

        # Draw grid lines with light green color
        line_color = (144, 238, 144)  # Light Green
        line_width = 2

        # Horizontal and vertical lines (5 lines for 5 intersections)
        for i in range(self._board_size):
            # Vertical lines
            x = i * cell_size + padding
            draw.line(
                [(x, padding), (x, board_size + padding)],
                fill=line_color,
                width=line_width,
            )
            # Horizontal lines
            y = i * cell_size + padding
            draw.line(
                [(padding, y), (board_size + padding, y)],
                fill=line_color,
                width=line_width,
            )

        # Draw diagonal lines
        # In Alquerque, every cell has diagonal connections
        for i in range(num_cells):
            for j in range(num_cells):
                x1, y1 = j * cell_size + padding, i * cell_size + padding
                x2, y2 = (j + 1) * cell_size + padding, (i + 1) * cell_size + padding

                # Draw both diagonals for every cell
                # Top-left to bottom-right
                draw.line([(x1, y1), (x2, y2)], fill=line_color, width=line_width)
                # Top-right to bottom-left
                draw.line([(x2, y1), (x1, y2)], fill=line_color, width=line_width)

        # Draw pieces on intersections (not in cell centers)
        piece_radius = cell_size // 3

        for r in range(self._board_size):
            for c in range(self._board_size):
                # Position at grid intersection with padding offset
                cx = c * cell_size + padding
                cy = r * cell_size + padding

                piece = board_state[r][c]

                if piece == "R":  # White piece (player 0)
                    draw.ellipse(
                        [
                            cx - piece_radius,
                            cy - piece_radius,
                            cx + piece_radius,
                            cy + piece_radius,
                        ],
                        fill=(255, 255, 255),  # White
                        outline=(180, 180, 180),
                        width=3,
                    )
                elif piece == "B":  # Black piece (player 1)
                    draw.ellipse(
                        [
                            cx - piece_radius,
                            cy - piece_radius,
                            cx + piece_radius,
                            cy + piece_radius,
                        ],
                        fill=(45, 45, 45),  # Dark gray/black
                        outline=(0, 0, 0),
                        width=3,
                    )

        return img

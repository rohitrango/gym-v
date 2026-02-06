"""SimpleTak game using TextArena."""

from __future__ import annotations

from collections import defaultdict
import re
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw
import textarena as ta

from gym_v import Env, Observation, get_logger

logger = get_logger()


class SimpleTakEnv(Env):
    # Meta: source=TextArena, category=games, turn=multi
    # Overrides: player_count=multi_player
    """
    SimpleTak game using TextArena's SimpleTak environment with visual rendering.

    Players take turns placing stones to form a continuous path connecting two opposite edges.
    """

    is_deterministic = True

    def __init__(
        self,
        board_size: int = 5,
        cell_size: int = 80,
        num_players: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._board_size = board_size
        self._cell_size = cell_size
        self.num_players = num_players

        # TextArena SimpleTak environment
        self._ta_env = ta.make("SimpleTak-v0-raw", board_size=self._board_size)

        self._agent_ids = {"white", "black"}
        self.possible_players = ["white", "black"]

    def _get_current_player(self) -> str:
        """Get the current acting player."""
        current_ta_player_id = self._ta_env.state.current_player_id
        return self.possible_players[current_ta_player_id]

    @property
    def description(self) -> dict[str, str]:
        base_description = dedent("""
            You are playing SimpleTak as {color}.

            Rules:
            - Players take turns placing stones on empty cells
            - Win by forming a continuous path connecting two opposite edges:
                * Top-to-bottom OR
                * Left-to-right
            - Diagonal connections don't count

            NOTE:
            1. Action format: "[cell_index]" where cell_index = row * board_size + col.
               For a 5x5 board, indices go from 0 to 24.
            2. Only legal moves are permitted.
        """).strip()

        return {
            "white": base_description.format(color="white"),
            "black": base_description.format(color="black"),
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

        logger.info(f"Reset SimpleTak: board_size {self._board_size}")

        return obs, info

    def render(self) -> Image.Image:
        """Render the SimpleTak board."""
        board = self._ta_env.state.game_state["board"]
        cell_size = self._cell_size
        margin = cell_size // 2
        board_pixel_size = self._board_size * cell_size

        width = board_pixel_size + 2 * margin
        height = board_pixel_size + 2 * margin

        # Create image with wooden background
        img = Image.new("RGB", (width, height), color=(210, 180, 140))
        draw = ImageDraw.Draw(img)

        # Draw wood grain texture
        for i in range(0, width, 20):
            shade = 210 + (i % 40) - 20
            draw.line(
                [(i, 0), (i, height)], fill=(shade, shade - 30, shade - 70), width=2
            )

        # Draw board background (lighter)
        board_rect = [
            margin,
            margin,
            margin + board_pixel_size,
            margin + board_pixel_size,
        ]
        draw.rectangle(board_rect, fill=(230, 200, 160))

        # Draw grid lines
        line_color = (80, 60, 40)
        line_width = max(2, cell_size // 40)

        # Highlight edges (thicker lines for winning condition)
        edge_width = max(3, cell_size // 20)

        # Top and bottom edges (thicker - for vertical connection)
        draw.rectangle(
            [margin, margin, margin + board_pixel_size, margin + edge_width],
            fill=(100, 80, 60),
            outline=None,
        )
        draw.rectangle(
            [
                margin,
                margin + board_pixel_size - edge_width,
                margin + board_pixel_size,
                margin + board_pixel_size,
            ],
            fill=(100, 80, 60),
            outline=None,
        )

        # Left and right edges (thicker - for horizontal connection)
        draw.rectangle(
            [margin, margin, margin + edge_width, margin + board_pixel_size],
            fill=(100, 80, 60),
            outline=None,
        )
        draw.rectangle(
            [
                margin + board_pixel_size - edge_width,
                margin,
                margin + board_pixel_size,
                margin + board_pixel_size,
            ],
            fill=(100, 80, 60),
            outline=None,
        )

        # Draw grid
        for i in range(self._board_size + 1):
            # Vertical lines
            x = margin + i * cell_size
            draw.line(
                [(x, margin), (x, margin + board_pixel_size)],
                fill=line_color,
                width=line_width,
            )
            # Horizontal lines
            y = margin + i * cell_size
            draw.line(
                [(margin, y), (margin + board_pixel_size, y)],
                fill=line_color,
                width=line_width,
            )

        # Draw stones
        stone_radius = int(cell_size * 0.35)

        for row in range(self._board_size):
            for col in range(self._board_size):
                cell_content = board[row][col]
                if cell_content != "":
                    center_x = margin + col * cell_size + cell_size // 2
                    center_y = margin + row * cell_size + cell_size // 2

                    # Draw stone shadow
                    shadow_offset = max(2, cell_size // 40)
                    draw.ellipse(
                        [
                            center_x - stone_radius + shadow_offset,
                            center_y - stone_radius + shadow_offset,
                            center_x + stone_radius + shadow_offset,
                            center_y + stone_radius + shadow_offset,
                        ],
                        fill=(100, 100, 100),
                    )

                    # Draw stone
                    if cell_content == "O":
                        # White stone with gradient
                        for i in range(stone_radius, 0, -1):
                            shade = 255 - (stone_radius - i) * 2
                            draw.ellipse(
                                [
                                    center_x - i,
                                    center_y - i,
                                    center_x + i,
                                    center_y + i,
                                ],
                                fill=(shade, shade, shade),
                            )
                        # Outline
                        draw.ellipse(
                            [
                                center_x - stone_radius,
                                center_y - stone_radius,
                                center_x + stone_radius,
                                center_y + stone_radius,
                            ],
                            outline=(200, 200, 200),
                            width=max(2, cell_size // 40),
                        )
                    else:  # "X"
                        # Black stone with gradient
                        for i in range(stone_radius, 0, -1):
                            shade = (stone_radius - i) * 2
                            draw.ellipse(
                                [
                                    center_x - i,
                                    center_y - i,
                                    center_x + i,
                                    center_y + i,
                                ],
                                fill=(shade, shade, shade),
                            )
                        # Outline
                        draw.ellipse(
                            [
                                center_x - stone_radius,
                                center_y - stone_radius,
                                center_x + stone_radius,
                                center_y + stone_radius,
                            ],
                            outline=(60, 60, 60),
                            width=max(2, cell_size // 40),
                        )

        return img

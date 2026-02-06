"""Wild Tic Tac Toe game using TextArena."""

from __future__ import annotations

from collections import defaultdict
from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont
import textarena as ta

from gym_v import Env, Observation, get_logger

logger = get_logger()


class WildTicTacToeEnv(Env):
    # Meta: source=TextArena, category=games, turn=multi
    # Overrides: player_count=multi_player
    """
    Wild Tic Tac Toe game using TextArena's WildTicTacToe environment with visual rendering.

    Two players take turns placing X or O marks on the board. Unlike classic Tic Tac Toe,
    players can place either mark on their turn. The first to get three of the same mark
    in a row wins.
    """

    assets_dir = resources.files("gym_v.envs") / "assets"
    is_deterministic = True

    def __init__(
        self,
        tile_size: int = 120,
        num_players: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._tile_size = tile_size
        self.num_players = num_players

        # TextArena WildTicTacToe environment
        self._ta_env = ta.make("WildTicTacToe-v0-raw")

        self._agent_ids = {"player_0", "player_1"}
        self.possible_players = ["player_0", "player_1"]

    def _get_current_player(self) -> str:
        """Get the current acting player."""
        current_ta_player_id = self._ta_env.state.current_player_id
        return self.possible_players[current_ta_player_id]

    @property
    def description(self) -> dict[str, str]:
        base_description = dedent("""
            You are Player {player_id} playing Wild Tic Tac Toe.
            Your goal is to be the first to get three of the same mark (X or O) in a row - horizontally, vertically, or diagonally.

            The twist: On your turn, you can place EITHER an 'X' OR an 'O' in any empty cell!
            This means you can win with either symbol, and you need to be strategic about which mark you place where.

            The board cells are numbered 0-8.

            NOTE:
            1. Action format: "[mark cell]" e.g. "[X 4]" to place X in cell 4.
            2. Every move must be legal. A high number of illegal attempts will cause the game to stop immediately and you to lose.
        """).strip()

        return {
            "player_0": base_description.format(player_id="0"),
            "player_1": base_description.format(player_id="1"),
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
            reward["player_0"] = ta_rewards[0]
            reward["player_1"] = ta_rewards[1]

            if ta_rewards[0] > ta_rewards[1]:
                reason = "Player 0 wins!"
            elif ta_rewards[1] > ta_rewards[0]:
                reason = "Player 1 wins!"
            else:
                reason = "Draw!"

            info["player_0"]["reward_reason"] = reason
            info["player_1"]["reward_reason"] = reason
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

        logger.info("Reset Wild Tic Tac Toe")

        return obs, info

    def _get_font(self, size: int) -> ImageFont.ImageFont:
        font_path = self.assets_dir / "DejaVuSans.ttf"
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size)
        else:
            logger.warning(f"Font file not found: {font_path}, using default font")
            return ImageFont.load_default()

    def render(self) -> Image.Image:
        """Render the Wild Tic Tac Toe board."""
        board = self._ta_env.state.game_state["board"]

        size = self._tile_size * 3
        cell_size = self._tile_size

        # Create image with light background
        img = Image.new("RGB", (size, size), (245, 245, 245))
        draw = ImageDraw.Draw(img)

        # Draw grid lines
        line_color = (60, 60, 60)
        line_width = 4
        for i in range(1, 3):
            # Vertical lines
            x = i * cell_size
            draw.line([(x, 0), (x, size)], fill=line_color, width=line_width)
            # Horizontal lines
            y = i * cell_size
            draw.line([(0, y), (size, y)], fill=line_color, width=line_width)

        # Try to load font
        font = self._get_font(cell_size // 2)

        # Draw marks and cell numbers
        for r in range(3):
            for c in range(3):
                cx = c * cell_size + cell_size // 2
                cy = r * cell_size + cell_size // 2

                cell_num = r * 3 + c
                mark = board[r][c] if r < len(board) and c < len(board[r]) else ""

                if mark == "X":
                    # Draw X
                    offset = cell_size // 3
                    draw.line(
                        [(cx - offset, cy - offset), (cx + offset, cy + offset)],
                        fill=(220, 20, 60),  # Crimson
                        width=8,
                    )
                    draw.line(
                        [(cx - offset, cy + offset), (cx + offset, cy - offset)],
                        fill=(220, 20, 60),
                        width=8,
                    )
                elif mark == "O":
                    # Draw O
                    radius = cell_size // 3
                    draw.ellipse(
                        [cx - radius, cy - radius, cx + radius, cy + radius],
                        outline=(30, 144, 255),  # Dodger Blue
                        width=8,
                    )
                else:
                    # Draw cell number for empty cells
                    text = str(cell_num)
                    draw.text(
                        (cx, cy), text, fill=(180, 180, 180), font=font, anchor="mm"
                    )

        return img

"""Ultimate Tic Tac Toe game using TextArena."""

from __future__ import annotations

from collections import defaultdict
from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation, get_logger
import textarena as ta

logger = get_logger()


class TextArenaUltimateTicTacToe(Env):
    """
    Ultimate Tic Tac Toe game using TextArena's UltimateTicTacToe environment with visual rendering.

    Two players compete on a 3x3 grid of mini Tic Tac Toe boards. Win three mini-boards in a row
    to win the game. Each move determines which mini-board your opponent must play in next.
    """

    assets_dir = resources.files("gym_v.envs") / "assets"
    is_deterministic = True

    def __init__(
        self,
        mini_board_size: int = 200,
        num_players: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._mini_board_size = mini_board_size
        self._cell_size = mini_board_size // 3
        self.num_players = num_players

        # TextArena UltimateTicTacToe environment
        self._ta_env = ta.make("UltimateTicTacToe-v0-raw")

        self._agent_ids = {"player_0", "player_1"}
        self.possible_players = ["player_0", "player_1"]

    def _get_current_player(self) -> str:
        """Get the current acting player."""
        current_ta_player_id = self._ta_env.state.current_player_id
        return self.possible_players[current_ta_player_id]

    @property
    def description(self) -> dict[str, str]:
        base_description = dedent("""
            You are Player {player_id} ({mark}) playing Ultimate Tic Tac Toe.
            Your goal is to win three mini-boards in a row on the macro board (horizontally, vertically, or diagonally).

            Game Structure:
            - The board contains 9 mini-boards (a 3x3 grid of Tic Tac Toe boards).
            - Each mini-board has 9 cells (numbered 0-8).
            - Win a mini-board by getting three of your marks in a row within it.
            - Win the game by winning three mini-boards in a row on the macro board.

            Key Rule - Move Constraint:
            Your move determines which mini-board your opponent MUST play in next!
            - When you place your mark in cell 'n' of a mini-board, your opponent's next move must be in mini-board 'n'.
            - To better assist you, the mini-board you are required to play in will be highlighted in yellow.
            - If that required mini-board is already won or completely full, your opponent can play in any other available mini-board.

            NOTE:
            1. Action format: "[macro_board micro_board]" e.g. "[4 0]" to place in mini-board 4, cell 0.
            2. Please be careful not to play in the wrong mini-board.
        """).strip()

        return {
            "player_0": base_description.format(player_id="0", mark="O"),
            "player_1": base_description.format(player_id="1", mark="X"),
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

        logger.info("Reset Ultimate Tic Tac Toe")

        return obs, info

    def _get_font(self, size: int) -> ImageFont.ImageFont:
        font_path = self.assets_dir / "DejaVuSans.ttf"
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size)
        else:
            logger.warning(f"Font file not found: {font_path}, using default font")
            return ImageFont.load_default()

    def _draw_x(self, draw, cx, cy, size, color, width=4):
        """Draw an X mark."""
        offset = size // 2
        draw.line(
            [(cx - offset, cy - offset), (cx + offset, cy + offset)],
            fill=color,
            width=width,
        )
        draw.line(
            [(cx - offset, cy + offset), (cx + offset, cy - offset)],
            fill=color,
            width=width,
        )

    def _draw_o(self, draw, cx, cy, size, color, width=4):
        """Draw an O mark."""
        radius = size // 2
        draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            outline=color,
            width=width,
        )

    def render(self) -> Image.Image:
        """Render the Ultimate Tic Tac Toe board."""
        gs = self._ta_env.state.game_state
        boards = gs["board"]
        macro_board = gs["macro_board"]
        next_board = gs.get("next_micro_board")

        # Total size: 3 mini-boards × mini_board_size + borders
        border = 6
        total_size = self._mini_board_size * 3 + border * 4

        # Create image
        img = Image.new("RGB", (total_size, total_size), (245, 245, 245))
        draw = ImageDraw.Draw(img)

        # Try to load font
        font_size = max(8, self._cell_size // 3)
        font = self._get_font(font_size)

        # Draw each mini-board
        for macro_idx in range(9):
            macro_row = macro_idx // 3
            macro_col = macro_idx % 3

            # Calculate position
            offset_x = border + macro_col * (self._mini_board_size + border)
            offset_y = border + macro_row * (self._mini_board_size + border)

            # Check if this mini-board is the next one to play
            is_next = next_board == macro_idx

            # Check if mini-board is won
            macro_mark = macro_board[macro_row][macro_col]
            is_won = macro_mark != " "

            # Background color
            bg_color = (255, 255, 200) if is_next else (240, 240, 240)
            if is_won:
                bg_color = (200, 255, 200) if macro_mark == "O" else (255, 200, 200)

            # Draw mini-board background
            draw.rectangle(
                [
                    offset_x,
                    offset_y,
                    offset_x + self._mini_board_size,
                    offset_y + self._mini_board_size,
                ],
                fill=bg_color,
                outline=(60, 60, 60),
                width=3 if is_next else 2,
            )

            # Draw grid lines for mini-board
            for i in range(1, 3):
                # Vertical lines
                x = offset_x + i * self._cell_size
                draw.line(
                    [(x, offset_y), (x, offset_y + self._mini_board_size)],
                    fill=(120, 120, 120),
                    width=1,
                )
                # Horizontal lines
                y = offset_y + i * self._cell_size
                draw.line(
                    [(offset_x, y), (offset_x + self._mini_board_size, y)],
                    fill=(120, 120, 120),
                    width=1,
                )

            # Draw marks in cells
            board = boards[macro_idx]
            for micro_idx in range(9):
                micro_row = micro_idx // 3
                micro_col = micro_idx % 3

                cx = offset_x + micro_col * self._cell_size + self._cell_size // 2
                cy = offset_y + micro_row * self._cell_size + self._cell_size // 2

                mark = board[micro_row][micro_col]

                if is_won:
                    # If mini-board is won, show the winning mark large
                    if micro_idx == 4:  # Center cell
                        if macro_mark == "X":
                            self._draw_x(
                                draw, cx, cy, self._cell_size, (220, 20, 60), width=10
                            )
                        elif macro_mark == "O":
                            self._draw_o(
                                draw, cx, cy, self._cell_size, (30, 144, 255), width=10
                            )
                elif mark == "X":
                    self._draw_x(
                        draw, cx, cy, self._cell_size // 3, (220, 20, 60), width=4
                    )
                elif mark == "O":
                    self._draw_o(
                        draw, cx, cy, self._cell_size // 3, (30, 144, 255), width=4
                    )
                else:
                    # Draw cell number for empty cells
                    text = str(micro_idx)
                    draw.text(
                        (cx, cy), text, fill=(180, 180, 180), font=font, anchor="mm"
                    )

            # Draw mini-board index in corner (only if font is available)
            if font:
                text = str(macro_idx)
                draw.text(
                    (offset_x + 3, offset_y + 3), text, fill=(100, 100, 100), font=font
                )

        return img

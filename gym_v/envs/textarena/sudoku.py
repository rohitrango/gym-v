"""Sudoku game using TextArena."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont
import textarena as ta

from gym_v import Env, Observation, get_logger

logger = get_logger()


class TextArenaSudokuEnv(Env):
    """Sudoku number placement puzzle game using TextArena's Sudoku environment.

    The player fills a 9x9 grid with digits 1-9 such that each row, column, and
    3x3 subgrid contains all digits without repetition. The puzzle starts with
    some cells pre-filled as clues. The goal is to complete the entire grid
    following Sudoku rules.

    Args:
        clues: Number of pre-filled cells provided as hints
        cell_size: Size of each cell in pixels for rendering
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    def __init__(
        self,
        clues: int = 30,
        cell_size: int = 50,
        num_players: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._clues = clues
        self._cell_size = cell_size
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._ta_env = ta.make(
            "Sudoku-v0-raw",
            clues=clues,
            max_turns=self._max_episode_steps - 1,
        )

        self._initial_board = None

    @property
    def description(self) -> str:
        return dedent("""
            You are playing Sudoku.
            Here is the current state of the Sudoku grid. Each row is numbered from 1 to 9, and each column is also numbered from 1 to 9.
            There are empty cells and pre-filled cells, and pre-filled cells contain digits from 1 to 9.

            Current Sudoku Grid:
            Your objective is to fill the empty cells in the 9x9 grid with digits from 1 to 9 such that:
            1. Each row contains all digits from 1 to 9 without repetition.
            2. Each column contains all digits from 1 to 9 without repetition.
            3. Each of the nine 3x3 subgrids contains all digits from 1 to 9 without repetition.

            Rules and Instructions:
            1. **Do not overwrite** the initial numbers provided in the grid.
            2. **Only fill** empty cells.
            3. You may respond in any manner you prefer, but ensure that your response includes the format of '[row column number]'.
            4. **Ensure** that your move does not violate Sudoku rules. Invalid moves will result in penalties.
            Examples:
            - **Valid Move**:
              - Grid Snippet Before Move:

              - Move: `[5 3 7]`
              - Explanation: Placing 7 at row 5, column 3 does not violate any Sudoku rules.

            - **Invalid Move** (Overwriting a pre-filled cell):
              - Grid Snippet Before Move:

              - Move: `[1 1 9]`
              - Explanation: Cell (1,1) is already filled with 5. You cannot overwrite it.

            - **Invalid Move** (Violating Sudoku rules):
              - Grid Snippet Before Move:

              - Move: `[1 3 5]`
              - Explanation: Placing 5 in row 1, column 3 violates the rule since 5 already exists in row 1.

            Good luck!
        """).strip()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        super().reset(seed=seed)

        self._ta_env.reset(num_players=self.num_players, seed=seed)
        self._initial_board = self._ta_env.state.game_state["board"].copy()

        logger.info("Reset Sudoku.")

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
        grid_size = 9
        cell_size = self._cell_size
        margin = 30
        thick_line_width = 3  # For 3x3 subgrid borders
        thin_line_width = 1  # For regular cell borders
        board_size = grid_size * cell_size
        img_size = board_size + 2 * margin

        # Create base image
        img = Image.new("RGB", (img_size, img_size), (250, 250, 250))
        draw = ImageDraw.Draw(img)

        # Draw grid background
        grid_color = (255, 255, 255)
        draw.rectangle(
            [margin, margin, margin + board_size, margin + board_size],
            fill=grid_color,
            outline=(100, 100, 100),
            width=2,
        )

        # Draw thin grid lines (for individual cells)
        for i in range(grid_size + 1):
            line_width = thick_line_width if i % 3 == 0 else thin_line_width
            line_color = (0, 0, 0) if i % 3 == 0 else (180, 180, 180)

            # Vertical lines
            x = margin + i * cell_size
            draw.line(
                [x, margin, x, margin + board_size], fill=line_color, width=line_width
            )

            # Horizontal lines
            y = margin + i * cell_size
            draw.line(
                [margin, y, margin + board_size, y], fill=line_color, width=line_width
            )

        # Get current board state
        board = self._ta_env.state.game_state["board"]
        initial_board = self._initial_board

        # Set up font
        font_path = self.assets_dir / "DejaVuSans.ttf"
        if font_path.exists():
            font = ImageFont.truetype(str(font_path), cell_size // 2)
        else:
            logger.warning(f"Font file not found: {font_path}, using default font")
            font = ImageFont.load_default()

        # Draw numbers
        for row in range(grid_size):
            for col in range(grid_size):
                if len(board) > row and len(board[row]) > col and board[row][col] != 0:
                    number = str(board[row][col])

                    # Determine if this is a pre-filled number or player input
                    is_initial = (
                        len(initial_board) > row
                        and len(initial_board[row]) > col
                        and initial_board[row][col] != 0
                    )

                    # Calculate text position (center of cell)
                    x = margin + col * cell_size + cell_size // 2
                    y = margin + row * cell_size + cell_size // 2

                    # Get text dimensions for centering
                    bbox = draw.textbbox((0, 0), number, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]

                    text_x = x - text_width // 2
                    text_y = y - text_height // 2 - bbox[1] // 2  # Adjust for baseline

                    # Choose color based on whether it's original or player input
                    if is_initial:
                        text_color = (0, 0, 0)  # Black for pre-filled numbers
                        shadow_color = (200, 200, 200)
                    else:
                        text_color = (0, 100, 200)  # Blue for player input
                        shadow_color = (150, 150, 150)

                    # Draw text shadow for better visibility
                    draw.text(
                        (text_x + 1, text_y + 1), number, fill=shadow_color, font=font
                    )
                    # Draw main text
                    draw.text((text_x, text_y), number, fill=text_color, font=font)

        return img

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

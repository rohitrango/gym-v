"""Minesweeper game using TextArena."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont
import textarena as ta

from gym_v import Env, Observation, get_logger

logger = get_logger()


class TextArenaMinesweeperEnv(Env):
    """Minesweeper logic puzzle game using TextArena's Minesweeper environment.

    The player reveals cells on a grid to find all safe cells while avoiding mines.
    Each revealed safe cell shows the number of adjacent mines. The goal is to
    reveal all non-mine cells without triggering any mines.

    Args:
        rows: Number of rows in the grid
        cols: Number of columns in the grid
        num_mines: Number of mines randomly placed on the grid
        cell_size: Size of each cell in pixels for rendering
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    def __init__(
        self,
        rows: int = 8,
        cols: int = 8,
        num_mines: int = 10,
        cell_size: int = 64,
        num_players: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._rows = rows
        self._cols = cols
        self._num_mines = num_mines
        self._cell_size = cell_size
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._ta_env = ta.make(
            "Minesweeper-v0-raw",
            rows=rows,
            cols=cols,
            num_mines=num_mines,
            max_turns=self._max_episode_steps - 1,
        )

    @property
    def description(self) -> str:
        return dedent("""
            You are playing the Minesweeper game.
            The objective of the game is to reveal all cells that do not contain mines.
            To make a move, simply specify the row and column coordinates you want to reveal using the format:
            - `[row col]`: Reveal the cell at the specified row and column.
            For example:
            - `[3 2]` to reveal the cell in Row 3, Column 2.
            - `[5 6]` to reveal the cell in Row 5, Column 6.
            On your first move, you will reveal an area around the cell you choose to ensure a safe start.
            Be mindful not to choose already revealed cells.
        """).strip()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        super().reset(seed=seed)

        self._ta_env.reset(num_players=self.num_players, seed=seed)

        logger.info("Reset Minesweeper.")

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
        grid = self._ta_env.grid
        revealed = self._ta_env.revealed

        img_width = self._cols * self._cell_size
        img_height = self._rows * self._cell_size

        # Create base image
        img = Image.new("RGB", (img_width, img_height), (200, 200, 200))
        draw = ImageDraw.Draw(img)

        # Try to load font
        font_path = self.assets_dir / "DejaVuSans.ttf"
        if font_path.exists():
            font = ImageFont.truetype(str(font_path), int(self._cell_size * 0.6))
        else:
            logger.warning(f"Font file not found: {font_path}, using default font")
            font = ImageFont.load_default()

        # Color scheme matching GitLab implementation
        colors = {
            "unrevealed": (189, 189, 189),  # Light gray for unrevealed cells
            "revealed": (255, 255, 255),  # White background for revealed cells
            "mine": (255, 200, 200),  # Light red background for mines
            "border": (128, 128, 128),  # Gray borders
            "text": (0, 0, 0),
        }

        # Number colors matching classic Minesweeper
        number_colors = [
            (0, 0, 0),  # 0 - black (not used)
            (0, 0, 255),  # 1 - blue
            (0, 128, 0),  # 2 - green
            (255, 0, 0),  # 3 - red
            (0, 0, 128),  # 4 - dark blue
            (128, 0, 0),  # 5 - dark red
            (0, 128, 128),  # 6 - teal
            (0, 0, 0),  # 7 - black
            (128, 128, 128),  # 8 - gray
        ]

        for r in range(self._rows):
            for c in range(self._cols):
                x0 = c * self._cell_size
                y0 = r * self._cell_size
                x1 = x0 + self._cell_size
                y1 = y0 + self._cell_size

                # Draw cell background
                if revealed[r][c]:
                    if grid[r][c] == -1:  # Mine
                        draw.rectangle([x0, y0, x1, y1], fill=colors["mine"])
                        # Draw mine symbol
                        text_x = x0 + self._cell_size // 2
                        text_y = y0 + self._cell_size // 2
                        draw.text(
                            (text_x, text_y),
                            "*",
                            fill=colors["text"],
                            font=font,
                            anchor="mm",
                        )
                    else:
                        draw.rectangle([x0, y0, x1, y1], fill=colors["revealed"])
                        # Draw number if > 0
                        if grid[r][c] > 0:
                            text_x = x0 + self._cell_size // 2
                            text_y = y0 + self._cell_size // 2
                            color = number_colors[
                                min(grid[r][c], len(number_colors) - 1)
                            ]
                            draw.text(
                                (text_x, text_y),
                                str(grid[r][c]),
                                fill=color,
                                font=font,
                                anchor="mm",
                            )
                else:
                    draw.rectangle([x0, y0, x1, y1], fill=colors["unrevealed"])

                # Draw border
                draw.rectangle([x0, y0, x1, y1], outline=colors["border"], width=1)

        return img

    def _get_observation_text(self) -> str:
        _, ta_obs = self._ta_env.get_observation()
        obs_text = []

        for _, msg, type in ta_obs:
            if type in [
                ta.ObservationType.GAME_ADMIN,
                ta.ObservationType.GAME_ACTION_DESCRIPTION,
            ]:
                obs_text.append(msg)

        if "reason" in self._ta_env.state.game_info[0]:
            obs_text.append(self._ta_env.state.game_info[0]["reason"])

        obs_text = "\n".join(obs_text) if obs_text else None

        return obs_text

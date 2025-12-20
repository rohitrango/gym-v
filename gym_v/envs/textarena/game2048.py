"""Game2048 game using TextArena."""

from __future__ import annotations

from functools import cached_property
from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont
import textarena as ta

from gym_v import Env, Observation, get_logger

logger = get_logger()


class TextArenaGame2048Env(Env):
    """2048 tile-merging puzzle game using TextArena's 2048 environment.

    The player slides numbered tiles on a grid. When two tiles with the same
    number touch, they merge into one tile with double the value. The goal is to
    create a tile with the target value by combining tiles strategically.

    Args:
        target_tile: Target tile value to achieve (must be a power of 2 from 4 to 8192)
        tile_size: Size of each tile in pixels for rendering
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    def __init__(
        self,
        target_tile: int = 2048,
        tile_size: int = 100,
        **kwargs,
    ):
        super().__init__(**kwargs)
        ALLOWED_TARGET_TILES = {2**i for i in range(2, 14)}  # 4~8192
        assert (
            target_tile in ALLOWED_TARGET_TILES
        ), f"Target tile must be one of {sorted(ALLOWED_TARGET_TILES)}"
        self._target_tile = target_tile
        self._tile_size = tile_size

        self._ta_env = ta.make(
            "2048-v0-raw",
            target_tile=target_tile,
        )

    @property
    def description(self) -> str:
        return dedent(f"""
            You are playing 2048 on a {self._ta_env.board_size}x{self._ta_env.board_size} board. Your goal is to reach a {self._target_tile} tile by sliding identical numbers together!
            Valid moves: [Up], [Down], [Left], [Right]. Tiles combine when they collide, doubling their value.
        """).strip()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)

        self._ta_env.reset(num_players=1, seed=seed)

        logger.info("Reset Game2048.")

        obs = Observation(image=self.render(), text=self._get_observation_text())
        info = {}

        return obs, info

    def inner_step(
        self, action: str
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        info = {}
        done, _ = self._ta_env.step(action)

        info["invalid_action"] = (
            self._ta_env.state.error_count > 0
            or self._ta_env.state.game_info[0]["invalid_move"]
        )

        if done:
            reward = self._ta_env.state.rewards[0]
            terminated = True
            truncated = False
        elif self._current_episode_steps >= self._max_episode_steps:
            reward = (
                self._ta_env.state.rewards[0]
                if self._ta_env.state.rewards
                else self._ta_env._get_percentage_completion()
            )
            terminated = True
            truncated = False
        else:
            reward = 0
            terminated = False
            truncated = False

        obs = Observation(image=self.render(), text=self._get_observation_text())

        return obs, reward, terminated, truncated, info

    def render(self) -> Image.Image:
        board_state = self._ta_env.state.game_state["board"]

        size = self._tile_size * 4
        cell_px = size // 4
        padding = max(2, cell_px // 10)

        img = Image.new("RGB", (size, size), (250, 248, 239))
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, size, size], fill=(187, 173, 160))

        base_font_sz = cell_px // 3

        for r in range(4):
            for c in range(4):
                val = int(board_state[r][c])
                x0 = c * cell_px + padding
                y0 = r * cell_px + padding
                x1 = (c + 1) * cell_px - padding
                y1 = (r + 1) * cell_px - padding

                col = self._colors_2048.get(val)
                draw.rectangle(
                    [x0, y0, x1, y1],
                    fill=col,
                    outline=(187, 173, 160),
                    width=max(1, padding // 4),
                )

                if val == 0:
                    continue

                # Dynamic font size for large numbers
                text = str(val)
                fsz = base_font_sz
                if len(text) == 3:
                    fsz = int(base_font_sz * 0.8)
                elif len(text) >= 4:
                    fsz = int(base_font_sz * 0.65)
                font = self._get_font(fsz)

                tw, th = draw.textbbox((0, 0), text, font=font)[2:]
                cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
                draw.text(
                    (cx - tw // 2, cy - th // 2),
                    text,
                    fill=self._light_text if val > 4 else self._dark_text,
                    font=font,
                )

        return img

    @cached_property
    def _colors_2048(self) -> dict[int, tuple[int, int, int]]:
        """Colors for each tile value as RGB"""
        return {
            0: (205, 193, 180),
            2: (238, 228, 218),
            4: (237, 224, 200),
            8: (242, 177, 121),
            16: (245, 149, 99),
            32: (246, 124, 95),
            64: (246, 94, 59),
            128: (237, 207, 114),
            256: (237, 204, 97),
            512: (237, 200, 80),
            1024: (237, 197, 63),
            2048: (237, 194, 46),
            4096: (60, 58, 50),
            8192: (60, 58, 50),
        }

    @cached_property
    def _dark_text(self) -> tuple[int, int, int]:
        return (119, 110, 101)

    @cached_property
    def _light_text(self) -> tuple[int, int, int]:
        return (249, 246, 242)

    def _get_font(self, size: int) -> ImageFont.ImageFont:
        font_path = self.assets_dir / "DejaVuSans.ttf"
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size)
        else:
            logger.warning(f"Font file not found: {font_path}, using default font")
            return ImageFont.load_default()

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

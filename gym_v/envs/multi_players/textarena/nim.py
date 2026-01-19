"""Nim game using TextArena."""

from __future__ import annotations

from collections import defaultdict
from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation, get_logger
import textarena as ta

logger = get_logger()


class TextArenaNim(Env):
    """
    Nim game using TextArena's Nim environment with visual rendering.

    Players take turns removing objects from piles. Last to take wins.
    """

    assets_dir = resources.files("gym_v.envs") / "assets"
    is_deterministic = True

    def __init__(
        self,
        piles: list[int] = None,
        pile_width: int = 100,
        num_players: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._initial_piles = piles if piles is not None else [3, 4, 5]
        self._pile_width = pile_width
        self.num_players = num_players

        # TextArena Nim environment
        self._ta_env = ta.make("Nim-v0-raw", piles=self._initial_piles)

        self._agent_ids = {"player_0", "player_1"}
        self.possible_players = ["player_0", "player_1"]

    def _get_current_player(self) -> str:
        """Get the current acting player."""
        current_ta_player_id = self._ta_env.state.current_player_id
        return self.possible_players[current_ta_player_id]

    @property
    def description(self) -> dict[str, str]:
        base_description = dedent("""
            You are Player {player_id} in a game of Nim.

            Rules:
            - On your turn, remove at least one object from exactly one pile
            - Whoever takes the last object(s) wins

            NOTE:
            1. Action format: "[pile_index quantity]" e.g. "[0 2]" to take 2 objects from pile 0.
            2. Only legal moves are permitted.
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

        logger.info(f"Reset Nim: piles {self._initial_piles}")

        return obs, info

    def _get_font(self, size: int) -> ImageFont.ImageFont:
        font_path = self.assets_dir / "DejaVuSans.ttf"
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size)
        else:
            logger.warning(f"Font file not found: {font_path}, using default font")
            return ImageFont.load_default()

    def render(self) -> Image.Image:
        """Render the Nim piles."""
        piles = self._ta_env.state.game_state["piles"]
        num_piles = len(self._initial_piles)
        max_pile = max(self._initial_piles) if self._initial_piles else 1

        # Calculate all sizes based on pile_width
        pile_width = self._pile_width
        pile_spacing = int(pile_width * 0.6)
        margin = int(pile_width * 0.6)
        object_size = int(pile_width * 0.4)
        object_spacing = int(pile_width * 0.15)

        width = num_piles * pile_width + (num_piles - 1) * pile_spacing + 2 * margin
        height = (
            max_pile * (object_size + object_spacing)
            + 2 * margin
            + int(pile_width * 0.6)
        )

        # Create gradient background
        img = Image.new("RGB", (width, height))
        draw = ImageDraw.Draw(img)

        # Draw gradient background (dark blue to lighter blue)
        for y in range(height):
            ratio = y / height
            r = int(30 + ratio * 25)
            g = int(40 + ratio * 35)
            b = int(60 + ratio * 45)
            draw.line([(0, y), (width, y)], fill=(r, g, b))

        # Draw each pile
        for pile_idx, count in enumerate(piles):
            x_center = margin + pile_idx * (pile_width + pile_spacing) + pile_width // 2

            # Draw pile base (wooden platform)
            platform_width = int(pile_width * 0.9)
            platform_height = int(pile_width * 0.1)
            platform_y = height - margin - platform_height
            draw.rectangle(
                [
                    x_center - platform_width // 2,
                    platform_y,
                    x_center + platform_width // 2,
                    platform_y + platform_height,
                ],
                fill=(139, 90, 43),  # Brown
                outline=(101, 67, 33),
                width=max(1, int(pile_width * 0.02)),
            )

            # Draw pile label
            font = self._get_font(int(pile_width * 0.24))
            label = f"Pile {pile_idx}"
            bbox = draw.textbbox((0, 0), label, font=font, anchor="mm")
            label_width = bbox[2] - bbox[0]
            label_height = bbox[3] - bbox[1]

            # Draw label with semi-transparent background
            label_bg_padding = int(pile_width * 0.08)
            label_y = int(pile_width * 0.35)
            draw.rounded_rectangle(
                [
                    x_center - label_width // 2 - label_bg_padding,
                    label_y - label_height // 2 - label_bg_padding,
                    x_center + label_width // 2 + label_bg_padding,
                    label_y + label_height // 2 + label_bg_padding,
                ],
                radius=int(pile_width * 0.08),
                fill=(255, 255, 255),
                outline=(200, 200, 200),
                width=max(1, int(pile_width * 0.02)),
            )
            draw.text(
                (x_center, label_y),
                label,
                fill=(40, 40, 40),
                font=font,
                anchor="mm",
            )

            # Draw coins stacked on platform
            for obj_idx in range(count):
                y_pos = (
                    platform_y
                    - int(pile_width * 0.05)
                    - (obj_idx + 1) * (object_size + object_spacing)
                )

                # Draw shadow
                shadow_offset = max(1, int(pile_width * 0.03))
                for i in range(3):
                    draw.ellipse(
                        [
                            x_center - object_size // 2 + shadow_offset + i,
                            y_pos + shadow_offset + i,
                            x_center + object_size // 2 + shadow_offset + i,
                            y_pos + object_size + shadow_offset + i,
                        ],
                        fill=(0, 0, 0),
                    )

                # Draw gold coin with gradient
                for i in range(object_size // 2):
                    ratio = i / (object_size // 2)
                    r = int(255 - ratio * 40)
                    g = int(215 - ratio * 50)
                    b = int(0 + ratio * 20)

                    radius = object_size // 2 - i
                    draw.ellipse(
                        [
                            x_center - radius,
                            y_pos + i,
                            x_center + radius,
                            y_pos + object_size - i,
                        ],
                        fill=(r, g, b),
                    )

                # Draw highlight
                highlight_size = object_size // 4
                draw.ellipse(
                    [
                        x_center - object_size // 4,
                        y_pos + object_size // 6,
                        x_center - object_size // 4 + highlight_size,
                        y_pos + object_size // 6 + highlight_size,
                    ],
                    fill=(255, 255, 220),
                )

                # Draw outline
                draw.ellipse(
                    [
                        x_center - object_size // 2,
                        y_pos,
                        x_center + object_size // 2,
                        y_pos + object_size,
                    ],
                    outline=(200, 160, 0),
                    width=max(1, int(pile_width * 0.03)),
                )

        return img

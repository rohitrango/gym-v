"""PegJump game using TextArena."""

from __future__ import annotations

from importlib import resources
import math
from textwrap import dedent
from types import MethodType
from typing import Any

from PIL import Image, ImageDraw, ImageFont
import textarena as ta

from gym_v import Env, Observation, get_logger

logger = get_logger()


class TextArenaPegJumpEnv(Env):
    """Peg Jump solitaire puzzle game using TextArena's PegJump environment.

    The player jumps pegs over adjacent pegs on a triangular board to remove them.
    A peg can jump over an adjacent peg into an empty hole, removing the jumped peg.
    The goal is to finish with exactly one peg remaining on the board.

    Args:
        initial_empty: Position (1-15) of the initially empty hole on the board
        peg_size: Size of each peg in pixels for rendering
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    def __init__(
        self,
        initial_empty: int = 1,
        peg_size: int = 80,
        num_players: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._initial_empty = initial_empty
        self._peg_size = peg_size
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._ta_env = ta.make(
            "PegJump-v0-raw",
            initial_empty=initial_empty,
        )

        corrected_base_triples = [
            (1, 2, 4),
            (1, 3, 6),
            (2, 4, 7),
            (2, 5, 9),
            (3, 5, 8),
            (3, 6, 10),
            (4, 5, 6),
            (4, 7, 11),
            (4, 8, 13),
            (5, 8, 12),
            (5, 9, 14),
            (6, 9, 13),
            (6, 10, 15),
            (7, 8, 9),
            (8, 9, 10),
            (11, 12, 13),
            (12, 13, 14),
            (13, 14, 15),
        ]

        def _get_percentage_completion(self) -> float:
            return 1 - (self.state.game_state["board"].count(True) - 1) / 13

        # Apply patch to TextArena environment
        self._ta_env._BASE_TRIPLES = corrected_base_triples
        self._ta_env._get_percentage_completion = MethodType(
            _get_percentage_completion, self._ta_env
        )

        # Generate ALLOWED_MOVES from corrected base triples (both directions)
        allowed_moves = []
        for from_pos, over_pos, to_pos in corrected_base_triples:
            allowed_moves.extend(
                [(from_pos, over_pos, to_pos), (to_pos, over_pos, from_pos)]
            )
        self._ta_env.ALLOWED_MOVES = allowed_moves

    @property
    def description(self) -> str:
        return dedent("""
            You are playing PegJump. Jump one peg over another into an empty hole, removing the jumped peg.
            Goal: finish with exactly **one** peg left. Action format: e.g. '[4 1]'.
        """).strip()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        super().reset(seed=seed)

        self._ta_env.reset(num_players=self.num_players, seed=seed)

        logger.info("Reset PegJump.")

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
        board = self._ta_env.state.game_state["board"]

        # Calculate board dimensions based on peg_size
        peg_radius = self._peg_size // 2
        spacing = self._peg_size
        board_width = (
            5 * self._peg_size + 6 * spacing
        )  # Enough space for 5 pegs in bottom row
        board_height = int(
            (5 * self._peg_size + 4 * spacing) * math.sqrt(3) / 2 + 2 * spacing
        )  # Triangle height ratio

        # Create base image
        img = Image.new("RGB", (board_width, board_height), (240, 240, 240))
        draw = ImageDraw.Draw(img)

        # Draw wooden triangle board background
        triangle_margin = spacing // 2
        triangle_points = [
            (board_width // 2, triangle_margin),  # Top
            (triangle_margin, board_height - triangle_margin),  # Bottom left
            (
                board_width - triangle_margin,
                board_height - triangle_margin,
            ),  # Bottom right
        ]

        # Draw triangle shadow
        shadow_points = [(x + 3, y + 3) for x, y in triangle_points]
        draw.polygon(shadow_points, fill=(160, 120, 80))

        # Draw main triangle (wood color)
        draw.polygon(
            triangle_points, fill=(210, 180, 140), outline=(150, 120, 90), width=3
        )

        # Get positions for pegs
        positions = self._get_peg_positions(board_width, board_height, spacing)

        # Load font based on peg_radius for consistent sizing
        font_path = self.assets_dir / "DejaVuSans.ttf"
        if font_path.exists():
            font = ImageFont.truetype(
                str(font_path), peg_radius
            )  # Use peg_radius for font size
        else:
            logger.warning(f"Font file not found: {font_path}, using default font")
            font = ImageFont.load_default()

        for pos in range(1, 16):  # Positions 1-15
            x, y = positions[pos - 1]

            # First draw hole (always visible)
            hole_radius = peg_radius + 2
            draw.ellipse(
                [x - hole_radius, y - hole_radius, x + hole_radius, y + hole_radius],
                fill=(120, 90, 60),
                outline=(80, 60, 40),
                width=2,
            )

            # Draw peg if present
            if board[pos]:
                # Red peg with 3D effect
                # Shadow
                draw.ellipse(
                    [
                        x - peg_radius + 1,
                        y - peg_radius + 1,
                        x + peg_radius + 1,
                        y + peg_radius + 1,
                    ],
                    fill=(150, 50, 50),
                )

                # Main peg
                draw.ellipse(
                    [x - peg_radius, y - peg_radius, x + peg_radius, y + peg_radius],
                    fill=(200, 60, 60),
                    outline=(120, 40, 40),
                    width=2,
                )

                # Highlight for 3D effect
                highlight_radius = peg_radius // 3
                draw.ellipse(
                    [
                        x - highlight_radius - peg_radius // 3,
                        y - highlight_radius - peg_radius // 3,
                        x - peg_radius // 3 + highlight_radius,
                        y - peg_radius // 3 + highlight_radius,
                    ],
                    fill=(220, 100, 100),
                )

            # Draw position number on the peg/hole
            text = str(pos)
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            text_x = x - text_width // 2
            text_y = y - text_height // 2 - bbox[1] // 2  # Adjust for baseline offset

            # Draw text shadow for better visibility
            draw.text((text_x + 1, text_y + 1), text, fill=(0, 0, 0), font=font)
            # Draw main text
            text_color = (255, 255, 255) if board[pos] else (255, 255, 255)
            draw.text((text_x, text_y), text, fill=text_color, font=font)

        return img

    def _get_peg_positions(
        self, board_width: int, board_height: int, spacing: int
    ) -> list[tuple[int, int]]:
        """Generate 15 peg positions in the triangular board."""
        tri_margin = spacing // 2
        A = (board_width / 2.0, float(tri_margin))
        B = (float(tri_margin), float(board_height - tri_margin))
        C = (float(board_width - tri_margin), float(board_height - tri_margin))

        peg_radius = self._peg_size / 2.0
        hole_radius = peg_radius + 2.0
        extra_pad = max(2.0, peg_radius / 6.0)
        m_base = hole_radius + extra_pad

        h_out = board_height - 2 * tri_margin

        inset_ratio = 0.1
        m_extra = (inset_ratio * h_out) / 3.0

        m = min(m_base + m_extra, h_out / 3.0 - 1.0)

        s = (h_out - 3.0 * m) / h_out

        Gx = (A[0] + B[0] + C[0]) / 3.0
        Gy = (A[1] + B[1] + C[1]) / 3.0

        def shrink(P):
            return (Gx + s * (P[0] - Gx), Gy + s * (P[1] - Gy))

        A2, B2, C2 = shrink(A), shrink(B), shrink(C)

        n = 4
        positions: list[tuple[int, int]] = []
        for r in range(n + 1):
            for i in range(r + 1):
                wA = (n - r) / n
                wB = (r - i) / n
                wC = i / n
                x = wA * A2[0] + wB * B2[0] + wC * C2[0]
                y = wA * A2[1] + wB * B2[1] + wC * C2[1]
                positions.append((int(round(x)), int(round(y))))
        return positions

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

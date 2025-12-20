"""RushHour game using TextArena."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any, Literal

from PIL import Image, ImageDraw, ImageFont
import textarena as ta

from gym_v import Env, Observation, get_logger

logger = get_logger()


class TextArenaRushHourEnv(Env):
    """RushHour sliding block puzzle game using TextArena's RushHour environment.

    The player slides vehicles on a 6x6 grid to clear a path for the red car (X)
    to exit through the right edge. Vehicles can only move forward or backward
    in their orientation (horizontal or vertical). The goal is to free the red car
    with the minimum number of moves.

    Args:
        difficulty: Puzzle difficulty level ("easy", "medium", or "hard")
        cell_size: Size of each grid cell in pixels for rendering
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    def __init__(
        self,
        difficulty: Literal["easy", "medium", "hard"] = "easy",
        cell_size: int = 80,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._difficulty = difficulty
        self._cell_size = cell_size

        self._ta_env = ta.make(
            "RushHour-v0-raw",
            difficulty=difficulty,
        )

    @property
    def description(self) -> str:
        return dedent(f"""
            You are playing a {self._difficulty} RushHour puzzle. Slide cars to free the red car [X] and drive it out the right edge.
            Actions: [A+], [B-], etc.  (+ = forward, - = backward).
        """).strip()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)

        self._ta_env.reset(num_players=1, seed=seed)

        logger.info("Reset RushHour.")

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
        # Constants for rendering
        grid_size = 6
        cell_size = self._cell_size
        margin = 20
        board_size = grid_size * cell_size
        img_size = board_size + 2 * margin

        # Create base image
        img = Image.new("RGB", (img_size + cell_size * 2, img_size), (240, 240, 240))
        draw = ImageDraw.Draw(img)

        # Draw grid background
        grid_color = (220, 220, 220)
        draw.rectangle(
            [margin, margin, margin + board_size, margin + board_size],
            fill=grid_color,
            outline=(180, 180, 180),
            width=2,
        )

        # Draw grid lines
        for i in range(grid_size + 1):
            # Vertical lines
            x = margin + i * cell_size
            draw.line(
                [x, margin, x, margin + board_size], fill=(180, 180, 180), width=1
            )

            # Horizontal lines
            y = margin + i * cell_size
            draw.line(
                [margin, y, margin + board_size, y], fill=(180, 180, 180), width=1
            )

        # Highlight the exit area (right edge, row 2) - make it more prominent
        exit_y = margin + 2 * cell_size
        exit_width = cell_size * 1.5
        exit_rect = [
            margin + board_size,
            exit_y,
            margin + board_size + exit_width,
            exit_y + cell_size,
        ]

        # Draw exit with gradient effect (multiple rectangles for depth)
        draw.rectangle(exit_rect, fill=(255, 100, 100), outline=(180, 0, 0), width=3)

        # Add inner highlight
        inner_rect = [
            exit_rect[0] + 3,
            exit_rect[1] + 3,
            exit_rect[2] - 3,
            exit_rect[3] - 3,
        ]
        draw.rectangle(inner_rect, fill=(255, 150, 150), outline=(220, 50, 50), width=2)

        # Add "EXIT" text
        font_path = self.assets_dir / "DejaVuSans.ttf"
        if font_path.exists():
            exit_font = ImageFont.truetype(
                str(font_path), cell_size // 4
            )  # Use peg_radius for font size
        else:
            exit_font = ImageFont.load_default()

        exit_text = "EXIT"
        exit_bbox = draw.textbbox((0, 0), exit_text, font=exit_font)
        exit_text_width = exit_bbox[2] - exit_bbox[0]
        exit_text_height = exit_bbox[3] - exit_bbox[1]
        exit_text_x = exit_rect[0] + (exit_width - exit_text_width) // 2
        exit_text_y = exit_rect[1] + (cell_size - exit_text_height) // 2

        # Draw exit text with shadow
        draw.text(
            (exit_text_x + 1, exit_text_y + 1),
            exit_text,
            fill=(0, 0, 0),
            font=exit_font,
        )
        draw.text(
            (exit_text_x, exit_text_y), exit_text, fill=(255, 255, 255), font=exit_font
        )

        # Add bigger exit arrow
        arrow_x = margin + board_size + exit_width + 5
        arrow_y = exit_y + cell_size // 2
        arrow_points = [
            (arrow_x, arrow_y - 12),
            (arrow_x + 18, arrow_y),
            (arrow_x, arrow_y + 12),
            (arrow_x + 5, arrow_y),  # Make arrow more pronounced
        ]
        draw.polygon(arrow_points, fill=(220, 50, 50), outline=(150, 0, 0), width=2)

        # Get game state
        vehicles_dict = self._ta_env.state.game_state["vehicles"]

        # Convert vehicle dict to list format for rendering
        vehicles = []
        for vid, vehicle_obj in vehicles_dict.items():
            vehicles.append(
                {
                    "vid": vid,
                    "row": vehicle_obj.row,
                    "col": vehicle_obj.col,
                    "length": vehicle_obj.length,
                    "orientation": "horizontal"
                    if vehicle_obj.horizontal
                    else "vertical",
                }
            )

        # Draw vehicles
        for vehicle in vehicles:
            vid = vehicle.get("vid")
            row = vehicle.get("row")
            col = vehicle.get("col")
            length = vehicle.get("length")
            orientation = vehicle.get("orientation")

            color = self._get_vehicle_color(vid)

            # Calculate vehicle rectangle
            padding = 3
            if orientation == "horizontal":
                x1 = margin + col * cell_size + padding
                y1 = margin + row * cell_size + padding
                x2 = margin + (col + length) * cell_size - padding
                y2 = margin + (row + 1) * cell_size - padding
            else:  # vertical
                x1 = margin + col * cell_size + padding
                y1 = margin + row * cell_size + padding
                x2 = margin + (col + 1) * cell_size - padding
                y2 = margin + (row + length) * cell_size - padding

            # Draw vehicle with rounded corners
            draw.rounded_rectangle(
                [x1, y1, x2, y2], radius=8, fill=color, outline=(0, 0, 0), width=2
            )

            # Add vehicle front indicator (small arrow or line)
            front_color = (255, 255, 255)  # White for front indicator
            if orientation == "horizontal":
                # Horizontal car: front is on the right side
                # Draw triangle pointing right (bigger)
                front_points = [
                    (x2 - cell_size // 3, y1 + (y2 - y1) // 2 - cell_size // 6),
                    (x2 - cell_size // 6, y1 + (y2 - y1) // 2),
                    (x2 - cell_size // 3, y1 + (y2 - y1) // 2 + cell_size // 6),
                ]
                draw.polygon(
                    front_points, fill=front_color, outline=(200, 200, 200), width=1
                )
            else:
                # Vertical car: front is at the bottom
                # Draw triangle pointing down (bigger)
                front_points = [
                    (x1 + (x2 - x1) // 2 - cell_size // 6, y2 - cell_size // 3),
                    (x1 + (x2 - x1) // 2 + cell_size // 6, y2 - cell_size // 3),
                    (x1 + (x2 - x1) // 2, y2 - cell_size // 6),
                ]
                draw.polygon(
                    front_points, fill=front_color, outline=(200, 200, 200), width=1
                )

            # Add vehicle ID text
            if font_path.exists():
                font = ImageFont.truetype(
                    str(font_path), cell_size // 2
                )  # Use peg_radius for font size
            else:
                logger.warning(f"Font file not found: {font_path}, using default font")
                font = ImageFont.load_default()

            bbox = draw.textbbox((0, 0), vid, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            text_x = x1 + (x2 - x1 - text_width) // 2
            text_y = (
                y1 + (y2 - y1 - text_height) // 2 - bbox[1] // 2
            )  # Adjust for baseline offset

            # Draw text shadow for better visibility
            draw.text((text_x + 1, text_y + 1), vid, fill=(0, 0, 0), font=font)
            # Draw main text
            draw.text((text_x, text_y), vid, fill=(255, 255, 255), font=font)

        return img

    def _get_vehicle_color(self, vehicle_id: str) -> tuple[int, int, int]:
        if vehicle_id == "X":  # Red car (target)
            return (220, 50, 50)

        # Color mapping for other vehicles
        colors = {
            "A": (100, 150, 255),  # Blue
            "B": (100, 255, 150),  # Green
            "C": (255, 200, 100),  # Orange
            "D": (255, 100, 200),  # Pink
            "E": (150, 100, 255),  # Purple
            "F": (255, 255, 100),  # Yellow
            "G": (100, 255, 255),  # Cyan
            "H": (255, 150, 100),  # Light Orange
            "I": (150, 255, 100),  # Light Green
            "J": (100, 200, 255),  # Light Blue
        }

        return colors[vehicle_id]

    def _get_observation_text(self) -> str:
        _, ta_obs = self._ta_env.get_observation()
        obs_text = []

        for _, msg, type in ta_obs:
            if type in [ta.ObservationType.GAME_ADMIN, ta.ObservationType.GAME_MESSAGE]:
                obs_text.append(msg)

        if "reason" in self._ta_env.state.game_info[0]:
            obs_text.append(self._ta_env.state.game_info[0]["reason"])

        obs_text = "\n".join(obs_text) if obs_text else None

        return obs_text

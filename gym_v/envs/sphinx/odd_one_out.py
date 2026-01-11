"""Odd One Out environments with procedural generation."""

from __future__ import annotations

from abc import abstractmethod
from textwrap import dedent
from typing import Any

from PIL import Image

from gym_v import Env, Observation, get_logger
from gym_v.envs.sphinx.utils import (
    POLY_STYLES,
    TRANSFORMS,
    _images_are_similar,
    apply_transform,
    compose_odd_one_out_8_options,
    generate_random_grid,
    generate_random_polygon,
)

logger = get_logger()


class SphinxOddOneOutBaseEnv(Env):
    """Base class for Odd One Out tasks.

    Display 8 shapes where 7 are transformations of the same shape,
    and 1 is different. The task is to identify the odd one out.
    """

    def __init__(
        self,
        option_size: int = 200,
        padding: int = 15,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._option_size = option_size
        self._padding = padding

        self._composed_image: Image.Image | None = None
        self._oracle_answer: str | None = None
        self._odd_idx: int | None = None

    @property
    def description(self) -> str:
        """Return task description for Odd One Out."""
        return dedent("""
            Look at the 8 shapes below.
            7 of them are transformations (rotations/reflections) of the same shape.
            1 of them is different (the "odd one out").

            Your task: Identify which option (a)-(h) is the odd one out.

            Possible transformations include:
            - Rotation: 90° clockwise, 180°, or 90° counterclockwise
            - Reflection: across horizontal, vertical, main diagonal, or anti-diagonal axis

            Answer format: A single letter in parentheses, e.g., (a), (b), ..., (h)
        """).strip()

    @abstractmethod
    def _generate_shape(self) -> Image.Image:
        """Generate a single random shape.

        Returns:
            PIL Image of the shape
        """
        pass

    @abstractmethod
    def _get_metadata(self) -> dict[str, Any]:
        """Get environment-specific metadata for observation."""
        pass

    @abstractmethod
    def _log_reset(self) -> None:
        """Log reset information."""
        pass

    def _prepare_reset(self) -> None:
        """Hook for subclasses to prepare before shape generation.

        Called after seed is set but before any shapes are generated.
        Useful for Poly envs to select a consistent style for all shapes.
        """
        pass

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)
        self._prepare_reset()

        # Generate base shape for the "same" group
        base_shape = self._generate_shape()

        # Generate different shape for odd one out
        odd_shape = self._generate_shape()

        # Ensure odd_shape is visually different (regenerate if too similar)
        max_attempts = 10
        for _ in range(max_attempts):
            if not _images_are_similar(base_shape, odd_shape):
                break
            odd_shape = self._generate_shape()

        # Generate 7 transforms of base shape (use all 8 transforms, pick 7)
        transforms_to_use = list(TRANSFORMS)
        self.np_random.shuffle(transforms_to_use)
        same_shapes = [apply_transform(base_shape, t) for t in transforms_to_use[:7]]

        # Combine: 7 same + 1 odd
        all_options = same_shapes + [odd_shape]

        # Shuffle and track odd position
        indices = list(range(8))
        self.np_random.shuffle(indices)
        shuffled_options = [all_options[i] for i in indices]
        self._odd_idx = indices.index(7)  # 7 was the odd one's original index

        # Compose image
        self._composed_image = compose_odd_one_out_8_options(
            shuffled_options,
            self._odd_idx,
            option_size=self._option_size,
            padding=self._padding,
        )

        labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]
        self._oracle_answer = labels[self._odd_idx]

        obs_text = (
            "7 shapes are transformations of the same original. "
            "Which option (a)-(h) is the odd one out?"
        )

        self._log_reset()

        obs = Observation(
            image=self.render(),
            text=obs_text,
            metadata={"odd_idx": self._odd_idx, **self._get_metadata()},
        )
        info = {
            "oracle_answer": self._oracle_answer,
            "odd_idx": self._odd_idx,
            **self._get_metadata(),
        }
        return obs, info

    def inner_step(
        self, action: str
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        """Evaluate the action against the correct answer."""
        action_clean = action.strip().lower()
        if not action_clean.startswith("("):
            action_clean = f"({action_clean})"
        if not action_clean.endswith(")"):
            action_clean = action_clean + ")"

        correct = action_clean == self._oracle_answer.lower()
        reward = 1.0 if correct else 0.0

        obs = Observation(
            image=self.render(),
            text=None,
            metadata={
                "odd_idx": self._odd_idx,
                "user_answer": action,
                "correct": correct,
            },
        )
        info = {
            "oracle_answer": self._oracle_answer,
            "odd_idx": self._odd_idx,
            "correct": correct,
        }

        return obs, reward, True, False, info

    def render(self) -> Image.Image:
        """Return the composed image with 8 options."""
        return self._composed_image


class SphinxOddOneOutEnv(SphinxOddOneOutBaseEnv):
    """Odd One Out task with colored grid patterns.

    Args:
        grid_size: Size of the grid (grid_size x grid_size)
        num_colors: Number of colors to use in the grid
        cell_size: Pixel size of each cell in the grid
        option_size: Size of each option image in pixels
        padding: Padding between elements in the composed image
    """

    def __init__(
        self,
        grid_size: int = 5,
        num_colors: int = 4,
        cell_size: int = 40,
        option_size: int = 200,
        padding: int = 15,
        **kwargs: Any,
    ):
        super().__init__(option_size=option_size, padding=padding, **kwargs)
        self._grid_size = grid_size
        self._num_colors = num_colors
        self._cell_size = cell_size

    def _generate_shape(self) -> Image.Image:
        return generate_random_grid(
            self.np_random,
            grid_size=self._grid_size,
            num_colors=self._num_colors,
            cell_size=self._cell_size,
        )

    def _get_metadata(self) -> dict[str, Any]:
        return {
            "grid_size": self._grid_size,
            "num_colors": self._num_colors,
        }

    def _log_reset(self) -> None:
        logger.info(
            f"Reset Sphinx OddOneOut: answer={self._oracle_answer}, "
            f"grid_size={self._grid_size}"
        )


class SphinxOddOneOutPolyEnv(SphinxOddOneOutBaseEnv):
    """Odd One Out task with polygon shapes.

    Args:
        img_size: Size of each shape image in pixels
        num_points: Number of vertices in the polygon
        line_width: Width of polygon lines
        grid_divisions: Number of grid divisions in background
        option_size: Size of each option image in the composed output
        padding: Padding between elements in the composed image
        style: Visual style ('outline', 'filled', 'nested', 'striped',
               'gradient', '3d', 'composite', 'pixelated'),
               or None for random selection each reset
    """

    def __init__(
        self,
        img_size: int = 200,
        num_points: int = 8,
        line_width: int = 3,
        grid_divisions: int = 8,
        option_size: int = 200,
        padding: int = 15,
        style: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(option_size=option_size, padding=padding, **kwargs)
        self._img_size = img_size
        self._num_points = num_points
        self._line_width = line_width
        self._grid_divisions = grid_divisions
        self._style = style
        self._current_style: str | None = None

    def _prepare_reset(self) -> None:
        """Select a consistent style for all shapes in this reset."""
        if self._style is not None:
            self._current_style = self._style
        else:
            # Random selection - same style for all shapes in this episode
            self._current_style = POLY_STYLES[
                int(self.np_random.integers(0, len(POLY_STYLES)))
            ]

    def _generate_shape(self) -> Image.Image:
        shape = generate_random_polygon(
            self.np_random,
            img_size=self._img_size,
            num_points=self._num_points,
            line_width=self._line_width,
            grid_lines=True,
            grid_divisions=self._grid_divisions,
            style=self._current_style,  # Use pre-selected style
        )

        return shape

    def _get_metadata(self) -> dict[str, Any]:
        return {
            "num_points": self._num_points,
            "style": self._current_style,
        }

    def _log_reset(self) -> None:
        logger.info(
            f"Reset Sphinx OddOneOutPoly: answer={self._oracle_answer}, "
            f"num_points={self._num_points}, style={self._current_style}"
        )

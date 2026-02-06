"""Transform Result Identify environments with procedural generation."""

from __future__ import annotations

from abc import abstractmethod
from textwrap import dedent
from typing import Any

from PIL import Image

from gym_v import Env, Observation, get_logger
from gym_v.envs.single_turn.perception.sphinx_utils import (
    TRANSFORMS,
    apply_transform,
    compose_8_options,
    generate_random_grid,
    generate_random_polygon,
)

logger = get_logger()

TRANSFORM_DESCRIPTIONS = {
    "identity": "no transformation",
    "rot90_cw": "rotate 90° clockwise",
    "rot180": "rotate 180°",
    "rot90_ccw": "rotate 90° counterclockwise",
    "flip_h": "reflect across a vertical line",
    "flip_v": "reflect across a horizontal line",
    "flip_diag": "reflect across the main diagonal",
    "flip_antidiag": "reflect across the anti-diagonal",
}


class SphinxTransformResultBaseEnv(Env):
    """Base class for Transform Result Identify tasks.

    Given an original shape and a transformation description, identify which
    of 8 options shows the correct transformation result.
    """

    def __init__(
        self,
        option_size: int = 280,
        padding: int = 20,
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._option_size = option_size
        self._padding = padding
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._original: Image.Image | None = None
        self._correct_transform: str | None = None
        self._correct_idx: int | None = None
        self._composed_image: Image.Image | None = None
        self._oracle_answer: str | None = None

    @property
    def description(self) -> str:
        """Return task description for Transform Result Identify."""
        return dedent("""
            Look at the original shape at the top of the image.
            A geometric transformation has been applied to create one of the 8 options below.

            Your task: Identify which option (a)-(h) shows the correct transformation result.

            The transformation could be:
            - Rotation: 90° clockwise, 180°, or 90° counterclockwise
            - Reflection: across horizontal, vertical, main diagonal, or anti-diagonal axis
            - Identity: no change

            Answer format: A single letter in parentheses, e.g., (a), (b), ..., (h)
        """).strip()

    def _build_problem_text(self) -> str:
        """Build the problem text based on the correct transformation."""
        desc = TRANSFORM_DESCRIPTIONS[self._correct_transform]
        return (
            f"After performing {desc} on the top figure, "
            f"which option (a)–(h) shows the correct outcome?"
        )

    @abstractmethod
    def _generate_original(self) -> Image.Image:
        """Generate the original shape image.

        Returns:
            PIL Image of the original shape
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

    def _generate_options(self) -> tuple[list[Image.Image], int]:
        """Generate 8 option images with all transformations.

        Returns:
            Tuple of (list of 8 option images, index of correct answer)
        """
        transformed = {t: apply_transform(self._original, t) for t in TRANSFORMS}

        transform_order = list(TRANSFORMS)
        self.np_random.shuffle(transform_order)

        options = [transformed[t] for t in transform_order]
        correct_idx = transform_order.index(self._correct_transform)

        return options, correct_idx

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        super().reset(seed=seed)

        self._original = self._generate_original()

        transform_idx = int(self.np_random.integers(0, len(TRANSFORMS)))
        self._correct_transform = TRANSFORMS[transform_idx]

        options_list, self._correct_idx = self._generate_options()

        self._composed_image = compose_8_options(
            self._original,
            options_list,
            self._correct_idx,
            option_size=self._option_size,
            padding=self._padding,
        )

        labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]
        self._oracle_answer = labels[self._correct_idx]

        self._log_reset()

        obs = Observation(
            image=self.render(),
            text=None,
            metadata={
                "text_prompt": None,
                "state_text": None,
                "transform": self._correct_transform,
                **self._get_metadata(),
            },
        )
        info = {
            "oracle_answer": self._oracle_answer,
            "transform": self._correct_transform,
            **self._get_metadata(),
        }
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
        """Evaluate the action against the correct answer."""
        agent_id = next(iter(self._agent_ids))
        single_action = action[agent_id]

        action_clean = single_action.strip().lower()
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
                "text_prompt": None,
                "state_text": None,
                "transform": self._correct_transform,
                "user_answer": single_action,
                "correct": correct,
            },
        )
        info = {
            "oracle_answer": self._oracle_answer,
            "transform": self._correct_transform,
            "correct": correct,
        }

        terminated = True
        truncated = False

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

    def render(self) -> Image.Image:
        """Return the composed image with 8 options."""
        return self._composed_image


class TransformResultEnv(SphinxTransformResultBaseEnv):
    # Meta: source=Sphinx, category=cognition, turn=single
    # Overrides: visual_complexity=simple_geometry
    """Transform Result Identify task with colored grid patterns.

    Args:
        grid_size: Size of the grid (grid_size x grid_size), controls difficulty
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
        option_size: int = 280,
        padding: int = 20,
        **kwargs: Any,
    ):
        super().__init__(option_size=option_size, padding=padding, **kwargs)
        self._grid_size = grid_size
        self._num_colors = num_colors
        self._cell_size = cell_size

    def _generate_original(self) -> Image.Image:
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
            f"Reset Sphinx TransformResult: transform={self._correct_transform}, "
            f"answer={self._oracle_answer}, grid_size={self._grid_size}"
        )


class TransformResultPolyEnv(SphinxTransformResultBaseEnv):
    # Meta: source=Sphinx, category=cognition, turn=single
    # Overrides: visual_complexity=simple_geometry
    """Transform Result Identify task with polygon shapes (original Sphinx style).

    Args:
        img_size: Size of each shape image in pixels
        num_points: Number of vertices in the polygon (controls complexity)
        line_width: Width of polygon lines
        grid_divisions: Number of grid divisions in background
        option_size: Size of each option image in the composed output
        padding: Padding between elements in the composed image
        style: Visual style ('outline', 'filled', 'nested', 'striped',
               'gradient', '3d', 'composite', 'pixelated'),
               or 'random' for random selection each reset
        difficulty: Difficulty level 1-8 (maps to styles in order).
    """

    def __init__(
        self,
        img_size: int = 300,
        num_points: int = 8,
        line_width: int = 3,
        grid_divisions: int = 8,
        option_size: int = 280,
        padding: int = 20,
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

    def _generate_original(self) -> Image.Image:
        original = generate_random_polygon(
            self.np_random,
            img_size=self._img_size,
            num_points=self._num_points,
            line_width=self._line_width,
            grid_lines=True,
            grid_divisions=self._grid_divisions,
            style=self._style,
        )

        self._current_style = self._style if self._style else "random"

        return original

    def _get_metadata(self) -> dict[str, Any]:
        return {
            "num_points": self._num_points,
            "style": self._current_style,
        }

    def _log_reset(self) -> None:
        logger.info(
            f"Reset Sphinx TransformResultPoly: transform={self._correct_transform}, "
            f"answer={self._oracle_answer}, num_points={self._num_points}, "
            f"style={self._current_style}"
        )

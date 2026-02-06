"""Sequence Completion environments with procedural generation."""

from __future__ import annotations

from abc import abstractmethod
from textwrap import dedent
from typing import Any

from PIL import Image

from gym_v import Env, Observation, get_logger
from gym_v.envs.single_turn.perception.sphinx_utils import (
    POLY_STYLES,
    SEQUENCE_PATTERNS,
    TRANSFORMS,
    _images_are_similar,
    apply_transform,
    compose_sequence_completion_image,
    generate_random_grid,
    generate_random_polygon,
)

logger = get_logger()


class SphinxSequenceCompletionBaseEnv(Env):
    """Base class for Sequence Completion tasks.

    Display a sequence of shapes following a transformation pattern,
    and ask which option completes the sequence.
    """

    def __init__(
        self,
        sequence_length: int = 4,
        pattern: str | None = None,
        option_size: int = 150,
        padding: int = 10,
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._sequence_length = sequence_length
        self._pattern = pattern
        self._option_size = option_size
        self._padding = padding
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._composed_image: Image.Image | None = None
        self._oracle_answer: str | None = None
        self._correct_idx: int | None = None
        self._current_pattern: str | None = None

    @property
    def description(self) -> str:
        """Return task description for Sequence Completion."""
        return dedent("""
            Look at the sequence of shapes in the top row.
            Each shape follows a pattern of geometric transformations.

            Your task: Identify which option (a)-(h) should come next in the sequence.

            Possible patterns include:
            - Rotation: shapes rotate by a fixed angle each step (90°, 180°)
            - Reflection: shapes alternate between original and flipped versions

            Answer format: A single letter in parentheses, e.g., (a), (b), ..., (h)
        """).strip()

    @abstractmethod
    def _generate_base_shape(self) -> Image.Image:
        """Generate the base shape for the sequence.

        Returns:
            PIL Image of the base shape
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
        Useful for Poly envs to select a consistent style.
        """
        pass

    def _select_pattern(self) -> tuple[str, list[str]]:
        """Select a pattern and return (pattern_name, transform_sequence)."""
        if self._pattern is not None and self._pattern in SEQUENCE_PATTERNS:
            return self._pattern, SEQUENCE_PATTERNS[self._pattern]

        # Random selection
        pattern_names = list(SEQUENCE_PATTERNS.keys())
        pattern_name = pattern_names[
            int(self.np_random.integers(0, len(pattern_names)))
        ]
        return pattern_name, SEQUENCE_PATTERNS[pattern_name]

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        super().reset(seed=seed)
        self._prepare_reset()

        # Generate base shape
        base_shape = self._generate_base_shape()

        # Select pattern
        self._current_pattern, transform_sequence = self._select_pattern()

        # Generate sequence items (shown sequence + correct next)
        sequence_items = []
        for i in range(self._sequence_length + 1):
            transform_idx = i % len(transform_sequence)
            transform = transform_sequence[transform_idx]
            sequence_items.append(apply_transform(base_shape, transform))

        # Split into shown sequence and correct answer
        shown_sequence = sequence_items[:-1]
        correct_next = sequence_items[-1]

        # Generate distractors (wrong transforms)
        distractors = []
        all_transforms = list(TRANSFORMS)
        self.np_random.shuffle(all_transforms)

        for t in all_transforms:
            if len(distractors) >= 7:
                break
            transformed = apply_transform(base_shape, t)
            # Skip if same as correct answer
            if not _images_are_similar(transformed, correct_next):
                distractors.append(transformed)

        # Fill remaining with double transforms if needed
        attempt = 0
        while len(distractors) < 7 and attempt < 20:
            t1 = all_transforms[attempt % len(all_transforms)]
            t2 = all_transforms[(attempt + 1) % len(all_transforms)]
            double_transformed = apply_transform(apply_transform(base_shape, t1), t2)
            if not any(
                _images_are_similar(double_transformed, d) for d in distractors
            ) and not _images_are_similar(double_transformed, correct_next):
                distractors.append(double_transformed)
            attempt += 1

        # If still not enough, just use transforms (may have some similar ones)
        while len(distractors) < 7:
            t = all_transforms[len(distractors) % len(all_transforms)]
            distractors.append(apply_transform(base_shape, t))

        # Combine and shuffle options
        all_options = [correct_next] + distractors[:7]
        indices = list(range(8))
        self.np_random.shuffle(indices)
        shuffled_options = [all_options[i] for i in indices]
        self._correct_idx = indices.index(0)

        # Compose image
        self._composed_image = compose_sequence_completion_image(
            shown_sequence,
            shuffled_options,
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
                "pattern": self._current_pattern,
                "correct_idx": self._correct_idx,
                **self._get_metadata(),
            },
        )
        info = {
            "oracle_answer": self._oracle_answer,
            "pattern": self._current_pattern,
            "correct_idx": self._correct_idx,
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
                "pattern": self._current_pattern,
                "user_answer": single_action,
                "correct": correct,
            },
        )
        info = {
            "oracle_answer": self._oracle_answer,
            "pattern": self._current_pattern,
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
        """Return the composed image with sequence and options."""
        return self._composed_image


class SequenceCompletionEnv(SphinxSequenceCompletionBaseEnv):
    # Meta: source=Sphinx, category=cognition, turn=single
    # Overrides: visual_complexity=simple_geometry
    """Sequence Completion task with colored grid patterns.

    Args:
        grid_size: Size of the grid (grid_size x grid_size)
        num_colors: Number of colors to use in the grid
        cell_size: Pixel size of each cell in the grid
        sequence_length: Number of items shown in sequence
        pattern: Specific pattern name or None for random
        option_size: Size of each option image in pixels
        padding: Padding between elements in the composed image
    """

    def __init__(
        self,
        grid_size: int = 4,
        num_colors: int = 3,
        cell_size: int = 35,
        sequence_length: int = 4,
        pattern: str | None = None,
        option_size: int = 150,
        padding: int = 10,
        **kwargs: Any,
    ):
        super().__init__(
            sequence_length=sequence_length,
            pattern=pattern,
            option_size=option_size,
            padding=padding,
            **kwargs,
        )
        self._grid_size = grid_size
        self._num_colors = num_colors
        self._cell_size = cell_size

    def _generate_base_shape(self) -> Image.Image:
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
            "sequence_length": self._sequence_length,
        }

    def _log_reset(self) -> None:
        logger.info(
            f"Reset Sphinx SequenceCompletion: pattern={self._current_pattern}, "
            f"answer={self._oracle_answer}, grid_size={self._grid_size}"
        )


class SequenceCompletionPolyEnv(SphinxSequenceCompletionBaseEnv):
    # Meta: source=Sphinx, category=cognition, turn=single
    # Overrides: visual_complexity=simple_geometry
    """Sequence Completion task with polygon shapes.

    Args:
        img_size: Size of each shape image in pixels
        num_points: Number of vertices in the polygon
        line_width: Width of polygon lines
        grid_divisions: Number of grid divisions in background
        sequence_length: Number of items shown in sequence
        pattern: Specific pattern name or None for random
        option_size: Size of each option image in the composed output
        padding: Padding between elements in the composed image
        style: Visual style ('outline', 'filled', 'nested', 'striped',
               'gradient', '3d', 'composite', 'pixelated'),
               or None for random selection each reset
    """

    def __init__(
        self,
        img_size: int = 200,
        num_points: int = 6,
        line_width: int = 3,
        grid_divisions: int = 8,
        sequence_length: int = 4,
        pattern: str | None = None,
        option_size: int = 150,
        padding: int = 10,
        style: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            sequence_length=sequence_length,
            pattern=pattern,
            option_size=option_size,
            padding=padding,
            **kwargs,
        )
        self._img_size = img_size
        self._num_points = num_points
        self._line_width = line_width
        self._grid_divisions = grid_divisions
        self._style = style
        self._current_style: str | None = None

    def _prepare_reset(self) -> None:
        """Select a consistent style for this reset."""
        if self._style is not None:
            self._current_style = self._style
        else:
            # Random selection
            self._current_style = POLY_STYLES[
                int(self.np_random.integers(0, len(POLY_STYLES)))
            ]

    def _generate_base_shape(self) -> Image.Image:
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
            "sequence_length": self._sequence_length,
        }

    def _log_reset(self) -> None:
        logger.info(
            f"Reset Sphinx SequenceCompletionPoly: pattern={self._current_pattern}, "
            f"answer={self._oracle_answer}, num_points={self._num_points}, "
            f"style={self._current_style}"
        )

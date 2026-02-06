"""Symmetry Fill environments with procedural generation."""

from __future__ import annotations

from abc import abstractmethod
from textwrap import dedent
from typing import Any

from PIL import Image

from gym_v import Env, Observation, get_logger
from gym_v.envs.single_turn.perception.sphinx_utils import (
    compose_symmetry_fill_8_options,
    generate_extra_distractors,
    generate_symmetric_2x2_grid,
    generate_symmetric_2x2_icons,
)

logger = get_logger()


class SphinxSymmetryFillBaseEnv(Env):
    """Base class for Symmetry Fill tasks.

    Given a 2x2 grid with one missing cell, identify which of 8 options
    completes the grid to satisfy vertical + horizontal mirror symmetry.
    """

    def __init__(
        self,
        option_size: int = 200,
        padding: int = 15,
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._option_size = option_size
        self._padding = padding
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._cells: list[Image.Image] | None = None
        self._hidden_idx: int | None = None
        self._question: Image.Image | None = None
        self._correct_idx: int | None = None
        self._composed_image: Image.Image | None = None
        self._oracle_answer: str | None = None

    @property
    def description(self) -> str:
        """Return task description for Symmetry Fill."""
        return dedent("""
            Look at the 2x2 grid on the left side of the image.
            One cell is blank (shown in black).

            Your task: Identify which option (a)-(h) should fill the blank cell
            so that the completed grid exhibits vertical + horizontal mirror symmetry.

            In a grid with V+H symmetry:
            - Left and right halves are horizontal mirrors
            - Top and bottom halves are vertical mirrors

            Answer format: A single letter in parentheses, e.g., (a), (b), ..., (h)
        """).strip()

    def _compose_question_grid(
        self,
        cells: list[Image.Image],
        hidden_idx: int,
        gap: int = 4,
    ) -> Image.Image:
        """Compose a 2x2 question grid with one cell hidden (black).

        Args:
            cells: List of 4 cell images [TL, TR, BL, BR]
            hidden_idx: Index of the cell to hide (0-3)
            gap: Gap between cells in pixels

        Returns:
            Composed 2x2 grid image with one black cell
        """
        cell_w, cell_h = cells[0].size

        total_w = 2 * cell_w + gap
        total_h = 2 * cell_h + gap
        canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))

        positions = [
            (0, 0),
            (cell_w + gap, 0),
            (0, cell_h + gap),
            (cell_w + gap, cell_h + gap),
        ]

        for i, (cell, pos) in enumerate(zip(cells, positions, strict=True)):
            if i == hidden_idx:
                black_cell = Image.new("RGB", (cell_w, cell_h), (0, 0, 0))
                canvas.paste(black_cell, pos)
            else:
                canvas.paste(cell, pos)

        return canvas

    @abstractmethod
    def _generate_cells(self) -> tuple[list[Image.Image], int]:
        """Generate the 2x2 grid cells with symmetry.

        Returns:
            Tuple of (list of 4 cell images, hidden cell index)
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

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        super().reset(seed=seed)

        self._cells, self._hidden_idx = self._generate_cells()

        correct_cell = self._cells[self._hidden_idx]
        self._question = self._compose_question_grid(self._cells, self._hidden_idx)

        other_cells = [self._cells[i] for i in range(4) if i != self._hidden_idx]

        extra_distractors = generate_extra_distractors(
            correct_cell,
            self._cells,
            self.np_random,
            num_extra=4,
        )

        all_options = [correct_cell] + other_cells + extra_distractors

        indices = list(range(8))
        self.np_random.shuffle(indices)

        shuffled_options = [all_options[i] for i in indices]
        self._correct_idx = indices.index(0)

        self._composed_image = compose_symmetry_fill_8_options(
            self._question,
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
                "hidden_idx": self._hidden_idx,
                **self._get_metadata(),
            },
        )
        info = {
            "oracle_answer": self._oracle_answer,
            "hidden_idx": self._hidden_idx,
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
                "user_answer": single_action,
                "correct": correct,
            },
        )
        info = {
            "oracle_answer": self._oracle_answer,
            "hidden_idx": self._hidden_idx,
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


class SymmetryFillEnv(SphinxSymmetryFillBaseEnv):
    # Meta: source=Sphinx, category=cognition, turn=single
    # Overrides: visual_complexity=simple_geometry
    """Symmetry Fill task with colored grid patterns.

    Args:
        cell_grid_size: Grid size within each cell, controls difficulty
        num_colors: Number of colors to use in the grids
        cell_size: Pixel size of each cell
        option_size: Size of each option image in pixels
        padding: Padding between elements in the composed image
    """

    def __init__(
        self,
        cell_grid_size: int = 4,
        num_colors: int = 3,
        cell_size: int = 100,
        option_size: int = 200,
        padding: int = 15,
        **kwargs: Any,
    ):
        super().__init__(option_size=option_size, padding=padding, **kwargs)
        self._cell_grid_size = cell_grid_size
        self._num_colors = num_colors
        self._cell_size = cell_size

    def _generate_cells(self) -> tuple[list[Image.Image], int]:
        return generate_symmetric_2x2_grid(
            self.np_random,
            cell_grid_size=self._cell_grid_size,
            num_colors=self._num_colors,
            cell_size=self._cell_size,
        )

    def _get_metadata(self) -> dict[str, Any]:
        return {
            "cell_grid_size": self._cell_grid_size,
            "num_colors": self._num_colors,
        }

    def _log_reset(self) -> None:
        logger.info(
            f"Reset Sphinx SymmetryFill: hidden_idx={self._hidden_idx}, "
            f"answer={self._oracle_answer}, cell_grid_size={self._cell_grid_size}"
        )


class SymmetryFillPolyEnv(SphinxSymmetryFillBaseEnv):
    # Meta: source=Sphinx, category=cognition, turn=single
    # Overrides: visual_complexity=simple_geometry
    """Symmetry Fill task with icon shapes (original Sphinx style).

    Args:
        cell_size: Pixel size of each cell/icon
        line_width: Width of icon lines
        option_size: Size of each option image in pixels
        padding: Padding between elements in the composed image
        style: Visual style ('simple', 'colored', 'nested', 'complex'),
               or None for random selection each reset
    """

    def __init__(
        self,
        cell_size: int = 200,
        line_width: int = 4,
        option_size: int = 200,
        padding: int = 15,
        style: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(option_size=option_size, padding=padding, **kwargs)
        self._cell_size = cell_size
        self._line_width = line_width
        self._style = style
        self._current_style: str | None = None

    def _generate_cells(self) -> tuple[list[Image.Image], int]:
        cells, hidden_idx = generate_symmetric_2x2_icons(
            self.np_random,
            cell_size=self._cell_size,
            line_width=self._line_width,
            style=self._style,
        )

        self._current_style = self._style if self._style else "random"

        return cells, hidden_idx

    def _get_metadata(self) -> dict[str, Any]:
        return {"style": self._current_style}

    def _log_reset(self) -> None:
        logger.info(
            f"Reset Sphinx SymmetryFillPoly: hidden_idx={self._hidden_idx}, "
            f"answer={self._oracle_answer}, style={self._current_style}"
        )

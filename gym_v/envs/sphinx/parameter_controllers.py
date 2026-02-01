"""Parameter controllers for Sphinx environments."""

from __future__ import annotations

from typing import Any

from gym_v.utils.parameter_controller import ParameterController


class SphinxDefaultController(ParameterController):
    """Default controller for Sphinx environments."""

    def _initialize_parameters(self) -> None:
        self._params: dict[str, Any] = {}

    def update(self) -> None:
        pass

    def get_parameters(self) -> dict[str, Any]:
        return dict(self._params)


class SphinxGridController(ParameterController):
    """Controller for grid-based Sphinx tasks."""

    def __init__(
        self,
        initial_difficulty: int = 0,
        min_grid: int = 3,
        max_grid: int = 7,
        min_colors: int = 2,
        max_colors: int = 5,
    ) -> None:
        self._min_grid = min_grid
        self._max_grid = max_grid
        self._min_colors = min_colors
        self._max_colors = max_colors
        super().__init__(initial_difficulty)

    def _initialize_parameters(self) -> None:
        self._grid_size = self._min_grid
        self._num_colors = self._min_colors

    def update(self) -> None:
        self._grid_size = min(self._max_grid, self._grid_size + 1)
        if self._grid_size >= self._min_grid + 2:
            self._num_colors = min(self._max_colors, self._num_colors + 1)

    def get_parameters(self) -> dict[str, Any]:
        return {"grid_size": self._grid_size, "num_colors": self._num_colors}


class SphinxSymmetryFillGridController(ParameterController):
    """Controller for SymmetryFill grid cell size."""

    def __init__(
        self,
        initial_difficulty: int = 0,
        min_cell_grid: int = 3,
        max_cell_grid: int = 7,
        min_colors: int = 2,
        max_colors: int = 5,
    ) -> None:
        self._min_cell_grid = min_cell_grid
        self._max_cell_grid = max_cell_grid
        self._min_colors = min_colors
        self._max_colors = max_colors
        super().__init__(initial_difficulty)

    def _initialize_parameters(self) -> None:
        self._cell_grid_size = self._min_cell_grid
        self._num_colors = self._min_colors

    def update(self) -> None:
        self._cell_grid_size = min(self._max_cell_grid, self._cell_grid_size + 1)
        if self._cell_grid_size >= self._min_cell_grid + 2:
            self._num_colors = min(self._max_colors, self._num_colors + 1)

    def get_parameters(self) -> dict[str, Any]:
        return {
            "cell_grid_size": self._cell_grid_size,
            "num_colors": self._num_colors,
        }


class SphinxSequenceController(ParameterController):
    """Controller for sequence length on Sphinx tasks."""

    def __init__(
        self, initial_difficulty: int = 0, min_len: int = 3, max_len: int = 6
    ) -> None:
        self._min_len = min_len
        self._max_len = max_len
        super().__init__(initial_difficulty)

    def _initialize_parameters(self) -> None:
        self._sequence_length = self._min_len

    def update(self) -> None:
        self._sequence_length = min(self._max_len, self._sequence_length + 1)

    def get_parameters(self) -> dict[str, Any]:
        return {"sequence_length": self._sequence_length}


class SphinxSequenceGridController(ParameterController):
    """Controller for sequence completion with grids."""

    def __init__(
        self,
        initial_difficulty: int = 0,
        min_grid: int = 3,
        max_grid: int = 7,
        min_colors: int = 2,
        max_colors: int = 5,
        min_len: int = 3,
        max_len: int = 6,
    ) -> None:
        self._min_grid = min_grid
        self._max_grid = max_grid
        self._min_colors = min_colors
        self._max_colors = max_colors
        self._min_len = min_len
        self._max_len = max_len
        super().__init__(initial_difficulty)

    def _initialize_parameters(self) -> None:
        self._grid_size = self._min_grid
        self._num_colors = self._min_colors
        self._sequence_length = self._min_len

    def update(self) -> None:
        self._grid_size = min(self._max_grid, self._grid_size + 1)
        self._num_colors = min(self._max_colors, self._num_colors + 1)
        self._sequence_length = min(self._max_len, self._sequence_length + 1)

    def get_parameters(self) -> dict[str, Any]:
        return {
            "grid_size": self._grid_size,
            "num_colors": self._num_colors,
            "sequence_length": self._sequence_length,
        }


class SphinxSequencePolyController(ParameterController):
    """Controller for sequence completion with polygons."""

    def __init__(
        self,
        initial_difficulty: int = 0,
        min_points: int = 5,
        max_points: int = 10,
        min_divisions: int = 6,
        max_divisions: int = 12,
        min_len: int = 3,
        max_len: int = 6,
    ) -> None:
        self._min_points = min_points
        self._max_points = max_points
        self._min_divisions = min_divisions
        self._max_divisions = max_divisions
        self._min_len = min_len
        self._max_len = max_len
        super().__init__(initial_difficulty)

    def _initialize_parameters(self) -> None:
        self._num_points = self._min_points
        self._grid_divisions = self._min_divisions
        self._sequence_length = self._min_len

    def update(self) -> None:
        self._num_points = min(self._max_points, self._num_points + 1)
        self._grid_divisions = min(self._max_divisions, self._grid_divisions + 1)
        self._sequence_length = min(self._max_len, self._sequence_length + 1)

    def get_parameters(self) -> dict[str, Any]:
        return {
            "num_points": self._num_points,
            "grid_divisions": self._grid_divisions,
            "sequence_length": self._sequence_length,
        }


class SphinxPolyController(ParameterController):
    """Controller for polygon-based Sphinx tasks."""

    def __init__(
        self,
        initial_difficulty: int = 0,
        min_points: int = 5,
        max_points: int = 10,
        min_divisions: int = 6,
        max_divisions: int = 12,
    ) -> None:
        self._min_points = min_points
        self._max_points = max_points
        self._min_divisions = min_divisions
        self._max_divisions = max_divisions
        super().__init__(initial_difficulty)

    def _initialize_parameters(self) -> None:
        self._num_points = self._min_points
        self._grid_divisions = self._min_divisions

    def update(self) -> None:
        self._num_points = min(self._max_points, self._num_points + 1)
        self._grid_divisions = min(self._max_divisions, self._grid_divisions + 1)

    def get_parameters(self) -> dict[str, Any]:
        return {
            "num_points": self._num_points,
            "grid_divisions": self._grid_divisions,
        }


class SphinxStyleController(ParameterController):
    """Controller for style-based Sphinx tasks."""

    def __init__(
        self, initial_difficulty: int = 0, styles: list[str] | None = None
    ) -> None:
        self._styles = styles or ["simple", "colored", "nested", "complex"]
        super().__init__(initial_difficulty)

    def _initialize_parameters(self) -> None:
        self._style = self._styles[0]

    def update(self) -> None:
        idx = min(len(self._styles) - 1, self.current_difficulty + 1)
        self._style = self._styles[idx]

    def get_parameters(self) -> dict[str, Any]:
        return {"style": self._style}


def get_controller_for_env(
    env_class_name: str, difficulty: int = 0
) -> ParameterController:
    """Get the appropriate controller for a Sphinx environment."""
    controllers = {
        "SphinxOddOneOutEnv": lambda d: SphinxGridController(d),
        "SphinxOddOneOutPolyEnv": lambda d: SphinxPolyController(d),
        "SphinxSequenceCompletionEnv": lambda d: SphinxSequenceGridController(d),
        "SphinxSequenceCompletionPolyEnv": lambda d: SphinxSequencePolyController(d),
        "SphinxSymmetryFillEnv": lambda d: SphinxSymmetryFillGridController(d),
        "SphinxSymmetryFillPolyEnv": lambda d: SphinxStyleController(d),
        "SphinxTransformResultEnv": lambda d: SphinxGridController(d),
        "SphinxTransformResultPolyEnv": lambda d: SphinxPolyController(d),
    }

    factory = controllers.get(env_class_name)
    if factory:
        return factory(difficulty)
    return SphinxDefaultController(difficulty)

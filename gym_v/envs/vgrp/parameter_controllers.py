"""Parameter controllers for VGRP (Visual Grid Reasoning Puzzles) environments.

VGRP environments typically have grid_size and other puzzle-specific parameters.
"""

from __future__ import annotations

from typing import Any

from gym_v.utils.parameter_controller import ParameterController


class VGRPDefaultController(ParameterController):
    """Default controller for VGRP environments."""

    def _initialize_parameters(self) -> None:
        self._grid_size = 6

    def update(self) -> None:
        self._grid_size = min(12, self._grid_size + 1)

    def get_parameters(self) -> dict[str, Any]:
        return {"grid_size": self._grid_size}


class VGRPBattleshipsController(ParameterController):
    """Controller for Battleships puzzle."""

    def _initialize_parameters(self) -> None:
        self._grid_size = 6
        self._num_ships = 3

    def update(self) -> None:
        d = self.current_difficulty + 1
        if d <= 3:
            self._grid_size = 6 + d
        else:
            self._grid_size = min(10, self._grid_size + 1)
            self._num_ships = min(5, self._num_ships + 1)

    def get_parameters(self) -> dict[str, Any]:
        return {"grid_size": self._grid_size, "num_ships": self._num_ships}


class VGRPSudokuController(ParameterController):
    """Controller for Sudoku-like VGRP puzzles (Futoshiki, etc.)."""

    def _initialize_parameters(self) -> None:
        self._size = 4

    def update(self) -> None:
        self._size = min(9, self._size + 1)

    def get_parameters(self) -> dict[str, Any]:
        return {"size": self._size}


def get_controller_for_env(
    env_class_name: str, difficulty: int = 0
) -> ParameterController:
    """Get the appropriate controller for a VGRP environment."""
    controllers = {
        "VGRPBattleshipsEnv": lambda d: VGRPBattleshipsController(d),
        "VGRPFutoshikiEnv": lambda d: VGRPSudokuController(d),
        "VGRPBinairoEnv": lambda d: VGRPDefaultController(d),
        "VGRPHitoriEnv": lambda d: VGRPDefaultController(d),
        "VGRPRenzokuEnv": lambda d: VGRPSudokuController(d),
        "VGRPStarbattleEnv": lambda d: VGRPDefaultController(d),
        "VGRPThermometersEnv": lambda d: VGRPDefaultController(d),
    }

    factory = controllers.get(env_class_name)
    if factory:
        return factory(difficulty)
    return VGRPDefaultController(difficulty)

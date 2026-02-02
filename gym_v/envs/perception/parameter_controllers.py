"""Parameter controllers for Perception environments.

Perception environments test visual understanding capabilities.
Controllers map difficulty to parameters where supported.
"""

from __future__ import annotations

from typing import Any

from gym_v.utils.parameter_controller import ParameterController


class PerceptionDefaultController(ParameterController):
    """Default controller for Perception environments."""

    def _initialize_parameters(self) -> None:
        self._params: dict[str, Any] = {}

    def update(self) -> None:
        pass

    def get_parameters(self) -> dict[str, Any]:
        return dict(self._params)


class PerceptionChartToTableController(ParameterController):
    """Controller for ChartToTable (category count)."""

    def _initialize_parameters(self) -> None:
        self._max_categories = 5

    def update(self) -> None:
        self._max_categories = min(12, self._max_categories + 1)

    def get_parameters(self) -> dict[str, Any]:
        return {"max_categories": self._max_categories}


class PerceptionGraphController(ParameterController):
    """Controller for graph-based perception tasks."""

    def __init__(
        self,
        initial_difficulty: int = 0,
        min_nodes: int = 4,
        max_nodes: int = 10,
        edge_probability: float | None = None,
        edge_step: float = 0.05,
    ) -> None:
        self._min_nodes = min_nodes
        self._max_nodes = max_nodes
        self._edge_probability = edge_probability
        self._edge_step = edge_step
        super().__init__(initial_difficulty)

    def _initialize_parameters(self) -> None:
        self._min_nodes_val = self._min_nodes
        self._max_nodes_val = self._min_nodes + 1
        self._edge_prob_val = self._edge_probability

    def update(self) -> None:
        self._min_nodes_val = min(self._max_nodes - 1, self._min_nodes_val + 1)
        self._max_nodes_val = min(self._max_nodes, self._max_nodes_val + 1)
        if self._edge_prob_val is not None:
            self._edge_prob_val = min(0.9, self._edge_prob_val + self._edge_step)

    def get_parameters(self) -> dict[str, Any]:
        params = {
            "min_nodes": self._min_nodes_val,
            "max_nodes": self._max_nodes_val,
        }
        if self._edge_prob_val is not None:
            params["edge_probability"] = self._edge_prob_val
        return params


class PerceptionFlowNetworkController(ParameterController):
    """Controller for flow network capacity + nodes."""

    def __init__(
        self,
        initial_difficulty: int = 0,
        min_nodes: int = 5,
        max_nodes: int = 10,
        max_capacity: int = 25,
    ) -> None:
        self._min_nodes = min_nodes
        self._max_nodes = max_nodes
        self._max_capacity = max_capacity
        super().__init__(initial_difficulty)

    def _initialize_parameters(self) -> None:
        self._min_nodes_val = self._min_nodes
        self._max_nodes_val = self._min_nodes + 1
        self._max_capacity_val = 10

    def update(self) -> None:
        self._min_nodes_val = min(self._max_nodes - 1, self._min_nodes_val + 1)
        self._max_nodes_val = min(self._max_nodes, self._max_nodes_val + 1)
        self._max_capacity_val = min(self._max_capacity, self._max_capacity_val + 2)

    def get_parameters(self) -> dict[str, Any]:
        return {
            "min_nodes": self._min_nodes_val,
            "max_nodes": self._max_nodes_val,
            "max_capacity": self._max_capacity_val,
        }


class PerceptionMSTController(ParameterController):
    """Controller for MST (nodes + weights)."""

    def __init__(
        self,
        initial_difficulty: int = 0,
        min_nodes: int = 5,
        max_nodes: int = 10,
        max_weight: int = 30,
    ) -> None:
        self._min_nodes = min_nodes
        self._max_nodes = max_nodes
        self._max_weight = max_weight
        super().__init__(initial_difficulty)

    def _initialize_parameters(self) -> None:
        self._min_nodes_val = self._min_nodes
        self._max_nodes_val = self._min_nodes + 1
        self._max_weight_val = 12

    def update(self) -> None:
        self._min_nodes_val = min(self._max_nodes - 1, self._min_nodes_val + 1)
        self._max_nodes_val = min(self._max_nodes, self._max_nodes_val + 1)
        self._max_weight_val = min(self._max_weight, self._max_weight_val + 2)

    def get_parameters(self) -> dict[str, Any]:
        return {
            "min_nodes": self._min_nodes_val,
            "max_nodes": self._max_nodes_val,
            "max_weight": self._max_weight_val,
        }


class PerceptionVectorFieldController(ParameterController):
    """Controller for vector field grid density."""

    def _initialize_parameters(self) -> None:
        self._grid_density = 9
        self._range = 3.0

    def update(self) -> None:
        self._grid_density = min(25, self._grid_density + 2)
        self._range = min(6.0, self._range + 0.5)

    def get_parameters(self) -> dict[str, Any]:
        return {
            "grid_density": self._grid_density,
            "xy_range": (-self._range, self._range),
        }


class PerceptionRangeController(ParameterController):
    """Controller for range-based perception tasks."""

    def __init__(
        self,
        initial_difficulty: int = 0,
        min_range: float = 3.0,
        max_range: float = 6.0,
        step: float = 0.5,
        range_key: str = "x_range",
    ) -> None:
        self._min_range = min_range
        self._max_range = max_range
        self._step = step
        self._range_key = range_key
        super().__init__(initial_difficulty)

    def _initialize_parameters(self) -> None:
        self._range = self._min_range

    def update(self) -> None:
        self._range = min(self._max_range, self._range + self._step)

    def get_parameters(self) -> dict[str, Any]:
        return {self._range_key: (-self._range, self._range)}


class PerceptionTreeController(ParameterController):
    """Controller for tree traversal (node counts)."""

    def __init__(
        self, initial_difficulty: int = 0, min_nodes: int = 5, max_nodes: int = 15
    ) -> None:
        self._min_nodes = min_nodes
        self._max_nodes = max_nodes
        super().__init__(initial_difficulty)

    def _initialize_parameters(self) -> None:
        self._min_nodes_val = self._min_nodes
        self._max_nodes_val = self._min_nodes + 1

    def update(self) -> None:
        self._min_nodes_val = min(self._max_nodes - 1, self._min_nodes_val + 1)
        self._max_nodes_val = min(self._max_nodes, self._max_nodes_val + 1)

    def get_parameters(self) -> dict[str, Any]:
        return {
            "min_nodes": self._min_nodes_val,
            "max_nodes": self._max_nodes_val,
        }


def get_controller_for_env(
    env_class_name: str, difficulty: int = 0
) -> ParameterController:
    """Get the appropriate controller for a Perception environment."""
    controllers = {
        "PerceptionChartToTableEnv": lambda d: PerceptionChartToTableController(d),
        "PerceptionDAGToTopoOrderEnv": lambda d: PerceptionGraphController(
            d, min_nodes=5, max_nodes=12, edge_probability=0.35
        ),
        "PerceptionGraphToAdjacencyEnv": lambda d: PerceptionGraphController(
            d, min_nodes=4, max_nodes=10, edge_probability=0.4
        ),
        "PerceptionGraphToMSTEnv": lambda d: PerceptionMSTController(
            d, min_nodes=5, max_nodes=10, max_weight=30
        ),
        "PerceptionFlowNetworkEnv": lambda d: PerceptionFlowNetworkController(
            d, min_nodes=5, max_nodes=10, max_capacity=25
        ),
        "PerceptionVectorFieldEnv": lambda d: PerceptionVectorFieldController(d),
        "PerceptionFunctionGraphEnv": lambda d: PerceptionRangeController(
            d, min_range=3.0, max_range=8.0, step=0.5, range_key="x_range"
        ),
        "PerceptionContourPlotEnv": lambda d: PerceptionRangeController(
            d, min_range=3.0, max_range=6.0, step=0.5, range_key="xy_range"
        ),
        "PerceptionTreeToTraversalEnv": lambda d: PerceptionTreeController(
            d, min_nodes=5, max_nodes=15
        ),
        # Parametric/Polar plots use expression generation without clear difficulty param
        # Using default controller since complexity is in the expression, not parameters
        "PerceptionParametricCurveEnv": lambda d: PerceptionDefaultController(d),
        "PerceptionPolarPlotEnv": lambda d: PerceptionDefaultController(d),
    }

    factory = controllers.get(env_class_name)
    if factory:
        return factory(difficulty)
    return PerceptionDefaultController(difficulty)

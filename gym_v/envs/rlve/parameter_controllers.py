"""Parameter controllers for RLVE environments.

Each controller maps difficulty levels to environment-specific parameters.
Difficulty 0 is the easiest setting, with parameters increasing progressively.
"""

from __future__ import annotations

from typing import Any

from gym_v.utils.parameter_controller import ParameterController

# Parameter name mappings for common difficulty-controlling parameters
PARAM_MAPPINGS = {
    # Grid size parameters
    "max_n_m": ["max_n_m"],
    "max_m_n": ["max_m_n"],
    "max_r_c": ["max_r_c"],
    "max_n": ["max_n"],
    "N": ["N"],
    "n": ["n"],
    # Sparsity/density
    "sparsity": ["sparsity"],
    "edge_density": ["edge_density"],
    # Steps
    "steps": ["steps"],
}


class RLVENumbrixController(ParameterController):
    """Controller for Numbrix puzzle difficulty.

    Difficulty progression:
    - 0-3: Increase grid size from 2x2 to 5x5
    - 4-8: Increase sparsity from 0.3 to 0.5 at grid size 5-6
    - 9+: Continue increasing grid size and sparsity
    """

    def _initialize_parameters(self) -> None:
        self.max_n_m = 2
        self.sparsity = 0.3

    def update(self) -> None:
        d = self.current_difficulty + 1
        if d <= 4:
            self.max_n_m = 2 + d
        elif d <= 8:
            self.max_n_m = 6
            self.sparsity = min(0.5, 0.3 + (d - 4) * 0.05)
        else:
            self.max_n_m = min(10, 6 + (d - 8))
            self.sparsity = min(0.7, 0.5 + (d - 8) * 0.02)

    def get_parameters(self) -> dict[str, Any]:
        return {"max_n_m": self.max_n_m, "sparsity": self.sparsity}


class RLVEGridSizeController(ParameterController):
    """Generic controller for grid-based environments.

    Used by environments with max_n_m parameter only.
    """

    def __init__(
        self,
        initial_difficulty: int = 0,
        min_size: int = 2,
        max_size: int = 10,
        param_name: str = "max_n_m",
    ) -> None:
        self._min_size = min_size
        self._max_size = max_size
        self._param_name = param_name
        super().__init__(initial_difficulty)

    def _initialize_parameters(self) -> None:
        self._size = self._min_size

    def update(self) -> None:
        self._size = min(self._max_size, self._size + 1)

    def get_parameters(self) -> dict[str, Any]:
        return {self._param_name: self._size}


class RLVEGridSparsityController(ParameterController):
    """Controller for grid environments with size and sparsity.

    Used by: Binario, Campsite, MagicSquare, etc.
    """

    def __init__(
        self,
        initial_difficulty: int = 0,
        min_size: int = 2,
        max_size: int = 8,
        min_sparsity: float = 0.3,
        max_sparsity: float = 0.7,
        size_param: str = "max_n_m",
    ) -> None:
        self._min_size = min_size
        self._max_size = max_size
        self._min_sparsity = min_sparsity
        self._max_sparsity = max_sparsity
        self._size_param = size_param
        super().__init__(initial_difficulty)

    def _initialize_parameters(self) -> None:
        self._size = self._min_size
        self._sparsity = self._min_sparsity

    def update(self) -> None:
        d = self.current_difficulty + 1
        # First half: increase size
        mid_point = (self._max_size - self._min_size) // 2
        if d <= mid_point:
            self._size = min(self._max_size, self._min_size + d)
        else:
            # Second half: increase sparsity
            self._size = self._max_size
            sparsity_steps = d - mid_point
            sparsity_range = self._max_sparsity - self._min_sparsity
            self._sparsity = min(
                self._max_sparsity,
                self._min_sparsity + sparsity_steps * (sparsity_range / 10),
            )

    def get_parameters(self) -> dict[str, Any]:
        return {self._size_param: self._size, "sparsity": self._sparsity}


class RLVEGraphController(ParameterController):
    """Controller for graph-based environments.

    Controls max_n (vertices) and edge_density.
    Used by: ColoringCounting, GraphIsomorphism, HamiltonianPath, etc.
    """

    def __init__(
        self,
        initial_difficulty: int = 0,
        min_n: int = 4,
        max_n: int = 15,
        min_density: float = 0.3,
        max_density: float = 0.7,
    ) -> None:
        self._min_n = min_n
        self._max_n = max_n
        self._min_density = min_density
        self._max_density = max_density
        super().__init__(initial_difficulty)

    def _initialize_parameters(self) -> None:
        self._n = self._min_n
        self._density = self._min_density

    def update(self) -> None:
        d = self.current_difficulty + 1
        # Increase vertices first, then density
        n_steps = self._max_n - self._min_n
        if d <= n_steps:
            self._n = min(self._max_n, self._min_n + d)
        else:
            self._n = self._max_n
            density_steps = d - n_steps
            density_range = self._max_density - self._min_density
            self._density = min(
                self._max_density,
                self._min_density + density_steps * (density_range / 10),
            )

    def get_parameters(self) -> dict[str, Any]:
        return {"max_n": self._n, "edge_density": self._density}


class RLVETreeController(ParameterController):
    """Controller for tree-based environments.

    Controls max_n (nodes).
    Used by: BinaryTree, TreeCenter, TreeColoring, etc.
    """

    def __init__(
        self,
        initial_difficulty: int = 0,
        min_n: int = 3,
        max_n: int = 20,
    ) -> None:
        self._min_n = min_n
        self._max_n = max_n
        super().__init__(initial_difficulty)

    def _initialize_parameters(self) -> None:
        self._n = self._min_n

    def update(self) -> None:
        self._n = min(self._max_n, self._n + 1)

    def get_parameters(self) -> dict[str, Any]:
        return {"max_n": self._n}


class RLVEPuzzleStepsController(ParameterController):
    """Controller for step-based puzzle environments.

    Controls number of scrambling/transformation steps.
    Used by: EightDigitPuzzle, NinePuzzle, etc.
    """

    def __init__(
        self,
        initial_difficulty: int = 0,
        min_steps: int = 3,
        max_steps: int = 30,
        step_increment: int = 2,
    ) -> None:
        self._min_steps = min_steps
        self._max_steps = max_steps
        self._step_increment = step_increment
        super().__init__(initial_difficulty)

    def _initialize_parameters(self) -> None:
        self._steps = self._min_steps

    def update(self) -> None:
        self._steps = min(self._max_steps, self._steps + self._step_increment)

    def get_parameters(self) -> dict[str, Any]:
        return {"steps": self._steps}


class RLVERangeController(ParameterController):
    """Controller for environments with min_n/max_n range.

    Used by: AdditionTable, MagicSquare ranges, etc.
    """

    def __init__(
        self,
        initial_difficulty: int = 0,
        initial_min: int = 3,
        initial_max: int = 5,
        max_limit: int = 15,
    ) -> None:
        self._initial_min = initial_min
        self._initial_max = initial_max
        self._max_limit = max_limit
        super().__init__(initial_difficulty)

    def _initialize_parameters(self) -> None:
        self._min_n = self._initial_min
        self._max_n = self._initial_max

    def update(self) -> None:
        # Expand the range upward
        self._max_n = min(self._max_limit, self._max_n + 1)

    def get_parameters(self) -> dict[str, Any]:
        return {"min_n": self._min_n, "max_n": self._max_n}


# Environment-specific controller mappings
# Maps environment class names to their controller factory functions


class RLVEDefaultController(ParameterController):
    """Default controller for environments without specific mappings.

    Simply tracks difficulty level without modifying parameters.
    Environments using this controller rely on their default parameter values.
    """

    def _initialize_parameters(self) -> None:
        pass

    def update(self) -> None:
        pass

    def get_parameters(self) -> dict[str, Any]:
        return {}


def get_controller_for_env(
    env_class_name: str, difficulty: int = 0
) -> ParameterController:
    """Get the appropriate controller for an RLVE environment.

    Args:
        env_class_name: Name of the environment class.
        difficulty: Initial difficulty level.

    Returns:
        ParameterController instance. Returns RLVEDefaultController if
        no specific controller is configured.
    """
    controllers = {
        # Grid + Sparsity environments
        "RLVENumbrixEnv": lambda d: RLVENumbrixController(d),
        "RLVEBinarioNoAdjacencyRequirementEnv": lambda d: RLVEGridSparsityController(
            d, min_size=2, max_size=6, size_param="max_n_m"
        ),
        "RLVECampsitePuzzleEnv": lambda d: RLVEGridSparsityController(
            d, min_size=2, max_size=6, size_param="max_n_m"
        ),
        "RLVEMagicSquarePuzzleEnv": lambda d: RLVEGridSparsityController(
            d, min_size=3, max_size=7, size_param="max_n"
        ),
        # Grid size only environments
        "RLVEBlockImageEnv": lambda d: RLVEGridSizeController(
            d, min_size=2, max_size=6, param_name="max_m_n"
        ),
        "RLVECirculatingGridEnv": lambda d: RLVEGridSizeController(
            d, min_size=3, max_size=8, param_name="max_r_c"
        ),
        "RLVEGridBFSEnv": lambda d: RLVEGridSizeController(
            d, min_size=4, max_size=12, param_name="max_n_m"
        ),
        "RLVEGridComponentEnv": lambda d: RLVEGridSizeController(
            d, min_size=4, max_size=12, param_name="max_n_m"
        ),
        "RLVEGridLocalMinimumCountingEnv": lambda d: RLVEGridSizeController(
            d, min_size=3, max_size=10, param_name="max_n_m"
        ),
        "RLVEHitoriPuzzleEnv": lambda d: RLVEGridSizeController(
            d, min_size=4, max_size=8, param_name="max_n_m"
        ),
        "RLVEKloBlocksEnv": lambda d: RLVEGridSizeController(
            d, min_size=3, max_size=6, param_name="max_n_m"
        ),
        "RLVENinePuzzleEnv": lambda d: RLVEGridSizeController(
            d, min_size=2, max_size=4, param_name="max_n_m"
        ),
        "RLVESkyscraperPuzzleEnv": lambda d: RLVEGridSizeController(
            d, min_size=3, max_size=7, param_name="max_n"
        ),
        # Graph environments
        "RLVEColoringCountingEnv": lambda d: RLVEGraphController(d, min_n=4, max_n=12),
        "RLVEGraphContainTreeCountingEnv": lambda d: RLVEGraphController(
            d, min_n=4, max_n=12
        ),
        "RLVEGraphIsomorphismEnv": lambda d: RLVEGraphController(
            d, min_n=4, max_n=12, min_density=0.2, max_density=0.5
        ),
        "RLVEHamiltonianPathEnv": lambda d: RLVEGraphController(d, min_n=4, max_n=12),
        "RLVELongestPathEnv": lambda d: RLVEGraphController(d, min_n=4, max_n=12),
        "RLVEMaximumCliqueEnv": lambda d: RLVEGraphController(d, min_n=4, max_n=12),
        "RLVEMaximumAchromaticNumberEnv": lambda d: RLVEGraphController(
            d, min_n=4, max_n=10
        ),
        "RLVEMaximumWeightMatchingEnv": lambda d: RLVEGraphController(
            d, min_n=4, max_n=12
        ),
        "RLVEMinimumChromaticNumberEnv": lambda d: RLVEGraphController(
            d, min_n=4, max_n=10
        ),
        # Tree environments
        "RLVEBinaryTreeLeafNumExpectationEnv": lambda d: RLVETreeController(
            d, min_n=5, max_n=20
        ),
        "RLVEFbiBinaryTreeEnv": lambda d: RLVETreeController(d, min_n=2, max_n=6),
        "RLVETreeCenterEnv": lambda d: RLVETreeController(d, min_n=5, max_n=20),
        "RLVETreeColoringEnv": lambda d: RLVETreeController(d, min_n=5, max_n=20),
        "RLVETreeAddOneEdgeDiameterEnv": lambda d: RLVETreeController(
            d, min_n=5, max_n=20
        ),
        "RLVETreeChangeOneEdgeDiameterEnv": lambda d: RLVETreeController(
            d, min_n=5, max_n=20
        ),
        "RLVETreeEvenPartitioningEnv": lambda d: RLVETreeController(
            d, min_n=4, max_n=16
        ),
        "RLVEPatrolEnv": lambda d: RLVETreeController(d, min_n=5, max_n=15),
        # Puzzle with steps
        "RLVEEightDigitPuzzleEnv": lambda d: RLVEPuzzleStepsController(
            d, min_steps=3, max_steps=25, step_increment=2
        ),
        # Range-based environments
        "RLVEAdditionTableEnv": lambda d: RLVERangeController(
            d, initial_min=3, initial_max=5, max_limit=15
        ),
        # Simple max_n environments
        "RLVEFaceRightWayEnv": lambda d: RLVEGridSizeController(
            d, min_size=4, max_size=15, param_name="max_n"
        ),
        "RLVECoinSquareGameEnv": lambda d: RLVEGridSizeController(
            d, min_size=4, max_size=15, param_name="max_n"
        ),
        "RLVEConvexHullEnv": lambda d: RLVEGridSizeController(
            d, min_size=5, max_size=20, param_name="N"
        ),
        "RLVECardColoringCountingEnv": lambda d: RLVEGridSizeController(
            d, min_size=4, max_size=12, param_name="N"
        ),
        "RLVEGraMinimaGameEnv": lambda d: RLVEGridSizeController(
            d, min_size=4, max_size=12, param_name="n"
        ),
        # GridSize-based environments (13 additional)
        "RLVEGridParityConstructionEnv": lambda d: RLVEGridSizeController(
            d, min_size=3, max_size=8, param_name="max_n_m"
        ),
        "RLVEMatrixPoolingEnv": lambda d: RLVEGridSizeController(
            d, min_size=3, max_size=10, param_name="max_n_m"
        ),
        "RLVEGridTriangleCountingEnv": lambda d: RLVEGridSizeController(
            d, min_size=3, max_size=10, param_name="max_n_m"
        ),
        "RLVEStoneIntervalsGameEnv": lambda d: RLVEGridSizeController(
            d, min_size=3, max_size=15, param_name="max_n"
        ),
        "RLVEStoneGameEnv": lambda d: RLVEGridSizeController(
            d, min_size=10, max_size=50, param_name="max_sum"
        ),
        "RLVENewNimGameEnv": lambda d: RLVEGridSizeController(
            d, min_size=3, max_size=10, param_name="N"
        ),
        "RLVELargestRectangleAmongPointsEnv": lambda d: RLVEGridSizeController(
            d, min_size=5, max_size=15, param_name="max_n"
        ),
        "RLVESumTriangleAreaEnv": lambda d: RLVEGridSizeController(
            d, min_size=3, max_size=12, param_name="max_n"
        ),
        "RLVEVisibleLineEnv": lambda d: RLVEGridSizeController(
            d, min_size=3, max_size=12, param_name="max_n"
        ),
        "RLVEMonochromeBlockCountingEnv": lambda d: RLVEGridSizeController(
            d, min_size=5, max_size=15, param_name="max_a_b"
        ),
        "RLVEWarehouseConstructionEnv": lambda d: RLVEGridSizeController(
            d, min_size=3, max_size=12, param_name="n"
        ),
        "RLVELargestConvexPolygonEnv": lambda d: RLVEGridSizeController(
            d, min_size=5, max_size=20, param_name="n_points"
        ),
        "RLVESmallestCircleEnv": lambda d: RLVEGridSizeController(
            d, min_size=5, max_size=20, param_name="n_points"
        ),
        # Graph-based environments (3 additional)
        "RLVESpyNetworkEnv": lambda d: RLVEGraphController(
            d, min_n=4, max_n=15, min_density=0.3, max_density=0.7
        ),
        "RLVEMinimumSpanningTreeCountingEnv": lambda d: RLVEGraphController(
            d, min_n=4, max_n=12, min_density=0.3, max_density=0.6
        ),
        "RLVEMinimumWeightedSpanningTreeEnv": lambda d: RLVEGraphController(
            d, min_n=4, max_n=12, min_density=0.3, max_density=0.6
        ),
        # Range-based environment
        "RLVESumManhattanCurvedSurfaceEnv": lambda d: RLVERangeController(
            d, initial_min=10, initial_max=30, max_limit=100
        ),
    }

    factory = controllers.get(env_class_name)
    if factory:
        return factory(difficulty)
    # Return default controller for unmapped environments
    return RLVEDefaultController(difficulty)

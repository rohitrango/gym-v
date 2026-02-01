"""Unit tests for the difficulty feature.

Tests the unified difficulty parameter across environment types.
"""

from __future__ import annotations

import importlib.util
import sys

import pytest

from gym_v.utils.parameter_controller import (
    CompositeController,
    LinearController,
    ParameterController,
    StageController,
)


def import_module_directly(module_name: str, file_path: str):
    """Import a module directly from file path, bypassing package __init__.py."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class TestParameterControllerBase:
    """Tests for the ParameterController base class."""

    def test_initial_difficulty_zero(self):
        """Test that controllers start at difficulty 0 by default."""

        class SimpleController(ParameterController):
            def _initialize_parameters(self):
                self.value = 10

            def update(self):
                self.value += 5

            def get_parameters(self):
                return {"value": self.value}

        controller = SimpleController()
        assert controller.current_difficulty == 0
        assert controller.get_parameters() == {"value": 10}

    def test_initial_difficulty_nonzero(self):
        """Test that controllers can be initialized with non-zero difficulty."""

        class SimpleController(ParameterController):
            def _initialize_parameters(self):
                self.value = 10

            def update(self):
                self.value += 5

            def get_parameters(self):
                return {"value": self.value}

        controller = SimpleController(initial_difficulty=3)
        assert controller.current_difficulty == 3
        assert controller.get_parameters() == {"value": 25}  # 10 + 3*5

    def test_reset_to_difficulty(self):
        """Test resetting to a specific difficulty level."""

        class SimpleController(ParameterController):
            def _initialize_parameters(self):
                self.value = 0

            def update(self):
                self.value += 1

            def get_parameters(self):
                return {"value": self.value}

        controller = SimpleController(initial_difficulty=5)
        assert controller.current_difficulty == 5
        assert controller.get_parameters() == {"value": 5}

        controller.reset_to_difficulty(2)
        assert controller.current_difficulty == 2
        assert controller.get_parameters() == {"value": 2}

    def test_negative_difficulty_raises(self):
        """Test that negative difficulty raises ValueError."""

        class SimpleController(ParameterController):
            def _initialize_parameters(self):
                pass

            def update(self):
                pass

            def get_parameters(self):
                return {}

        controller = SimpleController()
        with pytest.raises(ValueError, match="non-negative"):
            controller.reset_to_difficulty(-1)


class TestLinearController:
    """Tests for the LinearController class."""

    def test_linear_progression(self):
        """Test linear parameter progression."""

        class GridController(LinearController):
            def _get_param_configs(self):
                return {
                    "grid_size": {"min": 3, "max": 10, "step": 1},
                    "density": {"min": 0.2, "max": 0.8, "step": 0.1},
                }

        controller = GridController()
        params = controller.get_parameters()
        assert params["grid_size"] == 3
        assert params["density"] == pytest.approx(0.2)

        controller.update()
        controller.current_difficulty += 1
        params = controller.get_parameters()
        assert params["grid_size"] == 4
        assert params["density"] == pytest.approx(0.3)

    def test_linear_max_cap(self):
        """Test that linear progression caps at max."""

        class SmallController(LinearController):
            def _get_param_configs(self):
                return {"value": {"min": 0, "max": 2, "step": 1}}

        controller = SmallController(initial_difficulty=10)
        assert controller.get_parameters()["value"] == 2


class TestStageController:
    """Tests for the StageController class."""

    def test_stage_progression(self):
        """Test staged parameter progression."""

        class DifficultyController(StageController):
            def _get_stages(self):
                return [
                    (0, {"mode": "easy", "hints": 5}),
                    (3, {"mode": "medium", "hints": 3}),
                    (6, {"mode": "hard", "hints": 1}),
                ]

        controller = DifficultyController()
        assert controller.get_parameters() == {"mode": "easy", "hints": 5}

        controller.reset_to_difficulty(3)
        assert controller.get_parameters() == {"mode": "medium", "hints": 3}

        controller.reset_to_difficulty(8)
        assert controller.get_parameters() == {"mode": "hard", "hints": 1}

    def test_stage_partial_progression(self):
        """Test progression within a stage."""

        class DifficultyController(StageController):
            def _get_stages(self):
                return [
                    (0, {"level": 1}),
                    (5, {"level": 2}),
                ]

        controller = DifficultyController(initial_difficulty=2)
        assert controller.get_parameters() == {"level": 1}

        controller.reset_to_difficulty(4)
        assert controller.get_parameters() == {"level": 1}

        controller.reset_to_difficulty(5)
        assert controller.get_parameters() == {"level": 2}


class TestCompositeController:
    """Tests for the CompositeController class."""

    def test_composite_merges_parameters(self):
        """Test that composite controller merges parameters."""

        class SizeController(ParameterController):
            def _initialize_parameters(self):
                self.size = 3

            def update(self):
                self.size += 1

            def get_parameters(self):
                return {"size": self.size}

        class DensityController(ParameterController):
            def _initialize_parameters(self):
                self.density = 0.3

            def update(self):
                self.density += 0.1

            def get_parameters(self):
                return {"density": self.density}

        composite = CompositeController(
            {
                "size": SizeController(),
                "density": DensityController(),
            }
        )

        params = composite.get_parameters()
        assert params["size"] == 3
        assert params["density"] == pytest.approx(0.3)

        composite.reset_to_difficulty(2)
        params = composite.get_parameters()
        assert params["size"] == 5
        assert params["density"] == pytest.approx(0.5)


class TestRLVEControllers:
    """Tests for RLVE-specific controllers."""

    def test_numbrix_controller(self):
        """Test Numbrix controller progression."""
        from gym_v.envs.rlve.parameter_controllers import RLVENumbrixController

        controller = RLVENumbrixController()
        params = controller.get_parameters()
        assert params["max_n_m"] == 2
        assert params["sparsity"] == 0.3

        controller.reset_to_difficulty(3)
        params = controller.get_parameters()
        assert params["max_n_m"] == 5  # 2 + 3
        assert params["sparsity"] == 0.3

    def test_grid_size_controller(self):
        """Test generic grid size controller."""
        from gym_v.envs.rlve.parameter_controllers import RLVEGridSizeController

        controller = RLVEGridSizeController(
            min_size=4, max_size=10, param_name="max_n_m"
        )
        assert controller.get_parameters() == {"max_n_m": 4}

        controller.reset_to_difficulty(5)
        assert controller.get_parameters() == {"max_n_m": 9}

        controller.reset_to_difficulty(10)
        assert controller.get_parameters() == {"max_n_m": 10}  # capped at max

    def test_graph_controller(self):
        """Test graph controller progression."""
        from gym_v.envs.rlve.parameter_controllers import RLVEGraphController

        controller = RLVEGraphController(
            min_n=4, max_n=8, min_density=0.3, max_density=0.7
        )
        params = controller.get_parameters()
        assert params["max_n"] == 4
        assert params["edge_density"] == 0.3

    def test_get_controller_for_env(self):
        """Test getting controller for specific environment."""
        from gym_v.envs.rlve.parameter_controllers import get_controller_for_env

        controller = get_controller_for_env("RLVENumbrixEnv", 0)
        assert controller is not None
        assert "max_n_m" in controller.get_parameters()

        controller = get_controller_for_env("UnknownEnv", 0)
        assert controller is not None  # Returns default controller


class TestGameRLControllers:
    """Tests for GameRL-specific controllers."""

    def test_minesweeper_controller(self):
        """Test Minesweeper staged difficulty."""
        from gym_v.envs.gamerl.parameter_controllers import GameRLMinesweeperController

        controller = GameRLMinesweeperController()
        assert controller.get_parameters()["game_difficulty"] == "Easy"

        controller.reset_to_difficulty(4)
        assert controller.get_parameters()["game_difficulty"] == "Medium"

        controller.reset_to_difficulty(7)
        assert controller.get_parameters()["game_difficulty"] == "Hard"


class TestEnvBaseDifficulty:
    """Tests for Env base class difficulty support."""

    def test_env_difficulty_property(self):
        """Test difficulty property on Env base class."""
        from gym_v.core import Env

        class TestEnv(Env):
            def inner_step(self, action):
                pass

            @property
            def description(self):
                return "Test"

        env = TestEnv()
        assert env.difficulty is None

        env = TestEnv(difficulty=5)
        assert env.difficulty == 5

    def test_env_set_difficulty(self):
        """Test set_difficulty method."""
        from gym_v.core import Env

        class TestEnv(Env):
            def inner_step(self, action):
                pass

            @property
            def description(self):
                return "Test"

        env = TestEnv()
        env.set_difficulty(10)
        assert env.difficulty == 10

    def test_env_set_difficulty_negative_raises(self):
        """Test that negative difficulty raises ValueError."""
        from gym_v.core import Env

        class TestEnv(Env):
            def inner_step(self, action):
                pass

            @property
            def description(self):
                return "Test"

        env = TestEnv()
        with pytest.raises(ValueError, match="non-negative"):
            env.set_difficulty(-1)


class TestWrapperDifficulty:
    """Tests for Wrapper difficulty delegation."""

    def test_wrapper_difficulty_delegation(self):
        """Test that Wrapper delegates difficulty to wrapped env."""
        from gym_v.core import Env, Wrapper

        class TestEnv(Env):
            def inner_step(self, action):
                pass

            @property
            def description(self):
                return "Test"

        env = TestEnv(difficulty=5)
        wrapped = Wrapper(env)

        assert wrapped.difficulty == 5

        wrapped.set_difficulty(10)
        assert wrapped.difficulty == 10
        assert env.difficulty == 10


class TestReasoningGymControllers:
    """Tests for ReasoningGym-specific controllers."""

    @pytest.fixture(scope="class")
    def reasongym_controllers(self):
        """Import ReasoningGym parameter_controllers module directly."""
        import os

        module_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "gym_v",
            "envs",
            "reasongym",
            "parameter_controllers.py",
        )
        return import_module_directly(
            "reasongym_parameter_controllers", os.path.abspath(module_path)
        )

    def test_binary_matrix_controller(self, reasongym_controllers):
        """Test BinaryMatrix controller progression."""
        ReasoningGymBinaryMatrixController = (
            reasongym_controllers.ReasoningGymBinaryMatrixController
        )

        controller = ReasoningGymBinaryMatrixController()
        params = controller.get_parameters()
        assert params["dataset_kwargs"]["min_grid_size"] == 3
        assert params["dataset_kwargs"]["max_grid_size"] == 3

        controller.reset_to_difficulty(5)
        params = controller.get_parameters()
        assert params["dataset_kwargs"]["min_grid_size"] == 8
        assert params["dataset_kwargs"]["max_grid_size"] == 8

    def test_rotate_matrix_controller(self, reasongym_controllers):
        """Test RotateMatrix controller progression."""
        ReasoningGymRotateMatrixController = (
            reasongym_controllers.ReasoningGymRotateMatrixController
        )

        controller = ReasoningGymRotateMatrixController()
        params = controller.get_parameters()
        assert params["dataset_kwargs"]["min_size"] == 2

        controller.reset_to_difficulty(3)
        params = controller.get_parameters()
        assert params["dataset_kwargs"]["min_size"] == 5

    def test_spiral_matrix_controller(self, reasongym_controllers):
        """Test SpiralMatrix controller progression."""
        ReasoningGymSpiralMatrixController = (
            reasongym_controllers.ReasoningGymSpiralMatrixController
        )

        controller = ReasoningGymSpiralMatrixController()
        params = controller.get_parameters()
        assert params["dataset_kwargs"]["min_rows"] == 3
        assert params["dataset_kwargs"]["min_cols"] == 3

    def test_largest_island_controller(self, reasongym_controllers):
        """Test LargestIsland controller progression."""
        ReasoningGymLargestIslandController = (
            reasongym_controllers.ReasoningGymLargestIslandController
        )

        controller = ReasoningGymLargestIslandController()
        params = controller.get_parameters()
        assert params["dataset_kwargs"]["min_rows"] == 4

        controller.reset_to_difficulty(4)
        params = controller.get_parameters()
        assert params["dataset_kwargs"]["min_rows"] == 8

    def test_rectangle_count_controller(self, reasongym_controllers):
        """Test RectangleCount controller progression."""
        ReasoningGymRectangleCountController = (
            reasongym_controllers.ReasoningGymRectangleCountController
        )

        controller = ReasoningGymRectangleCountController()
        params = controller.get_parameters()
        assert params["dataset_kwargs"]["min_grid_size"] == 4

    def test_rotten_oranges_controller(self, reasongym_controllers):
        """Test RottenOranges controller progression."""
        ReasoningGymRottenOrangesController = (
            reasongym_controllers.ReasoningGymRottenOrangesController
        )

        controller = ReasoningGymRottenOrangesController()
        params = controller.get_parameters()
        assert params["dataset_kwargs"]["min_n"] == 3

    def test_shortest_path_controller(self, reasongym_controllers):
        """Test ShortestPath controller progression."""
        ReasoningGymShortestPathController = (
            reasongym_controllers.ReasoningGymShortestPathController
        )

        controller = ReasoningGymShortestPathController()
        params = controller.get_parameters()
        assert params["dataset_kwargs"]["min_grid_size"] == 4

        controller.reset_to_difficulty(10)
        params = controller.get_parameters()
        assert params["dataset_kwargs"]["min_grid_size"] == 14

    def test_kakurasu_controller(self, reasongym_controllers):
        """Test Kakurasu controller progression."""
        ReasoningGymKakurasuController = (
            reasongym_controllers.ReasoningGymKakurasuController
        )

        controller = ReasoningGymKakurasuController()
        params = controller.get_parameters()
        assert params["dataset_kwargs"]["min_n_rows"] == 3

    def test_survo_controller(self, reasongym_controllers):
        """Test Survo controller progression."""
        ReasoningGymSurvoController = reasongym_controllers.ReasoningGymSurvoController

        controller = ReasoningGymSurvoController()
        params = controller.get_parameters()
        assert params["dataset_kwargs"]["min_board_size"] == 3

    def test_circuit_logic_controller(self, reasongym_controllers):
        """Test CircuitLogic staged controller."""
        ReasoningGymCircuitLogicController = (
            reasongym_controllers.ReasoningGymCircuitLogicController
        )

        controller = ReasoningGymCircuitLogicController()
        params = controller.get_parameters()
        assert params["dataset_kwargs"]["min_inputs"] == 2
        assert params["dataset_kwargs"]["min_gates"] == 3

        controller.reset_to_difficulty(5)
        params = controller.get_parameters()
        assert params["dataset_kwargs"]["min_inputs"] == 3
        assert params["dataset_kwargs"]["min_gates"] == 5

        controller.reset_to_difficulty(8)
        params = controller.get_parameters()
        assert params["dataset_kwargs"]["min_inputs"] == 5
        assert params["dataset_kwargs"]["min_gates"] == 8

    def test_knight_swap_controller(self, reasongym_controllers):
        """Test KnightSwap staged controller."""
        ReasoningGymKnightSwapController = (
            reasongym_controllers.ReasoningGymKnightSwapController
        )

        controller = ReasoningGymKnightSwapController()
        params = controller.get_parameters()
        assert params["dataset_kwargs"]["min_board_size"] == 3
        assert params["dataset_kwargs"]["min_knights"] == 2

    def test_mini_sudoku_controller(self, reasongym_controllers):
        """Test MiniSudoku controller progression."""
        ReasoningGymMiniSudokuController = (
            reasongym_controllers.ReasoningGymMiniSudokuController
        )

        controller = ReasoningGymMiniSudokuController()
        params = controller.get_parameters()
        assert params["dataset_kwargs"]["min_empty"] == 4

    def test_tsumego_controller(self, reasongym_controllers):
        """Test Tsumego staged controller."""
        ReasoningGymTsumegoController = (
            reasongym_controllers.ReasoningGymTsumegoController
        )

        controller = ReasoningGymTsumegoController()
        params = controller.get_parameters()
        assert params["dataset_kwargs"]["board_size"] == 9
        assert params["dataset_kwargs"]["difficulty"] == "easy"

        controller.reset_to_difficulty(5)
        params = controller.get_parameters()
        assert params["dataset_kwargs"]["board_size"] == 13
        assert params["dataset_kwargs"]["difficulty"] == "medium"

    def test_arc1d_controller(self, reasongym_controllers):
        """Test Arc1D default controller."""
        ReasoningGymArc1DController = reasongym_controllers.ReasoningGymArc1DController

        controller = ReasoningGymArc1DController()
        params = controller.get_parameters()
        assert "dataset_kwargs" in params

    def test_get_controller_for_env_reasongym(self, reasongym_controllers):
        """Test getting controller for ReasoningGym environments."""
        get_controller_for_env = reasongym_controllers.get_controller_for_env

        # Test new controllers
        controller = get_controller_for_env("ReasoningGymBinaryMatrixEnv", 0)
        assert controller is not None
        assert "dataset_kwargs" in controller.get_parameters()

        controller = get_controller_for_env("ReasoningGymCircuitLogicEnv", 0)
        assert controller is not None

        # Test unknown returns default
        controller = get_controller_for_env("UnknownEnv", 0)
        assert controller is not None

    def test_reasongym_controller_monotonicity(self, reasongym_controllers):
        """Test that ReasoningGym controllers are monotonic (params don't decrease)."""
        ReasoningGymBinaryMatrixController = (
            reasongym_controllers.ReasoningGymBinaryMatrixController
        )
        ReasoningGymShortestPathController = (
            reasongym_controllers.ReasoningGymShortestPathController
        )

        # Test BinaryMatrix
        controller = ReasoningGymBinaryMatrixController()
        prev_size = controller.get_parameters()["dataset_kwargs"]["min_grid_size"]
        for _ in range(10):
            controller.update()
            controller.current_difficulty += 1
            new_size = controller.get_parameters()["dataset_kwargs"]["min_grid_size"]
            assert new_size >= prev_size
            prev_size = new_size

        # Test ShortestPath
        controller = ReasoningGymShortestPathController()
        prev_size = controller.get_parameters()["dataset_kwargs"]["min_grid_size"]
        for _ in range(15):
            controller.update()
            controller.current_difficulty += 1
            new_size = controller.get_parameters()["dataset_kwargs"]["min_grid_size"]
            assert new_size >= prev_size
            prev_size = new_size


class TestTextArenaControllers:
    """Tests for TextArena-specific controllers."""

    @pytest.fixture(scope="class")
    def textarena_controllers(self):
        """Import TextArena parameter_controllers module directly."""
        import os

        module_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "gym_v",
            "envs",
            "textarena",
            "parameter_controllers.py",
        )
        return import_module_directly(
            "textarena_parameter_controllers", os.path.abspath(module_path)
        )

    def test_lightsout_controller(self, textarena_controllers):
        """Test LightsOut controller progression."""
        TextArenaLightsOutController = (
            textarena_controllers.TextArenaLightsOutController
        )

        controller = TextArenaLightsOutController()
        params = controller.get_parameters()
        assert params["size"] == 3

        controller.reset_to_difficulty(4)
        params = controller.get_parameters()
        assert params["size"] == 7  # capped at max

    def test_game2048_controller(self, textarena_controllers):
        """Test Game2048 controller progression."""
        TextArenaGame2048Controller = textarena_controllers.TextArenaGame2048Controller

        controller = TextArenaGame2048Controller()
        params = controller.get_parameters()
        assert params["target_tile"] == 128

        controller.reset_to_difficulty(4)
        params = controller.get_parameters()
        assert params["target_tile"] == 2048

    def test_sokoban_controller(self, textarena_controllers):
        """Test Sokoban controller progression."""
        TextArenaSokobanController = textarena_controllers.TextArenaSokobanController

        controller = TextArenaSokobanController()
        params = controller.get_parameters()
        assert params["dim_room"] == (6, 6)
        assert params["num_boxes"] == 2

    def test_towerofhanoi_controller(self, textarena_controllers):
        """Test TowerOfHanoi controller progression."""
        TextArenaTowerOfHanoiController = (
            textarena_controllers.TextArenaTowerOfHanoiController
        )

        controller = TextArenaTowerOfHanoiController()
        params = controller.get_parameters()
        assert params["num_disks"] == 3

        controller.reset_to_difficulty(5)
        params = controller.get_parameters()
        assert params["num_disks"] == 8  # capped at max

    def test_wordle_controller(self, textarena_controllers):
        """Test Wordle controller progression."""
        TextArenaWordleController = textarena_controllers.TextArenaWordleController

        controller = TextArenaWordleController()
        params = controller.get_parameters()
        assert params["word_length"] == 4

        controller.reset_to_difficulty(3)
        params = controller.get_parameters()
        assert params["word_length"] == 7  # capped at max

    def test_fifteenpuzzle_controller(self, textarena_controllers):
        """Test FifteenPuzzle controller (no controllable params)."""
        TextArenaFifteenPuzzleController = (
            textarena_controllers.TextArenaFifteenPuzzleController
        )

        controller = TextArenaFifteenPuzzleController()
        params = controller.get_parameters()
        assert params == {}

    def test_frozenlake_controller(self, textarena_controllers):
        """Test FrozenLake composite controller."""
        TextArenaFrozenLakeController = (
            textarena_controllers.TextArenaFrozenLakeController
        )

        controller = TextArenaFrozenLakeController()
        params = controller.get_parameters()
        assert params["size"] == 4
        assert params["num_holes"] == 2

        controller.reset_to_difficulty(4)
        params = controller.get_parameters()
        assert params["size"] == 6
        assert params["num_holes"] == 6

    def test_minesweeper_controller(self, textarena_controllers):
        """Test Minesweeper composite controller."""
        TextArenaMinesweeperController = (
            textarena_controllers.TextArenaMinesweeperController
        )

        controller = TextArenaMinesweeperController()
        params = controller.get_parameters()
        assert params["rows"] == 5
        assert params["cols"] == 5
        assert params["num_mines"] == 5

    def test_wordsearch_controller(self, textarena_controllers):
        """Test WordSearch controller."""
        TextArenaWordSearchController = (
            textarena_controllers.TextArenaWordSearchController
        )

        controller = TextArenaWordSearchController()
        params = controller.get_parameters()
        assert params["hardcore"] is False

        controller.reset_to_difficulty(5)
        params = controller.get_parameters()
        assert params["hardcore"] is True

    def test_crosswords_controller(self, textarena_controllers):
        """Test Crosswords staged controller."""
        TextArenaCrosswordsController = (
            textarena_controllers.TextArenaCrosswordsController
        )

        controller = TextArenaCrosswordsController()
        params = controller.get_parameters()
        assert params["hardcore"] is False
        assert params["num_words"] == 3

        controller.reset_to_difficulty(5)
        params = controller.get_parameters()
        assert params["hardcore"] is False
        assert params["num_words"] == 5

        controller.reset_to_difficulty(8)
        params = controller.get_parameters()
        assert params["hardcore"] is True
        assert params["num_words"] == 7

    def test_pegjump_controller(self, textarena_controllers):
        """Test PegJump staged controller."""
        TextArenaPegJumpController = textarena_controllers.TextArenaPegJumpController

        controller = TextArenaPegJumpController()
        params = controller.get_parameters()
        assert params["initial_empty"] == 1

    def test_rushhour_controller(self, textarena_controllers):
        """Test RushHour staged controller."""
        TextArenaRushHourController = textarena_controllers.TextArenaRushHourController

        controller = TextArenaRushHourController()
        params = controller.get_parameters()
        assert params["difficulty"] == "easy"

        controller.reset_to_difficulty(4)
        params = controller.get_parameters()
        assert params["difficulty"] == "medium"

        controller.reset_to_difficulty(8)
        params = controller.get_parameters()
        assert params["difficulty"] == "hard"

    def test_get_controller_for_env_textarena(self, textarena_controllers):
        """Test getting controller for TextArena environments."""
        get_controller_for_env = textarena_controllers.get_controller_for_env

        # Test new controllers
        controller = get_controller_for_env("TextArenaLightsOutEnv", 0)
        assert controller is not None
        assert "size" in controller.get_parameters()

        controller = get_controller_for_env("TextArenaRushHourEnv", 0)
        assert controller is not None
        assert "difficulty" in controller.get_parameters()

        # Test unknown returns default
        controller = get_controller_for_env("UnknownEnv", 0)
        assert controller is not None

    def test_textarena_controller_monotonicity(self, textarena_controllers):
        """Test that TextArena controllers are monotonic (params don't decrease)."""
        TextArenaLightsOutController = (
            textarena_controllers.TextArenaLightsOutController
        )
        TextArenaTowerOfHanoiController = (
            textarena_controllers.TextArenaTowerOfHanoiController
        )

        # Test LightsOut
        controller = TextArenaLightsOutController()
        prev_size = controller.get_parameters()["size"]
        for _ in range(10):
            controller.update()
            controller.current_difficulty += 1
            new_size = controller.get_parameters()["size"]
            assert new_size >= prev_size
            prev_size = new_size

        # Test TowerOfHanoi
        controller = TextArenaTowerOfHanoiController()
        prev_disks = controller.get_parameters()["num_disks"]
        for _ in range(10):
            controller.update()
            controller.current_difficulty += 1
            new_disks = controller.get_parameters()["num_disks"]
            assert new_disks >= prev_disks
            prev_disks = new_disks


class TestIntegration:
    """Integration tests for difficulty with actual environments."""

    @pytest.mark.skip(reason="Requires full environment setup")
    def test_numbrix_difficulty_integration(self):
        """Test Numbrix environment with difficulty parameter."""
        from gym_v.envs.rlve.numbrix import RLVENumbrixEnv

        # Test default difficulty
        env = RLVENumbrixEnv()
        assert env.difficulty is None

        # Test with explicit difficulty
        env = RLVENumbrixEnv(difficulty=5)
        assert env.difficulty == 5

        # Test backward compatibility (explicit params override difficulty)
        env = RLVENumbrixEnv(max_n_m=6, difficulty=10)
        assert env._max_n_m == 6  # explicit param used

    @pytest.mark.skip(reason="Requires full environment setup")
    def test_set_difficulty_updates_parameters(self):
        """Test that set_difficulty updates environment parameters."""
        from gym_v.envs.rlve.numbrix import RLVENumbrixEnv

        env = RLVENumbrixEnv()
        initial_max_n_m = env._max_n_m

        env.set_difficulty(5)
        assert env._max_n_m > initial_max_n_m


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

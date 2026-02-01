"""Unit tests for the difficulty feature.

Tests the unified difficulty parameter across environment types.
"""

from __future__ import annotations

import pytest

from gym_v.utils.parameter_controller import (
    CompositeController,
    LinearController,
    ParameterController,
    StageController,
)


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

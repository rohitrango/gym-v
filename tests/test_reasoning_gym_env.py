"""Unified tests for all ReasoningGym environments."""

from pathlib import Path
import random
import unittest

from reasoning_gym.factory import create_dataset

import gym_v

# Mapping from gym-v env_id to reasoning-gym dataset name
REASONING_GYM_ENVS = {
    "ReasoningGym/GameOfLife-v0": "game_of_life",
    "ReasoningGym/Kakurasu-v0": "kakurasu",
    "ReasoningGym/KnightSwap-v0": "knight_swap",
    "ReasoningGym/Maze-v0": "maze",
    "ReasoningGym/MiniSudoku-v0": "mini_sudoku",
    "ReasoningGym/NQueens-v0": "n_queens",
    "ReasoningGym/Sudoku-v0": "sudoku",
    "ReasoningGym/Survo-v0": "survo",
    "ReasoningGym/TowerOfHanoi-v0": "tower_of_hanoi",
    "ReasoningGym/Tsumego-v0": "tsumego",
    "ReasoningGym/SpiralMatrix-v0": "spiral_matrix",
    "ReasoningGym/RotateMatrix-v0": "rotate_matrix",
    "ReasoningGym/BinaryMatrix-v0": "binary_matrix",
    "ReasoningGym/LargestIsland-v0": "largest_island",
    "ReasoningGym/RottenOranges-v0": "rotten_oranges",
    "ReasoningGym/ShortestPath-v0": "shortest_path",
    "ReasoningGym/RectangleCount-v0": "rectangle_count",
    "ReasoningGym/CircuitLogic-v0": "circuit_logic",
    "ReasoningGym/Arc1D-v0": "arc_1d",
}


class TestReasoningGym(unittest.TestCase):
    """Test all ReasoningGym single-turn environments."""

    def _get_output_dir(self, env_id: str) -> Path:
        """Get output directory for a given environment."""
        # Convert "ReasoningGym/GameOfLife-v0" -> "test_output_reasoning_gym_game_of_life"
        env_name = env_id.split("/")[1].replace("-v0", "")
        # CamelCase to snake_case
        snake_name = "".join(
            f"_{c.lower()}" if c.isupper() else c for c in env_name
        ).lstrip("_")
        return (
            Path(__file__).resolve().parent / f"test_output_reasoning_gym_{snake_name}"
        )

    def _setup_output_dir(self, output_dir: Path) -> None:
        """Create or clean output directory."""
        if output_dir.exists():
            for p in output_dir.glob("*"):
                if p.is_file():
                    p.unlink()
        else:
            output_dir.mkdir(parents=True, exist_ok=True)

    def _test_env(self, env_id: str, dataset_name: str) -> None:
        """Test a single ReasoningGym environment."""
        output_dir = self._get_output_dir(env_id)
        self._setup_output_dir(output_dir)

        # Use random seed for each test
        test_seed = random.randint(0, 9999)
        print(f"\n[{env_id}] Using random seed: {test_seed}")

        # Note: num_players defaults to 1 for ReasoningGym envs
        env = gym_v.make(env_id)

        # 1. Reset
        obs_dict, info_dict = env.reset(seed=test_seed)

        # Check dictionary return structure
        self.assertIsInstance(
            obs_dict, dict, f"{env_id}: reset() should return dict of observations"
        )
        self.assertIsInstance(
            info_dict, dict, f"{env_id}: reset() should return dict of infos"
        )

        # Get agent_0 (default single player)
        agent_id = "agent_0"
        self.assertIn(agent_id, obs_dict)
        self.assertIn(agent_id, info_dict)

        obs = obs_dict[agent_id]
        info = info_dict[agent_id]

        # 2. Save image
        self.assertIsNotNone(obs.image)
        obs.image.save(output_dir / "0_reset.png")

        # 3. Print description and obs.text
        oracle = env.get_wrapper_attr("_oracle_answer")
        self.assertIsInstance(oracle, str)
        self.assertGreater(len(oracle), 0)

        print("\n" + "=" * 80)
        print(f"[{env_id}] SEED: {test_seed}")
        print(f"[{env_id}] DESCRIPTION:\n")
        print(env.description[:500] if len(env.description) > 500 else env.description)
        print(f"\n[{env_id}] OBS.TEXT:\n")
        text = obs.text or "No text"
        print(text[:500] if len(text) > 500 else text)
        print(f"\n[{env_id}] ORACLE ANSWER:\n")
        print(oracle[:300] + "..." if len(oracle) > 300 else oracle)
        print("=" * 80 + "\n")

        # 4. Verify reward with correct answer
        actions = {agent_id: oracle}
        obs_dict2, reward_dict, terminated_dict, truncated_dict, info_dict2 = env.step(
            actions
        )

        self.assertIn(agent_id, reward_dict)
        self.assertIn(agent_id, terminated_dict)
        self.assertIn(agent_id, truncated_dict)
        self.assertIn("__all__", terminated_dict)
        self.assertIn("__all__", truncated_dict)

        self.assertTrue(terminated_dict[agent_id])
        self.assertTrue(truncated_dict[agent_id])
        self.assertIsInstance(reward_dict[agent_id], float)
        self.assertEqual(
            reward_dict[agent_id],
            1.0,
            f"{env_id}: Expected reward 1.0 for oracle answer",
        )

        # 5. Verify reward with wrong answer
        env.reset(seed=test_seed)
        actions_wrong = {agent_id: ""}
        _, reward_dict2, terminated_dict2, truncated_dict2, _ = env.step(actions_wrong)

        self.assertTrue(terminated_dict2[agent_id])
        self.assertTrue(truncated_dict2[agent_id])
        self.assertIsInstance(reward_dict2[agent_id], float)
        self.assertEqual(
            reward_dict2[agent_id],
            0.0,
            f"{env_id}: Expected reward 0.0 for empty answer",
        )

        # 6. Verify Q&A matches original reasoning-gym
        entry_idx = info["reasoning_gym_index"]
        original_dataset = create_dataset(dataset_name, seed=test_seed, size=500)
        original_entry = original_dataset[entry_idx]

        # Answer must match exactly
        self.assertEqual(
            oracle,
            original_entry["answer"],
            f"{env_id}: gym-v answer must match original reasoning-gym answer",
        )

        # Metadata must match
        for key in original_entry.get("metadata", {}):
            if key in obs.metadata:
                self.assertEqual(
                    obs.metadata[key],
                    original_entry["metadata"][key],
                    f"{env_id}: metadata[{key}] must match original reasoning-gym",
                )

        print(
            f"✅ {env_id}: Q&A verified to match original reasoning-gym (seed={test_seed}, entry={entry_idx})"
        )


def _make_test_method(env_id: str, dataset_name: str):
    """Factory function to create test methods for each environment."""

    def test_method(self):
        self._test_env(env_id, dataset_name)

    # Set a descriptive name for the test
    env_name = env_id.split("/")[1].replace("-v0", "")
    test_method.__name__ = f"test_{env_name.lower()}"
    test_method.__doc__ = f"Test {env_id} environment."
    return test_method


# Dynamically add test methods for each environment
for _env_id, _dataset_name in REASONING_GYM_ENVS.items():
    _test_method = _make_test_method(_env_id, _dataset_name)
    setattr(TestReasoningGym, _test_method.__name__, _test_method)


if __name__ == "__main__":
    unittest.main()

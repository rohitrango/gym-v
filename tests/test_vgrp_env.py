"""Unified tests for all VGRP-Bench environments."""

from pathlib import Path
import random
import unittest

import gym_v

# Mapping from gym-v env_id to environment name
VGRP_ENVS = {
    "VGRP/Binairo-v0": "binairo",
    "VGRP/Thermometers-v0": "thermometers",
    "VGRP/TreesAndTents-v0": "treesandtents",
    "VGRP/Battleships-v0": "battleships",
    "VGRP/Renzoku-v0": "renzoku",
    "VGRP/Futoshiki-v0": "futoshiki",
    "VGRP/Hitori-v0": "hitori",
    "VGRP/StarBattle-v0": "starbattle",
}


class TestVGRP(unittest.TestCase):
    """Test all VGRP puzzle environments."""

    def _get_output_dir(self, env_id: str) -> Path:
        """Get output directory for a given environment."""
        # Convert "VGRP/Binairo-v0" -> "test_output_vgrp_binairo"
        env_name = env_id.split("/")[1].replace("-v0", "")
        # CamelCase to snake_case
        snake_name = "".join(
            f"_{c.lower()}" if c.isupper() else c for c in env_name
        ).lstrip("_")
        return Path(__file__).resolve().parent / f"test_output_vgrp_{snake_name}"

    def _setup_output_dir(self, output_dir: Path) -> None:
        """Create or clean output directory."""
        if output_dir.exists():
            for p in output_dir.glob("*"):
                if p.is_file():
                    p.unlink()
        else:
            output_dir.mkdir(parents=True, exist_ok=True)

    def _test_env(self, env_id: str, env_name: str) -> None:
        """Test a single VGRP environment."""
        output_dir = self._get_output_dir(env_id)
        self._setup_output_dir(output_dir)

        # Use random seed for each test
        test_seed = random.randint(0, 9999)
        print(f"\n[{env_id}] Using random seed: {test_seed}")

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

        # 1. Save image
        self.assertIsNotNone(obs.image)
        obs.image.save(output_dir / "0_reset.png")

        # 2. Get oracle answer from info
        oracle = info.get("oracle_answer", None)
        self.assertIsNotNone(oracle, f"{env_id}: oracle_answer not found in info")
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

        # 3. Verify reward with correct answer
        actions = {agent_id: oracle}
        obs_dict2, reward_dict, terminated_dict, truncated_dict, info_dict2 = env.step(
            actions
        )

        self.assertIn(agent_id, reward_dict)
        self.assertIn(agent_id, terminated_dict)
        self.assertIn(agent_id, truncated_dict)

        self.assertTrue(terminated_dict[agent_id])
        # truncated is True because max_episode_steps=1 and step count reached 1
        self.assertTrue(truncated_dict[agent_id])
        self.assertIsInstance(reward_dict[agent_id], float)
        self.assertEqual(
            reward_dict[agent_id],
            1.0,
            f"{env_id}: Expected reward 1.0 for oracle answer",
        )

        # 4. Verify reward with wrong answer
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

        # 5. Test with multiple seeds to verify stability
        print(f"[{env_id}] Testing with 3 additional seeds...")
        for i in range(3):
            seed = random.randint(0, 9999)
            obs_dict_test, info_dict_test = env.reset(seed=seed)
            info_test = info_dict_test[agent_id]
            obs_test = obs_dict_test[agent_id]

            oracle_test = info_test.get("oracle_answer", None)

            # Save image
            obs_test.image.save(output_dir / f"{i+1}_seed_{seed}.png")

            # Verify oracle answer is valid
            self.assertIsNotNone(oracle_test)
            self.assertIsInstance(oracle_test, str)
            self.assertGreater(len(oracle_test), 0)

            # Verify reward
            _, reward_dict_test, _, _, _ = env.step({agent_id: oracle_test})
            self.assertEqual(
                reward_dict_test[agent_id],
                1.0,
                f"{env_id}: Expected reward 1.0 for oracle answer (seed={seed})",
            )
            print(f"  ✓ Seed {seed}: Generated valid puzzle with oracle answer")

        print(f"✅ {env_id}: All tests passed (primary_seed={test_seed})")


def _make_test_method(env_id: str, env_name: str):
    """Factory function to create test methods for each environment."""

    def test_method(self):
        self._test_env(env_id, env_name)

    # Set a descriptive name for the test
    test_method.__name__ = f"test_{env_name.lower()}"
    test_method.__doc__ = f"Test {env_id} environment."
    return test_method


# Dynamically add test methods for each environment
for _env_id, _env_name in VGRP_ENVS.items():
    _test_method = _make_test_method(_env_id, _env_name)
    setattr(TestVGRP, _test_method.__name__, _test_method)


if __name__ == "__main__":
    unittest.main()

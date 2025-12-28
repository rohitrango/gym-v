"""Unified tests for all VGRP-Bench environments."""

from pathlib import Path
import random
import unittest

try:
    import gym_v
except ModuleNotFoundError as e:  # pragma: no cover
    raise ModuleNotFoundError(
        "Failed to import `gym_v`. Run tests from the `gym-v/` directory "
        "(e.g. `cd gym-v && python -m unittest ...`) or install it with "
        "`pip install -e gym-v`."
    ) from e

# Mapping from gym-v env_id to environment name
VGRP_ENVS = {
    "VGRP/Binairo-v0": "binairo",
    "VGRP/Thermometers-v0": "thermometers",
    "VGRP/TreesAndTents-v0": "treesandtents",
    "VGRP/Battleships-v0": "battleships",
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
        obs, info = env.reset(seed=test_seed)

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
        obs2, reward, terminated, truncated, info2 = env.step(oracle)
        self.assertTrue(terminated)
        self.assertTrue(truncated)
        self.assertIsInstance(reward, float)
        self.assertEqual(
            reward, 1.0, f"{env_id}: Expected reward 1.0 for oracle answer"
        )

        # 4. Verify reward with wrong answer
        env.reset(seed=test_seed)
        _, reward2, terminated2, truncated2, _ = env.step("")
        self.assertTrue(terminated2)
        self.assertTrue(truncated2)
        self.assertIsInstance(reward2, float)
        self.assertEqual(
            reward2, 0.0, f"{env_id}: Expected reward 0.0 for empty answer"
        )

        # 5. Test with multiple seeds to verify stability
        print(f"[{env_id}] Testing with 3 additional seeds...")
        for i in range(3):
            seed = random.randint(0, 9999)
            obs_test, info_test = env.reset(seed=seed)
            oracle_test = info_test.get("oracle_answer", None)

            # Save image
            obs_test.image.save(output_dir / f"{i+1}_seed_{seed}.png")

            # Verify oracle answer is valid
            self.assertIsNotNone(oracle_test)
            self.assertIsInstance(oracle_test, str)
            self.assertGreater(len(oracle_test), 0)

            # Verify reward
            _, reward_test, _, _, _ = env.step(oracle_test)
            self.assertEqual(
                reward_test,
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

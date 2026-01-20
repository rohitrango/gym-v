"""Unified tests for RLVE puzzle environments."""

from __future__ import annotations

from pathlib import Path
import random
import unittest

import gym_v

RLVE_ENVS = {
    "RLVE/HitoriPuzzle-v0": "hitori_puzzle",
    "RLVE/SkyscraperPuzzle-v0": "skyscraper_puzzle",
    "RLVE/LightUpPuzzle-v0": "light_up_puzzle",
}


class TestRLVE(unittest.TestCase):
    """Test RLVE puzzle environments."""

    def _get_output_dir(self, env_id: str) -> Path:
        env_name = env_id.split("/")[1].replace("-v0", "")
        snake_name = "".join(
            f"_{c.lower()}" if c.isupper() else c for c in env_name
        ).lstrip("_")
        return Path(__file__).resolve().parent / f"test_output_rlve_{snake_name}"

    def _setup_output_dir(self, output_dir: Path) -> None:
        if output_dir.exists():
            for p in output_dir.glob("*"):
                if p.is_file():
                    p.unlink()
        else:
            output_dir.mkdir(parents=True, exist_ok=True)

    def _test_env(self, env_id: str, env_name: str) -> None:
        output_dir = self._get_output_dir(env_id)
        self._setup_output_dir(output_dir)

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

        self.assertIsNotNone(obs.image)
        obs.image.save(output_dir / "0_reset.png")

        self.assertIsNotNone(obs.metadata)
        self.assertIn("rlve_prompt", obs.metadata)
        self.assertIn("rlve_reference_answer", obs.metadata)

        oracle = info.get("reference_answer")
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
        _, reward_dict, terminated_dict, truncated_dict, _ = env.step(actions)

        self.assertIn(agent_id, reward_dict)
        self.assertIn(agent_id, terminated_dict)
        self.assertIn(agent_id, truncated_dict)
        self.assertIn("__all__", terminated_dict)
        self.assertIn("__all__", truncated_dict)

        self.assertTrue(terminated_dict[agent_id])
        self.assertTrue(truncated_dict[agent_id])
        self.assertIsInstance(reward_dict[agent_id], float)
        self.assertAlmostEqual(
            reward_dict[agent_id], 1.0, places=6, msg=f"{env_id}: Expected reward 1.0"
        )

        # 4. Verify reward with wrong answer
        env.reset(seed=test_seed)
        actions_wrong = {agent_id: ""}
        _, reward_dict_wrong, terminated_dict_wrong, truncated_dict_wrong, _ = env.step(
            actions_wrong
        )

        self.assertTrue(terminated_dict_wrong[agent_id])
        self.assertTrue(truncated_dict_wrong[agent_id])
        self.assertIsInstance(reward_dict_wrong[agent_id], float)
        self.assertLess(reward_dict_wrong[agent_id], 0.0)

        # 5. Test with multiple seeds
        print(f"[{env_id}] Testing with 3 additional seeds...")
        for i in range(3):
            seed = random.randint(0, 9999)
            obs_dict_test, info_dict_test = env.reset(seed=seed)
            info_test = info_dict_test[agent_id]
            obs_test = obs_dict_test[agent_id]

            oracle_test = info_test.get("reference_answer")

            obs_test.image.save(output_dir / f"{i + 1}_seed_{seed}.png")

            self.assertIsNotNone(oracle_test)
            self.assertIsInstance(oracle_test, str)
            self.assertGreater(len(oracle_test), 0)

            _, reward_dict_test, _, _, _ = env.step({agent_id: oracle_test})
            self.assertAlmostEqual(
                reward_dict_test[agent_id],
                1.0,
                places=6,
                msg=f"{env_id}: Expected reward 1.0 (seed={seed})",
            )
            print(f"  ✓ Seed {seed}: Generated valid puzzle with oracle answer")

        env.close()
        print(f"✅ {env_id}: All tests passed (primary_seed={test_seed})")


def _make_test_method(env_id: str, env_name: str):
    def test_method(self):
        self._test_env(env_id, env_name)

    test_method.__name__ = f"test_{env_name.lower()}"
    test_method.__doc__ = f"Test {env_id} environment."
    return test_method


for _env_id, _env_name in RLVE_ENVS.items():
    _test_method = _make_test_method(_env_id, _env_name)
    setattr(TestRLVE, _test_method.__name__, _test_method)


if __name__ == "__main__":
    unittest.main()

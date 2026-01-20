"""Tests for Sphinx visual reasoning environments."""

from pathlib import Path
import random
import unittest

import gym_v

# Sphinx environment IDs
SPHINX_ENVS = [
    "Sphinx/TransformResult-v0",
    "Sphinx/TransformResultPoly-v0",
    "Sphinx/SymmetryFill-v0",
    "Sphinx/SymmetryFillPoly-v0",
    "Sphinx/OddOneOut-v0",
    "Sphinx/OddOneOutPoly-v0",
    "Sphinx/SequenceCompletion-v0",
    "Sphinx/SequenceCompletionPoly-v0",
]


class TestSphinx(unittest.TestCase):
    """Test all Sphinx visual reasoning environments."""

    def _get_output_dir(self, env_id: str) -> Path:
        """Get output directory for a given environment."""
        # Convert "Sphinx/TransformResult-v0" -> "test_output_sphinx_transform_result"
        env_name = env_id.split("/")[1].replace("-v0", "")
        # CamelCase to snake_case
        snake_name = "".join(
            f"_{c.lower()}" if c.isupper() else c for c in env_name
        ).lstrip("_")
        return Path(__file__).resolve().parent / f"test_output_sphinx_{snake_name}"

    def _setup_output_dir(self, output_dir: Path) -> None:
        """Create or clean output directory."""
        if output_dir.exists():
            for p in output_dir.glob("*"):
                if p.is_file():
                    p.unlink()
        else:
            output_dir.mkdir(parents=True, exist_ok=True)

    def _test_env(self, env_id: str) -> None:
        """Test a single Sphinx environment."""
        output_dir = self._get_output_dir(env_id)
        self._setup_output_dir(output_dir)

        # Use random seed for each test
        test_seed = random.randint(0, 9999)
        print(f"\n[{env_id}] Using random seed: {test_seed}")

        env = gym_v.make(env_id)
        obs_dict, info_dict = env.reset(seed=test_seed)

        # Get the first agent's data
        agent_id = next(iter(obs_dict.keys()))
        obs = obs_dict[agent_id]
        info = info_dict[agent_id]

        # 1. Save image
        self.assertIsNotNone(obs.image)
        obs.image.save(output_dir / "0_reset.png")

        # 2. Verify oracle answer exists
        oracle = info.get("oracle_answer")
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
        # The env expects a dict of actions
        action_dict = {agent_id: oracle}
        obs2_dict, reward_dict, terminated_dict, truncated_dict, info2_dict = env.step(
            action_dict
        )

        self.assertTrue(terminated_dict[agent_id])
        self.assertIsInstance(reward_dict[agent_id], float)
        self.assertEqual(
            reward_dict[agent_id],
            1.0,
            f"{env_id}: Expected reward 1.0 for oracle answer",
        )

        # 4. Verify reward with wrong answer
        obs_dict, info_dict = env.reset(seed=test_seed)
        agent_id = next(iter(obs_dict.keys()))

        # Use a clearly wrong answer
        wrong_answer = "(z)"
        action_dict = {agent_id: wrong_answer}

        _, reward2_dict, terminated2_dict, _, _ = env.step(action_dict)
        self.assertTrue(terminated2_dict[agent_id])
        self.assertIsInstance(reward2_dict[agent_id], float)
        self.assertEqual(
            reward2_dict[agent_id],
            0.0,
            f"{env_id}: Expected reward 0.0 for wrong answer",
        )

        print(
            f"✅ {env_id}: Oracle answer verified (seed={test_seed}, answer={oracle})"
        )

    def _test_deterministic(self, env_id: str) -> None:
        """Test that same seed produces same output."""
        test_seed = 42

        env = gym_v.make(env_id)

        # Reset twice with same seed
        obs1_dict, info1_dict = env.reset(seed=test_seed)
        agent_id = next(iter(obs1_dict.keys()))

        obs1 = obs1_dict[agent_id]
        info1 = info1_dict[agent_id]
        oracle1 = info1.get("oracle_answer")

        obs2_dict, info2_dict = env.reset(seed=test_seed)
        agent_id2 = next(iter(obs2_dict.keys()))
        assert agent_id == agent_id2

        obs2 = obs2_dict[agent_id]
        info2 = info2_dict[agent_id]
        oracle2 = info2.get("oracle_answer")

        # Answers must match
        self.assertEqual(
            oracle1,
            oracle2,
            f"{env_id}: Same seed must produce same oracle answer",
        )

        # Images must match (convert to bytes and compare)
        img1_bytes = obs1.image.tobytes()
        img2_bytes = obs2.image.tobytes()
        self.assertEqual(
            img1_bytes,
            img2_bytes,
            f"{env_id}: Same seed must produce same image",
        )

        # Reset with different seed should produce different result
        obs3_dict, info3_dict = env.reset(seed=test_seed + 1)
        agent_id3 = next(iter(obs3_dict.keys()))

        obs3 = obs3_dict[agent_id3]
        info3 = info3_dict[agent_id3]
        oracle3 = info3.get("oracle_answer")
        img3_bytes = obs3.image.tobytes()

        # At least one of answer or image should differ
        answer_differs = oracle1 != oracle3
        image_differs = img1_bytes != img3_bytes
        self.assertTrue(
            answer_differs or image_differs,
            f"{env_id}: Different seeds should produce different outputs",
        )

        print(f"✅ {env_id}: Deterministic generation verified (seed={test_seed})")

    def _test_multiple_resets(self, env_id: str, num_resets: int = 5) -> None:
        """Test multiple resets produce valid outputs."""
        env = gym_v.make(env_id)

        for i in range(num_resets):
            obs_dict, info_dict = env.reset(seed=i * 100)
            agent_id = next(iter(obs_dict.keys()))

            obs = obs_dict[agent_id]
            info = info_dict[agent_id]

            # Image must exist
            self.assertIsNotNone(obs.image)

            # Oracle answer must exist and be valid format (a)-(h)
            oracle = info.get("oracle_answer")
            self.assertIsInstance(oracle, str)
            self.assertIn(
                oracle,
                ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"],
                f"{env_id}: Oracle answer must be (a)-(h), got {oracle}",
            )

            # Step with oracle answer must give reward 1.0
            action_dict = {agent_id: oracle}
            _, reward_dict, _, _, _ = env.step(action_dict)
            self.assertEqual(
                reward_dict[agent_id],
                1.0,
                f"{env_id}: Reset {i} - Oracle answer should give reward 1.0",
            )

        print(f"✅ {env_id}: Multiple resets verified ({num_resets} resets)")


def _make_test_method(env_id: str):
    """Factory function to create test methods for each environment."""

    def test_method(self):
        self._test_env(env_id)

    # Set a descriptive name for the test
    env_name = env_id.split("/")[1].replace("-v0", "")
    test_method.__name__ = f"test_{env_name.lower()}"
    test_method.__doc__ = f"Test {env_id} environment."
    return test_method


def _make_deterministic_test(env_id: str):
    """Factory function to create deterministic test methods."""

    def test_method(self):
        self._test_deterministic(env_id)

    env_name = env_id.split("/")[1].replace("-v0", "")
    test_method.__name__ = f"test_{env_name.lower()}_deterministic"
    test_method.__doc__ = f"Test {env_id} deterministic generation."
    return test_method


def _make_multiple_resets_test(env_id: str):
    """Factory function to create multiple resets test methods."""

    def test_method(self):
        self._test_multiple_resets(env_id)

    env_name = env_id.split("/")[1].replace("-v0", "")
    test_method.__name__ = f"test_{env_name.lower()}_multiple_resets"
    test_method.__doc__ = f"Test {env_id} multiple resets."
    return test_method


# Dynamically add test methods for each environment
for _env_id in SPHINX_ENVS:
    _test_method = _make_test_method(_env_id)
    setattr(TestSphinx, _test_method.__name__, _test_method)

    _det_test = _make_deterministic_test(_env_id)
    setattr(TestSphinx, _det_test.__name__, _det_test)

    _multi_test = _make_multiple_resets_test(_env_id)
    setattr(TestSphinx, _multi_test.__name__, _multi_test)


if __name__ == "__main__":
    unittest.main()

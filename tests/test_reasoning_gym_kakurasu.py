from pathlib import Path
import unittest

try:
    import gym_v
except ModuleNotFoundError as e:  # pragma: no cover
    raise ModuleNotFoundError(
        "Failed to import `gym_v`. Run tests from the `gym-v/` directory "
        "(e.g. `cd gym-v && python -m unittest ...`) or install it with "
        "`pip install -e gym-v`."
    ) from e


class TestReasoningGymKakurasu(unittest.TestCase):
    def test_kakurasu_single_turn_env(self):
        # Save reset image for quick manual inspection/debugging.
        output_dir = (
            Path(__file__).resolve().parent / "test_output_reasoning_gym_kakurasu"
        )
        if output_dir.exists():
            # Keep it deterministic/clean across runs
            for p in output_dir.glob("*"):
                if p.is_file():
                    p.unlink()
        else:
            output_dir.mkdir(parents=True, exist_ok=True)

        env_id = "ReasoningGym/Kakurasu-v0"
        env = gym_v.make(env_id)

        obs, info = env.reset(seed=42)
        self.assertIsNotNone(obs.image)
        obs.image.save(output_dir / "0_reset.png")

        # Correct answer should get full reward.
        oracle = getattr(env, "_oracle_answer", None)
        self.assertIsInstance(oracle, str)
        self.assertGreater(len(oracle), 0)

        print("\n" + "=" * 80)
        print("[ReasoningGym/Kakurasu-v0] QUESTION:\n")
        print(obs.text)
        print("\n[ReasoningGym/Kakurasu-v0] ORACLE ANSWER:\n")
        print(oracle)
        print("=" * 80 + "\n")

        obs2, reward, terminated, truncated, info2 = env.step(oracle)

        self.assertTrue(terminated)
        self.assertTrue(truncated)
        self.assertIsInstance(reward, float)
        self.assertEqual(reward, 1.0)

        # Wrong answer should get low reward.
        env.reset(seed=42)
        _, reward2, terminated2, truncated2, _ = env.step("")
        self.assertTrue(terminated2)
        self.assertTrue(truncated2)
        self.assertIsInstance(reward2, float)
        self.assertEqual(reward2, 0.0)


if __name__ == "__main__":
    unittest.main()

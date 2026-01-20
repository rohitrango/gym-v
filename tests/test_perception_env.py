"""Unified tests for Perception environments."""

from __future__ import annotations

from pathlib import Path
import random
import unittest

import gym_v

PERCEPTION_ENVS = {
    "Perception/ChartToTable-v0": "chart_to_table",
    "Perception/GraphToAdjacency-v0": "graph_to_adjacency",
    "Perception/TreeToTraversal-v0": "tree_to_traversal",
    "Perception/DAGToTopoOrder-v0": "dag_to_topo_order",
    "Perception/GraphToMST-v0": "graph_to_mst",
    "Perception/FlowNetwork-v0": "flow_network",
    "Perception/FunctionGraph-v0": "function_graph",
    "Perception/ContourPlot-v0": "contour_plot",
    "Perception/PolarPlot-v0": "polar_plot",
    "Perception/VectorField-v0": "vector_field",
    "Perception/ParametricCurve-v0": "parametric_curve",
}


class TestPerception(unittest.TestCase):
    """Test Perception environments.

    Note: Perception environments return reward=0.0 from inner_step() by design.
    The reward calculation is left to external evaluators that can properly
    compare the agent's extracted data with the oracle answer.

    These tests verify:
    1. Environment can be created and reset
    2. Observations contain valid images
    3. Oracle answers are provided and non-empty
    4. Step function works correctly
    5. Multiple seeds generate valid puzzles
    """

    def _get_output_dir(self, env_id: str) -> Path:
        env_name = env_id.split("/")[1].replace("-v0", "")
        snake_name = "".join(
            f"_{c.lower()}" if c.isupper() else c for c in env_name
        ).lstrip("_")
        return Path(__file__).resolve().parent / f"test_output_perception_{snake_name}"

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

        # 1. Reset - Perception envs now return dicts for multi-agent support
        obs_dict, info_dict = env.reset(seed=test_seed)

        # Assume agent_0 is the default agent
        agent_id = "agent_0"
        self.assertIn(agent_id, obs_dict, f"{env_id}: {agent_id} not in obs_dict")
        obs = obs_dict[agent_id]
        info = info_dict[agent_id]

        # Check Observation structure
        self.assertIsNotNone(obs.image, f"{env_id}: obs.image should not be None")
        obs.image.save(output_dir / "0_reset.png")

        oracle = info.get("oracle_answer")
        self.assertIsInstance(
            oracle, str, f"{env_id}: oracle_answer should be a string"
        )
        self.assertGreater(
            len(oracle), 0, f"{env_id}: oracle_answer should not be empty"
        )

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

        # 2. Step - verify step function works with correct answer
        # Must wrap action in dict for multi-agent interface
        action_dict = {agent_id: oracle}
        obs_dict2, reward_dict, terminated_dict, truncated_dict, info_dict2 = env.step(
            action_dict
        )

        reward = reward_dict[agent_id]
        terminated = terminated_dict[agent_id]
        info2 = info_dict2[agent_id]

        self.assertTrue(terminated, f"{env_id}: terminated should be True after step")
        self.assertIsInstance(reward, float, f"{env_id}: reward should be a float")
        self.assertEqual(
            reward,
            1.0,
            f"{env_id}: Expected reward 1.0 for oracle answer, got {reward}",
        )

        # 3. Verify info still contains oracle_answer after step
        oracle2 = info2.get("oracle_answer")
        self.assertIsNotNone(
            oracle2, f"{env_id}: info should contain oracle_answer after step"
        )

        # 4. Test with wrong answer
        env.reset(seed=test_seed)
        action_wrong = {agent_id: ""}
        obs_d_w, reward_d_w, term_d_w, trunc_d_w, info_d_w = env.step(action_wrong)

        reward_wrong = reward_d_w[agent_id]
        terminated_wrong = term_d_w[agent_id]

        self.assertTrue(
            terminated_wrong, f"{env_id}: terminated should be True for wrong answer"
        )
        self.assertEqual(
            reward_wrong,
            0.0,
            f"{env_id}: Expected reward 0.0 for wrong answer, got {reward_wrong}",
        )

        # 5. Test with multiple seeds
        print(f"[{env_id}] Testing with 3 additional seeds...")
        for i in range(3):
            seed = random.randint(0, 9999)
            obs_d, info_d = env.reset(seed=seed)

            obs_test = obs_d[agent_id]
            info_test = info_d[agent_id]

            oracle_test = info_test.get("oracle_answer")

            self.assertIsNotNone(
                obs_test.image, f"{env_id}: obs.image should not be None (seed={seed})"
            )
            obs_test.image.save(output_dir / f"{i + 1}_seed_{seed}.png")

            self.assertIsNotNone(
                oracle_test, f"{env_id}: oracle_answer should not be None (seed={seed})"
            )
            self.assertIsInstance(
                oracle_test,
                str,
                f"{env_id}: oracle_answer should be string (seed={seed})",
            )
            self.assertGreater(
                len(oracle_test),
                0,
                f"{env_id}: oracle_answer should not be empty (seed={seed})",
            )

            # Verify step works
            act_d = {agent_id: oracle_test}
            obs_s, rew_s, term_s, trunc_s, info_s = env.step(act_d)
            term_test = term_s[agent_id]
            self.assertTrue(
                term_test, f"{env_id}: terminated should be True (seed={seed})"
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


for _env_id, _env_name in PERCEPTION_ENVS.items():
    _test_method = _make_test_method(_env_id, _env_name)
    setattr(TestPerception, _test_method.__name__, _test_method)


if __name__ == "__main__":
    unittest.main()

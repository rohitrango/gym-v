"""Tests for offline dataset-backed envs."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

import gym_v
import gym_v.envs  # noqa: F401  # register built-in envs


def _image_to_base64(img: Image.Image) -> str:
    """Convert PIL Image to base64 string (OpenAI API format)."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


class TestOfflineSingleTurnEnv(unittest.TestCase):
    def _write_dataset(self, root: Path) -> Path:
        img = Image.new("RGB", (32, 32), (255, 0, 0))
        img_b64 = _image_to_base64(img)

        jsonl_path = root / "dataset.jsonl"
        rows = [
            {
                "text": "Q1: What is 2+2?",
                "image": img_b64,
                "answer": "4",
                "metadata": {"id": 1},
            },
            {
                "text": "Q2: Capital of France?",
                "image": img_b64,
                "answer": "Paris",
                "metadata": {"id": 2},
            },
        ]
        with jsonl_path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return jsonl_path

    def test_seed_determinism_and_reward(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            dataset_path = self._write_dataset(root)

            env = gym_v.make(
                "Offline/SingleTurn-v0",
                datasource_type="jsonl",
                datasource_kwargs={"data_path": str(dataset_path)},
                shuffle=True,
                grader="exact_match",
                num_players=1,
            )

            agent_id = "agent_0"

            obs_dict1, info_dict1 = env.reset(seed=123)
            obs_dict2, info_dict2 = env.reset(seed=123)
            self.assertEqual(
                info_dict1[agent_id]["index"], info_dict2[agent_id]["index"]
            )
            self.assertIsNotNone(obs_dict1[agent_id].image)
            self.assertIsInstance(obs_dict1[agent_id].text, str)

            # Verify reward with the sampled oracle answer (exact_match should ignore whitespace)
            oracle = info_dict2[agent_id]["oracle_answer"]
            self.assertIsInstance(oracle, str)
            _, reward_dict, term_dict, trunc_dict, info_dict = env.step(
                {agent_id: f"  {oracle}  "}
            )
            self.assertTrue(term_dict["__all__"])
            self.assertTrue(trunc_dict["__all__"])
            self.assertEqual(reward_dict[agent_id], 1.0)
            self.assertTrue(info_dict[agent_id]["correct"])

            env.reset(seed=123)
            _, reward_dict, _, _, info_dict = env.step({agent_id: "__wrong__"})
            self.assertEqual(reward_dict[agent_id], 0.0)
            self.assertFalse(info_dict[agent_id]["correct"])

    def test_shuffle_sampling_no_repeats_in_epoch(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            dataset_path = self._write_dataset(root)

            env = gym_v.make(
                "Offline/SingleTurn-v0",
                datasource_type="jsonl",
                datasource_kwargs={"data_path": str(dataset_path)},
                shuffle=True,
                grader="exact_match",
                num_players=1,
            )

            agent_id = "agent_0"

            # With 2 samples, first two resets should cover both indices (no repeats).
            _, info_dict_a = env.reset(seed=123)
            _, info_dict_b = env.reset()
            self.assertNotEqual(
                info_dict_a[agent_id]["index"], info_dict_b[agent_id]["index"]
            )

            # Next reset starts a new epoch; index can repeat across epochs.
            _, info_dict_c = env.reset()
            self.assertIn(info_dict_c[agent_id]["index"], {0, 1})

    def test_batch_size_multiple_agents(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            dataset_path = self._write_dataset(root)

            num_players = 16
            env = gym_v.make(
                "Offline/SingleTurn-v0",
                datasource_type="jsonl",
                datasource_kwargs={"data_path": str(dataset_path)},
                shuffle=True,
                grader="exact_match",
                num_players=num_players,
            )

            obs_dict, info_dict = env.reset(seed=42)

            # Check all agents have observations and info
            self.assertEqual(len(obs_dict), num_players)
            self.assertEqual(len(info_dict), num_players)

            for i in range(num_players):
                agent_id = f"agent_{i}"
                self.assertIn(agent_id, obs_dict)
                self.assertIn(agent_id, info_dict)
                self.assertIsNotNone(obs_dict[agent_id].image)
                self.assertIsInstance(info_dict[agent_id]["oracle_answer"], str)

            # Each agent answers with their own oracle answer
            actions = {
                f"agent_{i}": info_dict[f"agent_{i}"]["oracle_answer"]
                for i in range(num_players)
            }
            _, reward_dict, term_dict, trunc_dict, result_info = env.step(actions)

            # All agents should get reward 1.0 for correct answers
            for i in range(num_players):
                agent_id = f"agent_{i}"
                self.assertEqual(reward_dict[agent_id], 1.0)
                self.assertTrue(result_info[agent_id]["correct"])

            self.assertTrue(term_dict["__all__"])
            self.assertTrue(trunc_dict["__all__"])

            env.close()


if __name__ == "__main__":
    unittest.main()

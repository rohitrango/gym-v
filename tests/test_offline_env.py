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
            )

            obs1, info1 = env.reset(seed=123)
            obs2, info2 = env.reset(seed=123)
            self.assertEqual(info1["index"], info2["index"])
            self.assertIsNotNone(obs1.image)
            self.assertIsInstance(obs1.text, str)

            # Verify reward with the sampled oracle answer (exact_match should ignore whitespace)
            oracle = info2["oracle_answer"]
            self.assertIsInstance(oracle, str)
            _, r_ok, term_ok, trunc_ok, info_ok = env.step(f"  {oracle}  ")
            self.assertTrue(term_ok)
            self.assertTrue(trunc_ok)
            self.assertEqual(r_ok, 1.0)
            self.assertTrue(info_ok["correct"])

            env.reset(seed=123)
            _, r_bad, _, _, info_bad = env.step("__wrong__")
            self.assertEqual(r_bad, 0.0)
            self.assertFalse(info_bad["correct"])

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
            )

            # With 2 samples, first two resets should cover both indices (no repeats).
            _, info_a = env.reset(seed=123)
            _, info_b = env.reset()
            self.assertNotEqual(info_a["index"], info_b["index"])

            # Next reset starts a new epoch; index can repeat across epochs.
            _, info_c = env.reset()
            self.assertIn(info_c["index"], {0, 1})


if __name__ == "__main__":
    unittest.main()

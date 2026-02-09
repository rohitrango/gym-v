import base64
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

if sys.version_info < (3, 10):
    raise unittest.SkipTest("gym-v requires Python 3.10+.")

from gym_v.envs.eval.t2ieval.client import BaseNetworkClient
from gym_v.envs.eval.t2ieval.geneval_env import GenevalEnv


def _image_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


async def _fake_request_many(self, payloads):
    responses = []
    for _ in payloads:
        responses.append(
            {
                "geneval": {
                    "score": 1.0,
                    "correct": True,
                    "strict_correct": True,
                    "tag": "single_object",
                }
            }
        )
    return responses


class TestGenevalEnv(unittest.TestCase):
    def test_env_reset_and_step(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "geneval.jsonl"
            rows = [
                {"prompt": "a cat", "tag": "single_object"},
                {"prompt": "a dog", "tag": "single_object"},
            ]
            with dataset_path.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")

            img = Image.new("RGB", (16, 16), (255, 0, 0))
            img_b64 = _image_to_data_url(img)

            with patch.object(BaseNetworkClient, "request_many", new=_fake_request_many):
                env = GenevalEnv(dataset_path=str(dataset_path), server_url="http://test")
                obs_dict, info_dict = env.reset()

                actions = {agent_id: {"image": img_b64} for agent_id in obs_dict}
                _, reward_dict, terminated, truncated, info = env.step(actions)

                self.assertEqual(set(reward_dict.keys()), set(obs_dict.keys()))
                for agent_id in reward_dict:
                    self.assertEqual(reward_dict[agent_id], 1.0)

                self.assertTrue(terminated["__all__"])
                self.assertFalse(truncated["__all__"])
                self.assertIn("__all__", info)

                env.close()


if __name__ == "__main__":
    unittest.main()

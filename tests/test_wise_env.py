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
from gym_v.envs.eval.t2ieval.wise_env import WiseEnv


def _image_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


async def _fake_request_many(self, payloads):
    content = "Consistency: 2\nRealism: 1\nAesthetic Quality: 0"
    responses = []
    for _ in payloads:
        responses.append({"choices": [{"message": {"content": content}}]})
    return responses


class TestWiseEnv(unittest.TestCase):
    def test_env_reset_and_step(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "wise.json"
            rows = [
                {
                    "Prompt": "a cat",
                    "Explanation": "a simple cat",
                    "prompt_id": 1,
                    "Subcategory": "CULTURE",
                }
            ]
            with dataset_path.open("w", encoding="utf-8") as f:
                json.dump(rows, f)

            img = Image.new("RGB", (16, 16), (255, 255, 255))
            img_b64 = _image_to_data_url(img)

            with patch.object(BaseNetworkClient, "request_many", new=_fake_request_many):
                env = WiseEnv(
                    dataset_path=str(dataset_path),
                    openai_base_url="http://test",
                    openai_api_key="sk-test",
                    model="test-model",
                )
                obs_dict, info_dict = env.reset()

                actions = {agent_id: {"image": img_b64} for agent_id in obs_dict}
                _, reward_dict, terminated, truncated, info = env.step(actions)

                self.assertEqual(set(reward_dict.keys()), set(obs_dict.keys()))
                for agent_id in reward_dict:
                    self.assertAlmostEqual(reward_dict[agent_id], 0.8, places=2)
                    self.assertIn("wiscore", info[agent_id])

                self.assertTrue(terminated["__all__"])
                self.assertFalse(truncated["__all__"])

                env.close()


if __name__ == "__main__":
    unittest.main()

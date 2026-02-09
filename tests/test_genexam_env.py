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
from gym_v.envs.eval.t2ieval.genexam_env import GenExamEnv


def _image_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


async def _fake_request_many(self, payloads):
    content = """```json
{"global_evaluation": {"Clarity and Readability": {"score": 2}, "Logical Consistency": {"score": 2}, "Spelling": {"score": 2}},
 "answers": [{"answer": 1}, {"answer": 0}]}
```"""
    responses = []
    for _ in payloads:
        responses.append({"choices": [{"message": {"content": content}}]})
    return responses


class TestGenExamEnv(unittest.TestCase):
    def test_env_reset_and_step(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            images_dir = root / "images"
            ann_dir = root / "annotations"
            images_dir.mkdir(parents=True, exist_ok=True)
            ann_dir.mkdir(parents=True, exist_ok=True)

            gt_img = Image.new("RGB", (16, 16), (0, 0, 255))
            gt_path = images_dir / "0001.png"
            gt_img.save(gt_path)

            annotation_path = ann_dir / "All_Subjects.jsonl"
            row = {
                "prompt": "a cat",
                "image_path": "0001.png",
                "scoring_points": [
                    {"question": "Is there a cat?", "score": 0.6},
                    {"question": "Is it black?", "score": 0.4},
                ],
            }
            with annotation_path.open("w", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

            gen_img = Image.new("RGB", (16, 16), (255, 255, 0))
            gen_img_b64 = _image_to_data_url(gen_img)

            with patch.object(BaseNetworkClient, "request_many", new=_fake_request_many):
                env = GenExamEnv(
                    dataset_root=str(root),
                    openai_base_url="http://test",
                    openai_api_key="sk-test",
                    model="test-model",
                    reward_mode="relaxed",
                    max_tokens=64,
                    image_max_size=32,
                )
                obs_dict, info_dict = env.reset()

                actions = {agent_id: {"image": gen_img_b64} for agent_id in obs_dict}
                _, reward_dict, terminated, truncated, info = env.step(actions)

                self.assertEqual(set(reward_dict.keys()), set(obs_dict.keys()))
                for agent_id in reward_dict:
                    self.assertAlmostEqual(reward_dict[agent_id], 0.72, places=2)
                    self.assertIn("relaxed_score", info[agent_id])
                    self.assertIn("strict_score", info[agent_id])

                self.assertTrue(terminated["__all__"])
                self.assertFalse(truncated["__all__"])

                env.close()


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from collections import defaultdict
import json
import os
from typing import Any

from PIL import Image
from torch.utils.data import Dataset

from gym_v.core import Env, Observation
from gym_v.envs.eval.t2ieval.client import (
    BaseNetworkClient,
    run_coroutine,
)
from deploy.utils import (
    convert_base64_to_local_image_or_video,
    convert_local_image_or_video_to_base_64,
)
from gym_v.envs.eval.t2ieval.utils import select_indices


class GenevalPromptDataset(Dataset):
    def __init__(
        self,
        dataset_root: str | None = None,
        split: str = "test",
        file_path: str | None = None,
    ):
        if file_path is None and dataset_root and dataset_root.endswith(".jsonl"):
            file_path = dataset_root
        if file_path is None:
            split_path = os.path.join(dataset_root, f"{split}_metadata.jsonl")
            eval_path = os.path.join(dataset_root, "evaluation_metadata.jsonl")
            file_path = split_path if os.path.exists(split_path) else eval_path

        self.file_path = file_path
        with open(self.file_path, encoding="utf-8") as f:
            self.metadatas = [json.loads(line) for line in f]
        self.prompts = [item.get("prompt", "") for item in self.metadatas]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas


def _postprocess_geneval_results(results: list[dict[str, Any]]):
    required_keys = [
        "single_object",
        "two_object",
        "counting",
        "colors",
        "position",
        "color_attr",
    ]
    scores = []
    strict_rewards = []
    grouped_strict_rewards = defaultdict(list)
    rewards = []
    grouped_rewards = defaultdict(list)
    for result in results:
        strict_rewards.append(1.0 if result.get("strict_correct") else 0.0)
        scores.append(result.get("score"))
        rewards.append(1.0 if result.get("correct") else 0.0)
        tag = result.get("tag")
        if tag in required_keys:
            grouped_strict_rewards[tag].append(
                1.0 if result.get("strict_correct") else 0.0
            )
            grouped_rewards[tag].append(1.0 if result.get("correct") else 0.0)
    return (
        scores,
        rewards,
        strict_rewards,
        {key: grouped_rewards.get(key, []) for key in required_keys},
        {key: grouped_strict_rewards.get(key, []) for key in required_keys},
    )

def _preprocess_geneval_payload(
    image: Image.Image,
    prompt: str | None,
    metadata: Any,
    *,
    only_strict: bool,
    model: str,
) -> dict[str, Any]:
    meta = metadata
    prompt_text = str(prompt)
    data_url = convert_local_image_or_video_to_base_64(image, media_type="image")
    return {
        "model": model,
        "prompt": prompt_text,
        "multimodal_outputs": {"image": data_url},
        "metadata": {
            **meta,
            "only_strict": bool(only_strict),
        },
    }


def geneval_score_async(
    *,
    server_url: str,
    only_strict: bool = True,
    timeout_s: float = 120.0,
    max_retries: int = 1000,
    backoff_factor: float = 1.0,
    model: str = "geneval",
    max_concurrency: int | None = None,
):
    client = BaseNetworkClient(
        endpoint=f"{server_url}/v1/generate",
        timeout_s=timeout_s,
        max_retries=max_retries,
        backoff_factor=backoff_factor,
        concurrency=1 if max_concurrency is None else max_concurrency,
    )

    async def _fn(images, prompts=None, metadatas=None):
        images_list = list(images)
        prompts_list = list(prompts)
        metadatas_list = list(metadatas)

        payloads = [
            _preprocess_geneval_payload(
                image,
                prompt,
                metadata,
                only_strict=only_strict,
                model=model,
            )
            for image, prompt, metadata in zip(
                images_list, prompts_list, metadatas_list, strict=False
            )
        ]
        responses = await client.request_many(payloads)
        unwrapped: list[dict[str, Any]] = []
        for response in responses:
            parsed = response['geneval']
            unwrapped.append(parsed)
        return _postprocess_geneval_results(unwrapped)

    return _fn


class GenevalEnv(Env):
    """
    T2I eval environment for GenEval.

    reset(): returns prompts + metadata for all items (or a subset via options).
    step(): expects images as actions, queries the GenEval reward server, returns rewards.
    """

    def __init__(
        self,
        dataset_root: str | None = None,
        dataset_path: str | None = None,
        split: str = "test",
        server_url: str | None = None,
        only_strict: bool = False,
        timeout_s: float = 120.0,
        max_retries: int = 1000,
        backoff_factor: float = 1.0,
        max_concurrency: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(max_episode_steps=1)

        server_url = os.environ.get("SERVER_URL", server_url)
        if dataset_root is None and dataset_path is None:
            raise ValueError(
                "Geneval requires dataset_root or dataset_path."
            )

        self.dataset = GenevalPromptDataset(
            dataset_root=dataset_root, split=split, file_path=dataset_path
        )
        self.prompts = self.dataset.prompts
        self.metadatas = self.dataset.metadatas
        self.indices = list(range(len(self.prompts)))

        self._agent_ids = {f"agent_{i}" for i in self.indices}
        self._active_indices = self.indices

        self.only_strict = only_strict
        self.timeout_s = timeout_s
        self.max_concurrency = max_concurrency

        if server_url is None:
            server_url = os.environ.get("GENEVAL_SERVER_URL")
        if not server_url:
            raise ValueError("Geneval requires GENEVAL_SERVER_URL or server_url.")
        self.server_url = server_url

        self._score_async_fn = geneval_score_async(
            server_url=self.server_url,
            only_strict=self.only_strict,
            timeout_s=self.timeout_s,
            max_retries=max_retries,
            backoff_factor=backoff_factor,
            max_concurrency=self.max_concurrency,
        )

    @property
    def description(self) -> str:
        return f"GenEval T2I eval env with {len(self.prompts)} prompts."

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)

        indices = select_indices(self.indices, options, self.np_random)

        self._active_indices = indices
        self._agent_ids = {f"agent_{i}" for i in indices}

        obs_dict = {}
        info_dict = {}
        for idx in indices:
            agent_id = f"agent_{idx}"
            prompt = self.prompts[idx]
            metadata = self.metadatas[idx]
            obs_dict[agent_id] = Observation(image=None, text=prompt, metadata=metadata)
            info_dict[agent_id] = {
                "index": idx,
                "prompt": prompt,
                "metadata": metadata,
            }

        return obs_dict, info_dict

    def step(self, action_dict):
        agent_ids = list(action_dict.keys())

        images = []
        prompts = []
        metadatas = []
        indices = []

        for agent_id in agent_ids:
            action = action_dict[agent_id]
            pil_image = convert_base64_to_local_image_or_video(
                action["image"], media_type="image"
            )
            images.append(pil_image)
            idx = int(agent_id.split("_")[-1])
            indices.append(idx)
            prompts.append(self.prompts[idx])
            metadatas.append(self.metadatas[idx])

        scores, rewards, strict_rewards, group_rewards, group_strict_rewards = (
            run_coroutine(
                self._score_async_fn(images, prompts=prompts, metadatas=metadatas)
            )
        )

        reward_dict = {}
        info_dict = {}
        for idx, agent_id in enumerate(agent_ids):
            score = scores[idx]
            reward = rewards[idx]
            strict_reward = strict_rewards[idx]
            reward_value = strict_reward if self.only_strict else reward
            reward_dict[agent_id] = float(reward_value)
            info_dict[agent_id] = {
                "index": indices[idx],
                "score": score,
                "reward": reward,
                "strict_reward": strict_reward,
            }

        terminated = {agent_id: True for agent_id in agent_ids}
        truncated = {agent_id: False for agent_id in agent_ids}
        terminated["__all__"] = True
        truncated["__all__"] = False
        info_dict["__all__"] = {
            "group_rewards": group_rewards,
            "group_strict_rewards": group_strict_rewards,
        }

        return {}, reward_dict, terminated, truncated, info_dict

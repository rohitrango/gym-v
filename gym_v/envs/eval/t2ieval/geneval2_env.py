from __future__ import annotations

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
from scipy.stats import gmean

SCORE_SCALE = 100.0
YES_ANSWER_LIST = ["Yes", "yes", " yes", " Yes"]
SUPPORTED_METHODS = {"vqascore", "tifa", "soft_tifa_am", "soft_tifa_gm"}


class Geneval2PromptDataset(Dataset):
    def __init__(self, dataset_root: str | None = None, file_path: str | None = None):
        if file_path is None and dataset_root and dataset_root.endswith(".jsonl"):
            file_path = dataset_root
        if file_path is None:
            file_path = os.path.join(dataset_root, "geneval2_data.jsonl")

        self.file_path = file_path
        with open(self.file_path, encoding="utf-8") as f:
            self.metadatas = [json.loads(line) for line in f]
        self.prompts = [item.get("prompt") for item in self.metadatas]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas


def _postprocess_geneval2_results(
    *,
    responses: list[dict[str, Any]],
    payload_metas: list[dict[str, Any]],
    method: str,
) -> tuple[list[float], list[list[float]]]:
    item_scores = {}
    for response, meta in zip(responses, payload_metas, strict=False):
        answer_list = meta["answer_list"]
        item_idx = meta["item_idx"]
        if method == "tifa":
            pred = response.get("texts")
            if isinstance(pred, list):
                pred = pred[0] if pred else ""
            if isinstance(pred, str):
                score = 1.0 if pred.lower() in answer_list else 0.0
            else:
                score = 0.0
        else:
            scores_list = response.get("Scores")
            score = sum(scores_list)
        item_scores.setdefault(item_idx, []).append(float(score))

    all_score_lists = []
    for item_idx in sorted(item_scores.keys()):
        score_list = item_scores[item_idx]
        all_score_lists.append(score_list)

    # Calculating total score
    if method == 'soft_tifa_gm':
        per_prompt_scores = [gmean(s) for s in all_score_lists]
    else:
        per_prompt_scores = [sum(s)/len(s) for s in all_score_lists]
    total_scores = 100 * per_prompt_scores

    return total_scores, all_score_lists

def per_skill_analysis(all_score_lists, all_skill_lists):
    object_score, object_total = 0, 0
    count_score, count_total = 0, 0
    position_score, position_total = 0, 0
    verb_score, verb_total = 0, 0
    attribute_score, attribute_total = 0, 0

    for score_list, skill_list in zip(all_score_lists, all_skill_lists):
        for i in range(len(score_list)):
            if skill_list[i] == "object":
                object_score += score_list[i]
                object_total += 1
            elif skill_list[i] == "count":
                count_score += score_list[i]
                count_total += 1
            elif skill_list[i] == "position":
                position_score += score_list[i]
                position_total += 1
            elif skill_list[i] == "verb":
                verb_score += score_list[i]
                verb_total += 1
            elif skill_list[i] == "attribute":
                attribute_score += score_list[i]
                attribute_total += 1

    object_accuracy = 100 * object_score / object_total
    attribute_accuracy = 100 * attribute_score / attribute_total
    count_accuracy = 100 * count_score / count_total
    position_accuracy = 100 * position_score / position_total
    verb_accuracy = 100 * verb_score / verb_total
    return (
        object_accuracy,
        attribute_accuracy,
        count_accuracy,
        position_accuracy,
        verb_accuracy,
    )


# Per-atomicity analysis (Soft-TIFA GM)
def per_atomicity_analysis(all_score_lists, atomicity_list):
    all_atomicity_dict = {k: {} for k in range(3, 11)}
    for k in all_atomicity_dict:
        all_atomicity_dict[k] = {'correct': 0, 'total': 0}
    
    for score_list, atomicity in zip(all_score_lists, atomicity_list):
        all_atomicity_dict[atomicity]['correct'] += gmean(score_list)
        all_atomicity_dict[atomicity]['total'] += 1

    # Here, too, "accuracy" is an estimate.
    for atomicity in all_atomicity_dict:
        all_atomicity_dict[atomicity]['accuracy'] = \
                (all_atomicity_dict[atomicity]['correct'])*100 / \
                all_atomicity_dict[atomicity]['total']
    return all_atomicity_dict

def soft_tifa_report(
    method: str,
    score_lists: list[list[float]],
    atom_counts: list[int],
    skills: list[list[str]],
):
    if method == "soft_tifa_am":
        (
            object_accuracy,
            attribute_accuracy,
            count_accuracy,
            position_accuracy,
            verb_accuracy,
        ) = per_skill_analysis(score_lists, skills)
        return {
            "method": method,
            "per_skill": {
                "object": object_accuracy,
                "attribute": attribute_accuracy,
                "count": count_accuracy,
                "position": position_accuracy,
                "verb": verb_accuracy,
            },
        }
    if method == "soft_tifa_gm":
        atomicity = per_atomicity_analysis(score_lists, atom_counts)
        return {
            "method": method,
            "per_atomicity": atomicity,
        }
    return {"method": method}


def return_numeric_string(number: str) -> str:
    match number:
        case "one":
            return "1"
        case "two":
            return "2"
        case "three":
            return "3"
        case "four":
            return "4"
        case "five":
            return "5"
        case "six":
            return "6"
        case "seven":
            return "7"
        case "eight":
            return "8"
        case "nine":
            return "9"
        case "ten":
            return "10"
    return "other"


def _build_geneval2_payloads(
    *,
    images: list[Image.Image],
    prompts: list[str],
    metadatas: list[Any],
    method: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    def _payload(message: str, data_url: str, answer_list: list[str]) -> dict[str, Any]:
        gen_cfg = {
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": True,
        }
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": data_url},
                        {"type": "text", "text": message},
                    ],
                }
            ],
            "multimodal_inputs": {"image": data_url},
            "metadata": {"gen": gen_cfg, "answer_list": answer_list},
        }

    def _answer_list(question: str, answer: str) -> list[str]:
        if question.startswith("How many"):
            return [
                answer,
                answer.capitalize(),
                " " + answer,
                " " + answer.capitalize(),
                return_numeric_string(answer),
                " " + return_numeric_string(answer),
            ]
        return YES_ANSWER_LIST

    payloads: list[dict[str, Any]] = []
    payload_metas: list[dict[str, Any]] = []
    for item_idx, (image, prompt, metadata) in enumerate(
        zip(images, prompts, metadatas, strict=False)
    ):
        data_url = convert_local_image_or_video_to_base_64(
            image, media_type="image"
        )
        if method == "vqascore":
            qa_pairs = [
                (
                    f'Does this image show "{prompt}"? Answer the question with Yes or No.',
                    YES_ANSWER_LIST,
                )
            ]
        else:
            qa_pairs = [
                (
                    f"{question} Answer in one word.",
                    _answer_list(question, answer),
                )
                for question, answer in metadata["vqa_list"]
            ]

        for message, answer_list in qa_pairs:
            payloads.append(_payload(message, data_url, answer_list))
            payload_metas.append(
                {
                    "item_idx": item_idx,
                    "answer_list": answer_list,
                }
            )

    return payloads, payload_metas


def geneval2_score_async(
    *,
    server_url: str,
    method: str = "soft_tifa_gm",
    timeout_s: float = 120.0,
    max_retries: int = 3,
    backoff_factor: float = 1.0,
    max_concurrency: int | None = None,
    reward_name: str = "qwen3vl",
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

        payloads, payload_metas = _build_geneval2_payloads(
            images=images_list,
            prompts=prompts_list,
            metadatas=metadatas_list,
            method=method,
        )
        responses = await client.request_many(payloads)
        unwrapped: list[dict[str, Any]] = []
        for response in responses:
            parsed = response[reward_name]
            unwrapped.append(parsed)
        return _postprocess_geneval2_results(
            responses=unwrapped,
            payload_metas=payload_metas,
            method=method,
        )

    return _fn


class Geneval2Env(Env):
    """
    T2I eval environment for GenEval2.

    reset(): returns prompts + metadata for all items (or subset).
    step(): expects images as actions, queries the GenEval2 reward server.
    """

    def __init__(
        self,
        dataset_root: str | None = None,
        dataset_path: str | None = None,
        method: str = "soft_tifa_gm",
        soft_tifa_report_enabled: bool = False,
        timeout_s: float = 120.0,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
        max_concurrency: int | None = None,
        model: str | None = None,
        server_url: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(max_episode_steps=1)

        server_url = os.environ.get("SERVER_URL", server_url)
        if dataset_root is None and dataset_path is None:
            raise ValueError(
                "GENEVAL2 requires dataset_root or dataset_path."
            )

        self.dataset = Geneval2PromptDataset(
            dataset_root=dataset_root, file_path=dataset_path
        )
        self.prompts = self.dataset.prompts
        self.indices = list(range(len(self.prompts)))

        self._agent_ids = {f"agent_{i}" for i in self.indices}

        self.method = method
        if self.method not in SUPPORTED_METHODS:
            raise NotImplementedError(f"Unsupported method: {self.method}")
        self.soft_tifa_report_enabled = soft_tifa_report_enabled
        self.timeout_s = timeout_s

        if server_url is None:
            raise ValueError("GenEval2 requires server_url.")

        self._score_async_fn = geneval2_score_async(
            server_url=server_url,
            method=self.method,
            timeout_s=self.timeout_s,
            max_retries=max_retries,
            backoff_factor=backoff_factor,
            max_concurrency=max_concurrency,
        )

    @property
    def description(self) -> str:
        return f"GenEval2 T2I eval env with {len(self.prompts)} prompts."

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)

        indices = select_indices(self.indices, options, self.np_random)

        self._agent_ids = {f"agent_{i}" for i in indices}

        obs_dict = {}
        info_dict = {}
        for idx in indices:
            agent_id = f"agent_{idx}"
            prompt = self.prompts[idx]
            obs_dict[agent_id] = Observation(
                image=None,
                text=prompt,
                metadata=self.dataset.metadatas[idx],
            )
            info_dict[agent_id] = {
                "index": idx,
                "prompt": prompt,
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
            idx = int(agent_id.split("_")[-1])
            indices.append(idx)
            image = convert_base64_to_local_image_or_video(
                action["image"], media_type="image"
            )
            prompt = self.prompts[idx]

            images.append(image)
            prompts.append(prompt)
            metadatas.append(self.dataset.metadatas[idx])

        scores, score_lists = run_coroutine(
            self._score_async_fn(images, prompts=prompts, metadatas=metadatas)
        )

        reward_dict = {}
        info_dict = {}
        for idx, agent_id in enumerate(agent_ids):
            score = scores[idx]
            reward_dict[agent_id] = score
            info_dict[agent_id] = {
                "index": indices[idx],
                "method": self.method,
                "score": score,
                "score_list": score_lists[idx],
            }

        if self.soft_tifa_report_enabled:
            info_dict["__all__"] = {
                "soft_tifa_report": soft_tifa_report(
                    self.method,
                    score_lists,
                    [self.dataset.metadatas[idx]["atom_count"] for idx in indices],
                    [self.dataset.metadatas[idx]["skills"] for idx in indices],
                )
            }

        terminated = {agent_id: True for agent_id in agent_ids}
        truncated = {agent_id: False for agent_id in agent_ids}
        terminated["__all__"] = True
        truncated["__all__"] = False

        return {}, reward_dict, terminated, truncated, info_dict

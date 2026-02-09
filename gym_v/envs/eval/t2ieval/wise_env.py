from __future__ import annotations

import json
import os
import re
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


class WisePromptDataset(Dataset):
    def __init__(self, dataset_path: str | list[str]):
        paths = (
            list(dataset_path)
            if isinstance(dataset_path, list | tuple)
            else [dataset_path]
        )
        self.dataset_path = paths
        self.metadatas = []
        for path in paths:
            with open(path, 'r') as f:
                self.metadatas.extend(json.load(f))
        self.metadatas.sort(key=lambda item: item.get("prompt_id"))
        self.prompts = [item.get("Prompt") for item in self.metadatas]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}


def extract_scores(txt: str) -> dict[str, float]:
    pat = r"\*{0,2}(Consistency|Realism|Aesthetic Quality)\*{0,2}\s*[::]?\s*(\d)"
    matches = re.findall(pat, txt, re.IGNORECASE)
    out = {}
    for k, v in matches:
        out[k.lower().replace(" ", "_")] = float(v)
    return out


WISE_CATEGORIES = ("CULTURE", "TIME", "SPACE", "BIOLOGY", "PHYSICS", "CHEMISTRY")
WISE_CATEGORY_WEIGHTS = {
    "CULTURE": 0.4,
    "TIME": 0.167,
    "SPACE": 0.133,
    "BIOLOGY": 0.1,
    "PHYSICS": 0.1,
    "CHEMISTRY": 0.1,
}


def _calculate_wiscore(
    consistency: float, realism: float, aesthetic_quality: float
) -> float:
    return (0.7 * consistency + 0.2 * realism + 0.1 * aesthetic_quality) / 2


def _wise_category(prompt_id: int) -> str | None:
    if 1 <= prompt_id <= 400:
        return "CULTURE"
    if 401 <= prompt_id <= 567:
        return "TIME"
    if 568 <= prompt_id <= 700:
        return "SPACE"
    if 701 <= prompt_id <= 800:
        return "BIOLOGY"
    if 801 <= prompt_id <= 900:
        return "PHYSICS"
    if 901 <= prompt_id <= 1000:
        return "CHEMISTRY"
    return None


def _build_wise_report(
    score_results: list[dict[str, float]],
    metadatas: list[dict[str, Any]],
) -> dict[str, Any]:
    sums = {cat: 0.0 for cat in WISE_CATEGORIES}
    counts = {cat: 0 for cat in WISE_CATEGORIES}

    for scores, meta in zip(score_results, metadatas, strict=False):
        prompt_id = meta.get("prompt_id")
        category = _wise_category(prompt_id)
        wiscore = _calculate_wiscore(
            scores["consistency"],
            scores["realism"],
            scores["aesthetic_quality"],
        )
        sums[category] += wiscore
        counts[category] += 1

    per_category = {
        cat: {
            "avg": sums[cat] / counts[cat] if counts[cat] else 0.0,
            "count": counts[cat],
        }
        for cat in WISE_CATEGORIES
    }

    overall = None
    if all(counts[cat] > 0 for cat in WISE_CATEGORIES):
        overall = sum(
            (sums[cat] / counts[cat]) * WISE_CATEGORY_WEIGHTS[cat]
            for cat in WISE_CATEGORIES
        )

    return {"per_category": per_category, "overall": overall}


def _build_wise_messages(
    prompt_data: dict[str, Any], image_data_url: str
) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a professional Vincennes image quality audit expert, please evaluate the image quality strictly according to the protocol.",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""Please evaluate strictly and return ONLY the three scores as requested.

# Text-to-Image Quality Evaluation Protocol

## System Instruction
You are an AI quality auditor for text-to-image generation. Apply these rules with ABSOLUTE RUTHLESSNESS. Only images meeting the HIGHEST standards should receive top scores.

**Input Parameters**
- PROMPT: [User's original prompt to]
- EXPLANATION: [Further explanation of the original prompt]
---

## Scoring Criteria

**Consistency (0-2):**  How accurately and completely the image reflects the PROMPT.
* **0 (Rejected):**  Fails to capture key elements of the prompt, or contradicts the prompt.
* **1 (Conditional):** Partially captures the prompt. Some elements are present, but not all, or not accurately.  Noticeable deviations from the prompt's intent.
* **2 (Exemplary):**  Perfectly and completely aligns with the PROMPT.  Every single element and nuance of the prompt is flawlessly represented in the image. The image is an ideal, unambiguous visual realization of the given prompt.

**Realism (0-2):**  How realistically the image is rendered.
* **0 (Rejected):**  Physically implausible and clearly artificial. Breaks fundamental laws of physics or visual realism.
* **1 (Conditional):** Contains minor inconsistencies or unrealistic elements.  While somewhat believable, noticeable flaws detract from realism.
* **2 (Exemplary):**  Achieves photorealistic quality, indistinguishable from a real photograph.  Flawless adherence to physical laws, accurate material representation, and coherent spatial relationships. No visual cues betraying AI generation.

**Aesthetic Quality (0-2):**  The overall artistic appeal and visual quality of the image.
* **0 (Rejected):**  Poor aesthetic composition, visually unappealing, and lacks artistic merit.
* **1 (Conditional):**  Demonstrates basic visual appeal, acceptable composition, and color harmony, but lacks distinction or artistic flair.
* **2 (Exemplary):**  Possesses exceptional aesthetic quality, comparable to a masterpiece.  Strikingly beautiful, with perfect composition, a harmonious color palette, and a captivating artistic style. Demonstrates a high degree of artistic vision and execution.

---

## Output Format

**Do not include any other text, explanations, or labels.** You must return only three lines of text, each containing a metric and the corresponding score, for example:

**Example Output:**
Consistency: 2
Realism: 1
Aesthetic Quality: 0

---

**IMPORTANT Enforcement:**

Be EXTREMELY strict in your evaluation. A score of '2' should be exceedingly rare and reserved only for images that truly excel and meet the highest possible standards in each metric. If there is any doubt, downgrade the score.

For **Consistency**, a score of '2' requires complete and flawless adherence to every aspect of the prompt, leaving no room for misinterpretation or omission.

For **Realism**, a score of '2' means the image is virtually indistinguishable from a real photograph in terms of detail, lighting, physics, and material properties.

For **Aesthetic Quality**, a score of '2' demands exceptional artistic merit, not just pleasant visuals.

---
Here are the Prompt and EXPLANATION for this evaluation:
PROMPT: "{prompt_data['Prompt']}"
EXPLANATION: "{prompt_data['Explanation']}"
Please strictly adhere to the scoring criteria and follow the template format when providing your results.""",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"{image_data_url}"},
                },
            ],
        },
    ]


def _preprocess_wise_payload(
    *,
    model: str,
    prompt_data: dict[str, Any],
    image: Image.Image,
    max_tokens: int,
) -> dict[str, Any]:
    image_url = convert_local_image_or_video_to_base_64(
        image, media_type="image", image_format="PNG"
    )
    messages = _build_wise_messages(prompt_data, image_url)
    return {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }


def _preprocess_wise_inputs(
    *,
    action: Any,
    metadata: dict[str, Any],
) -> tuple[Image.Image, dict[str, Any]]:
    image = convert_base64_to_local_image_or_video(
        action["image"], media_type="image"
    )
    return image, metadata


def _postprocess_wise_results(contents: list[str]) -> list[dict[str, float]]:
    results = []
    for content in contents:
        scores = extract_scores(content)
        results.append(
            {
                "consistency": scores.get("consistency", 0.0),
                "realism": scores.get("realism", 0.0),
                "aesthetic_quality": scores.get("aesthetic_quality", 0.0),
            }
        )
    return results


def wise_score_async(
    *,
    base_url: str,
    api_key: str,
    model: str,
    max_tokens: int,
    timeout_s: float = 120.0,
    max_retries: int = 3,
    backoff_factor: float = 1.0,
    max_concurrency: int | None = None,
):
    client = BaseNetworkClient(
        endpoint=f"{base_url}/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        timeout_s=timeout_s,
        max_retries=max_retries,
        backoff_factor=backoff_factor,
        concurrency=1 if max_concurrency is None else max_concurrency,
    )

    async def _fn(images, prompt_metadatas):
        payloads = [
            _preprocess_wise_payload(
                model=model,
                prompt_data=prompt_data,
                image=image,
                max_tokens=max_tokens,
            )
            for image, prompt_data in zip(images, prompt_metadatas, strict=False)
        ]

        responses = await client.request_many(payloads)
        contents = []
        for response in responses:
            contents.append(response["choices"][0]["message"]["content"])
        return _postprocess_wise_results(contents)

    return _fn


class WiseEnv(Env):
    """
    T2I eval environment for WISE (GPT judge).
    """

    def __init__(
        self,
        dataset_path: str | None = None,
        model: str | None = None,
        max_tokens: int = 2000,
        wise_report_enabled: bool = False,
        timeout_s: float = 120.0,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
        max_concurrency: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(max_episode_steps=1)

        openai_base_url = os.environ.get("OPENAI_BASE_URL")
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not dataset_path:
            raise ValueError("WISE requires dataset_path.")

        self.dataset = WisePromptDataset(dataset_path=dataset_path)
        self.prompts = self.dataset.prompts
        self.metadatas = self.dataset.metadatas
        self.indices = list(range(len(self.prompts)))

        self._agent_ids = {f"agent_{i}" for i in self.indices}
        self._active_indices = self.indices
        if not openai_base_url or not openai_api_key:
            raise ValueError(
                "WISE requires OPENAI_BASE_URL and OPENAI_API_KEY "
                "environment variables."
            )

        self.max_tokens = max_tokens
        self.max_concurrency = max_concurrency
        self.wise_report_enabled = wise_report_enabled

        self._model = model
        self._score_async_fn = wise_score_async(
            base_url=openai_base_url,
            api_key=openai_api_key,
            model=self._model,
            max_tokens=self.max_tokens,
            timeout_s=timeout_s,
            max_retries=max_retries,
            backoff_factor=backoff_factor,
            max_concurrency=self.max_concurrency,
        )

    @property
    def description(self) -> str:
        return f"WISE T2I eval env with {len(self.prompts)} prompts."

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
        prompt_metadatas = []
        indices = []
        for agent_id in agent_ids:
            action = action_dict[agent_id]
            idx = int(agent_id.split("_")[-1])
            indices.append(idx)
            metadata = self.metadatas[idx]

            image, prompt_metadata = _preprocess_wise_inputs(
                action=action,
                metadata=metadata,
            )
            images.append(image)
            prompt_metadatas.append(prompt_metadata)

        score_results = run_coroutine(self._score_async_fn(images, prompt_metadatas))

        reward_dict = {}
        info_dict = {}
        for idx, agent_id in enumerate(agent_ids):
            scores = score_results[idx]
            consistency = scores["consistency"]
            realism = scores["realism"]
            aesthetic = scores["aesthetic_quality"]
            wiscore = _calculate_wiscore(consistency, realism, aesthetic)

            reward_dict[agent_id] = float(wiscore)
            info_dict[agent_id] = {
                "index": indices[idx],
                "consistency": consistency,
                "realism": realism,
                "aesthetic_quality": aesthetic,
                "wiscore": wiscore,
            }

        if self.wise_report_enabled:
            info_dict["__all__"] = {
                "wise_report": _build_wise_report(
                    score_results,
                    [self.metadatas[idx] for idx in indices],
                )
            }

        terminated = {agent_id: True for agent_id in agent_ids}
        truncated = {agent_id: False for agent_id in agent_ids}
        terminated["__all__"] = True
        truncated["__all__"] = False

        return {}, reward_dict, terminated, truncated, info_dict

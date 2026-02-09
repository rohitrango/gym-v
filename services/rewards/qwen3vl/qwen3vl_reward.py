from __future__ import annotations

import re
from typing import Any, Optional

import torch

from gym_v.logger import get_logger
from pydantic import BaseModel, Field
from services.rewards.base import BaseReward
from services.rewards.registry import register_reward
from services.rewards.qwen3vl.processing_utils import process_vision_info

logger = get_logger()

class GenerateResponseAPI(BaseModel):
    """API response model for image/video generation."""

    texts: Optional[list[str]] = Field(None, description="Response Text")
    Scores: Optional[list[float] ]= Field(None, description="Response Scores")


def _build_processor_kwargs(mm_inputs_list: list[dict[str, Any]]) -> dict[str, Any]:
    images_list: list[Any] = []
    videos_list: list[Any] = []
    has_images = False
    has_videos = False
    for mm_inputs in mm_inputs_list:
        images = mm_inputs.get("images") or mm_inputs.get("image") or []
        videos = mm_inputs.get("videos") or mm_inputs.get("video") or []
        images_list.append(images)
        videos_list.append(videos)
        if images:
            has_images = True
        if videos:
            has_videos = True
    processor_kwargs: dict[str, Any] = {}
    if has_images:
        processor_kwargs["images"] = images_list
    if has_videos:
        processor_kwargs["videos"] = videos_list
    return processor_kwargs


def _build_messages(
    data: dict[str, Any],
    prompt_key: str,
    as_conversation: bool,
    multimodal_keys: list[str] | None = None,
):
    prompt = data.get(prompt_key)

    if isinstance(prompt, str):
        if not as_conversation:
            return prompt
        prompt = [{"role": "user", "content": prompt}]
    elif isinstance(prompt, dict):
        prompt = [prompt]

    if multimodal_keys:
        multimodal_inputs = data.get("multimodal_inputs") or {}
        multimodals: dict[str, tuple[str, list[Any]]] = {}
        for key in multimodal_keys:
            content = multimodal_inputs.get(key)
            if content is None:
                continue
            items = list(content) if isinstance(content, list | tuple) else [content]
            multimodals[f"<{key}>"] = (key, items)

        if multimodals:
            pattern = "(" + "|".join(re.escape(p) for p in multimodals.keys()) + ")"

            for message in prompt:
                if isinstance(message["content"], str):
                    content_list = []
                    inserted = False
                    for segment in re.split(pattern, message["content"]):
                        if not segment:
                            continue
                        if segment in multimodals:
                            key, content = multimodals[segment]
                            content_list.append({"type": key, key: content.pop(0)})
                            inserted = True
                        else:
                            content_list.append({"type": "text", "text": segment})
                    if not inserted and multimodals:
                        for key, content in multimodals.values():
                            if content:
                                content_list.insert(
                                    len(content_list) - 1,
                                    {"type": key, key: content.pop(0)},
                                )
                    message["content"] = content_list
                elif isinstance(message["content"], list):
                    for item in message["content"]:
                        if isinstance(item, dict) and item.get("type") == "image_url":
                            image_url = item.get("image_url") or {}
                            if isinstance(image_url, dict) and "url" in image_url:
                                item["type"] = "image"
                                item["image"] = image_url["url"]
                                item.pop("image_url", None)
                    logger.warning(
                        "message['content'] is a list of dicts, no processing will be done."
                    )

    return prompt


def _extract_answer_scores(
    *,
    scores,
    idx: int,
    answer_list: list[str] | None,
    tokenizer,
) -> list[float] | None:
    
    if answer_list:
        step_scores = scores[0]
        probs = torch.nn.functional.softmax(step_scores, dim=-1)
        lm_prob = []
        for answer in answer_list:
            token_ids = tokenizer.encode(answer, add_special_tokens=False)
            if not token_ids:
                continue
            token_id = token_ids[0]
            lm_prob.append(float(probs[idx, token_id].item()))
        return lm_prob
    else:
        return [step[idx].detach().cpu().tolist() for step in scores]


class Qwen3VLReward(BaseReward):
    """
    Local Qwen3-VL inference reward.

    Accepts arbitrary prompt + image and returns model outputs via the reward server.
    """

    def __init__(
        self,
        *,
        model: str = "Qwen/Qwen3-VL-8B-Instruct",
        dtype: torch.dtype = None,
        device: torch.device | str = "cuda",
    ) -> None:
        super().__init__(device=device)

        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(model)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(model, dtype=dtype)
        self.model = self.model.to(self.device)
        self.model.eval()
        self._tokenizer = getattr(self.processor, "tokenizer", None)

    def __call__(self, samples):
        sample_list = list(samples or [])
        messages_list: list[Any] = []
        metadatas: list[Any] = []
        multimodal_inputs_list: list[dict[str, Any]] = []
        for sample in sample_list:
            data = sample if isinstance(sample, dict) else sample.to_dict()
            mm_inputs = data.get("multimodal_inputs", None)
            messages = _build_messages(
                data,
                prompt_key="prompt",
                as_conversation=True,
                multimodal_keys=list(mm_inputs.keys()) if mm_inputs else None,
            )
            if self.processor is not None:
                mm_inputs = process_vision_info(messages, self.processor)
            multimodal_inputs_list.append(mm_inputs)
            metadatas.append(data.get("metadata"))
            messages_list.append(messages)

        gen_cfg: dict[str, Any] = {}
        if metadatas:
            metadata = metadatas[0]
            if isinstance(metadata, dict):
                gen_cfg = dict(metadata.get("gen") or metadata.get("generation") or {})

        max_new_tokens = int(gen_cfg["max_tokens"])
        temperature = float(gen_cfg["temperature"])
        logprobs = bool(gen_cfg["logprobs"])

        messages = messages_list
        texts = [
            self.processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True
            )
            for msg in messages
        ]
        processor_kwargs = _build_processor_kwargs(multimodal_inputs_list)

        inputs = self.processor(
            text=texts,
            padding=True,
            return_tensors="pt",
            **processor_kwargs,
        )
        inputs = inputs.to(self.model.device)

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0.0,
            "temperature": temperature,
        }
        if logprobs:
            generate_kwargs["output_scores"] = True
            generate_kwargs["return_dict_in_generate"] = True

        with torch.inference_mode():
            generated = self.model.generate(**inputs, **generate_kwargs)

        if logprobs:
            sequences = generated.sequences
            scores = generated.scores
        else:
            sequences = generated
            scores = None

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, sequences, strict=False)
        ]
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        results: list[dict[str, Any]] = []
        for idx, text in enumerate(output_texts):
            metadata = metadatas[idx]
            answer_list = metadata.get("answer_list", None)
            score_list = None
            if logprobs and scores:
                score_list = _extract_answer_scores(
                    scores=scores,
                    idx=idx,
                    answer_list=answer_list,
                    tokenizer=self._tokenizer,
                )
            item = GenerateResponseAPI(texts=[text], Scores=score_list)
            results.append(item.model_dump(exclude_none=True))

        return results


@register_reward("qwen3vl")
def build_qwen3vl_reward(**kwargs: Any) -> BaseReward:
    return Qwen3VLReward(**kwargs)

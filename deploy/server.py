from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse
from ray import serve

from deploy.utils import convert_base64_to_local_image_or_video
from services.rewards import build_rm_model_manager


@dataclass
class Sample:
    """The sample generated."""

    prompt: str | list[dict[str, str]] = ""
    multimodal_inputs: dict[str, Any] | None = None
    multimodal_train_inputs: dict[str, Any] | None = None
    response: str = ""
    multimodal_outputs: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    train_metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Sample:
        data = dict(data)
        field_names = set(Sample.__dataclass_fields__.keys())
        init_data = {key: value for key, value in data.items() if key in field_names}
        sample = Sample(**init_data)

        for key, value in data.items():
            if key not in field_names:
                setattr(sample, key, value)

        return sample


def _convert_sample_images(sample: Sample) -> Sample:
    data = sample.to_dict()
    for key in ("multimodal_outputs", "multimodal_inputs", "multimodal_train_inputs"):
        container = data.get(key)
        if isinstance(container, dict):
            if "image" in container:
                container["image"] = convert_base64_to_local_image_or_video(
                    container["image"], media_type="image"
                )
            if "images" in container:
                container["images"] = convert_base64_to_local_image_or_video(
                    container["images"], media_type="image"
                )
            if "video" in container:
                container["video"] = convert_base64_to_local_image_or_video(
                    container["video"], media_type="video"
                )
            if "videos" in container:
                container["videos"] = convert_base64_to_local_image_or_video(
                    container["videos"], media_type="video"
                )
    return Sample.from_dict(data)


def _normalize_scores(scores: Any, size: int) -> list[Any]:
    if isinstance(scores, list):
        return scores
    if isinstance(scores, dict):
        items: list[dict[str, Any]] = [dict() for _ in range(size)]
        for key, value in scores.items():
            if isinstance(value, list | tuple) and len(value) == size:
                for idx in range(size):
                    items[idx][key] = value[idx]
            else:
                for idx in range(size):
                    items[idx][key] = value
        return items
    return [scores for _ in range(size)]


def build_reward_service(
    *,
    max_batch_size: int = 32,
    batch_wait_timeout_s: float = 0.01,
    max_ongoing_requests: int = 5120,
):
    app = FastAPI()

    @serve.deployment(max_ongoing_requests=max_ongoing_requests)
    @serve.ingress(app)
    class RewardService:
        def __init__(self, score_dict: dict[str, Any], device: str = "cuda"):
            self.rm_model_manager = build_rm_model_manager(score_dict)

        def _score_samples(self, payloads: list[dict[str, Any]]) -> list[Any]:
            samples = [
                _convert_sample_images(Sample.from_dict(item)) for item in payloads
            ]
            scores, _ = self.rm_model_manager(samples)
            return _normalize_scores(scores, len(samples))

        @app.get("/health")
        async def health(self):
            return Response(status_code=200, content="OK")

        @app.get("/get_model_info")
        async def get_model_info(self):
            if hasattr(self.rm_model_manager, "get_model_info"):
                return JSONResponse(content=self.rm_model_manager.get_model_info())
            return JSONResponse(content={})

        @app.post(
            "/v1/generate",
            response_model=dict[str, Any],
            response_model_exclude_none=True,
        )
        async def score_sample(self, payload: dict[str, Any]):
            result = await self.score_sample_batch(payload)
            if "error" in result:
                raise HTTPException(status_code=400, detail=result["error"])
            return result

        @serve.batch(
            max_batch_size=max_batch_size, batch_wait_timeout_s=batch_wait_timeout_s
        )
        async def score_sample_batch(
            self, payloads: list[dict[str, Any]]
        ) -> list[dict[str, Any]]:
            return self._score_samples(payloads)

    return RewardService

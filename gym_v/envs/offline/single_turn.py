"""A generic single-turn environment backed by offline examples."""

from __future__ import annotations

from typing import Any

from PIL import Image

from gym_v import Env, Observation, get_logger
from gym_v.envs.offline.grader import GRADER_REGISTRY
from gym_v.envs.offline.sampler import IndexSampler
from gym_v.envs.offline.source import DATASOURCE_REGISTRY, DataSource, OfflineSample

logger = get_logger()


class OfflineSingleTurnEnv(Env):
    """Single-turn QA environment backed by offline data.

    Samples (image, text, answer) from an offline dataset.
    Each episode presents one sample; the agent responds and receives a graded reward.

    Args:
        datasource_type: Data source type (e.g., "jsonl")
        datasource_kwargs: Arguments for the data source constructor
        grader: Grading function name (default: "exact_match")
        description: Custom environment description
        shuffle: Whether to shuffle samples each epoch
    """

    def __init__(
        self,
        datasource_type: str,
        datasource_kwargs: dict[str, Any],
        grader: str = "exact_match",
        description: str | None = None,
        shuffle: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self._source: DataSource = self._build_source(
            datasource_type, datasource_kwargs
        )
        self._grader_name = grader
        self._grader = GRADER_REGISTRY.get(self._grader_name)
        if self._grader is None:
            raise ValueError(
                f"unknown grader={self._grader_name}, expected: {list(GRADER_REGISTRY)}"
            )

        self._description = description

        self._sampler = IndexSampler(len(self._source), shuffle=shuffle)
        self._current_sample: OfflineSample | None = None
        self._current_index: int | None = None

    def _build_source(
        self, datasource_type: str, datasource_kwargs: dict[str, Any]
    ) -> DataSource:
        """Create a data source from type and kwargs."""
        source_cls = DATASOURCE_REGISTRY.get(datasource_type)
        if source_cls is None:
            raise ValueError(
                f"unknown datasource_type={datasource_type}, expected: {list(DATASOURCE_REGISTRY)}"
            )
        return source_cls(**datasource_kwargs)

    @property
    def description(self) -> str:
        if self._description:
            return self._description
        return (
            "Single-turn QA environment.\n"
            "Observe the image/text, respond with an answer.\n"
            f"Grader: {self._grader_name}"
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self._sampler.reset(seed)

        idx = self._sampler()
        sample = self._source[idx]
        self._current_index = idx
        self._current_sample = sample

        obs = Observation(
            image=sample.image,
            text=sample.text,
            metadata=sample.metadata or {},
        )
        info = {
            "index": idx,
            "oracle_answer": sample.answer,
            "grader": self._grader_name,
        }
        return obs, info

    def inner_step(
        self, action: str
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        sample = self._current_sample
        reward, reward_extra_info = self._grader(action, sample.answer)

        obs = Observation(
            image=sample.image,
            text=sample.text,
            metadata=sample.metadata or {},
        )
        info = {
            "index": self._current_index,
            "oracle_answer": sample.answer,
            "grader": self._grader_name,
            **reward_extra_info,
        }

        return obs, reward, True, False, info

    def render(self) -> Image.Image | list[Image.Image] | None:
        if self._current_sample.image is None:
            raise RuntimeError("current sample has no image")
        return self._current_sample.image

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

    Multi-agent environment compatible with Ray RLlib API.
    Each agent receives a different sample from the dataset in parallel.

    Args:
        datasource_type: Data source type (e.g., "jsonl")
        datasource_kwargs: Arguments for the data source constructor
        grader: Grading function name (default: "exact_match")
        description: Custom environment description
        shuffle: Whether to shuffle samples each epoch
        num_players: Number of parallel agents (default: 16)
    """

    def __init__(
        self,
        datasource_type: str,
        datasource_kwargs: dict[str, Any],
        grader: str = "exact_match",
        description: str | None = None,
        shuffle: bool = True,
        num_players: int = 16,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.num_players = num_players
        self._agent_ids = [f"agent_{i}" for i in range(num_players)]

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
        # Each agent has its own sample
        self._current_samples: dict[str, OfflineSample] = {}
        self._current_indices: dict[str, int] = {}

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
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self._sampler.reset(seed)

        obs_dict = {}
        info_dict = {}

        for agent_id in self._agent_ids:
            idx = self._sampler()
            sample = self._source[idx]
            self._current_indices[agent_id] = idx
            self._current_samples[agent_id] = sample

            obs_dict[agent_id] = Observation(
                image=sample.image,
                text=sample.text,
                metadata=sample.metadata or {},
            )
            info_dict[agent_id] = {
                "index": idx,
                "oracle_answer": sample.answer,
                "grader": self._grader_name,
            }

        return obs_dict, info_dict

    def inner_step(
        self, action: dict[str, str]
    ) -> tuple[
        dict[str, Observation],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, Any],
    ]:
        obs_dict = {}
        reward_dict = {}
        info_dict = {}

        # Grade each agent's action
        for agent_id in self._agent_ids:
            sample = self._current_samples[agent_id]
            reward, reward_extra_info = self._grader(action[agent_id], sample.answer)

            obs_dict[agent_id] = Observation(
                image=sample.image,
                text=sample.text,
                metadata=sample.metadata or {},
            )
            reward_dict[agent_id] = reward
            info_dict[agent_id] = {
                "index": self._current_indices[agent_id],
                "oracle_answer": sample.answer,
                "grader": self._grader_name,
                **reward_extra_info,
            }

        # Single-turn environment: both terminated and truncated are True
        # - terminated: task is complete (answered the question)
        # - truncated: episode ends after one step (no more interactions)
        terminated = True
        truncated = True

        return (
            obs_dict,
            reward_dict,
            {
                **{agent_id: terminated for agent_id in self._agent_ids},
                "__all__": terminated,
            },
            {
                **{agent_id: truncated for agent_id in self._agent_ids},
                "__all__": truncated,
            },
            info_dict,
        )

    def render(self) -> list[Image.Image]:
        """Return list of images for all agents (in agent_id order)."""
        if not self._current_samples:
            raise RuntimeError("no current samples, call reset() first")

        images = []
        for agent_id in self._agent_ids:
            sample = self._current_samples[agent_id]
            if sample.image is None:
                raise RuntimeError(f"agent {agent_id} has no image in current sample")
            images.append(sample.image)
        return images

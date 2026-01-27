"""History recorder wrapper for gym-v environments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from gym_v.core import Env, Observation, Wrapper
from gym_v.utils import RecordConstructorArgs


@dataclass
class StepRecord:
    """A single step record containing observation and action."""

    observation: Observation
    action: str | None = None


@dataclass
class AgentHistory:
    """History of observations and actions for a single agent."""

    records: list[StepRecord] = field(default_factory=list)

    def __len__(self) -> int:
        """Return the number of records (turns)."""
        return len(self.records)

    def add_observation(self, obs: Observation) -> None:
        """Add a new observation (starts a new record)."""
        self.records.append(StepRecord(observation=obs))

    def add_action(self, action: str) -> None:
        """Add action to the most recent record."""
        assert self.records, "Cannot add action: no observation recorded yet"
        assert (
            self.records[-1].action is None
        ), f"Cannot add action: last record already has action '{self.records[-1].action}'"
        self.records[-1].action = action

    def prune(self, target_turns: int) -> None:
        """Prune history from the beginning, keeping the most recent records.

        Args:
            target_turns: Target number of turns to keep
        """
        if len(self.records) > target_turns:
            self.records = self.records[-target_turns:]

    def clear(self) -> None:
        """Clear all history."""
        self.records.clear()

    def to_dict(self) -> list[dict[str, Any]]:
        """Convert history to list of dicts."""
        return [
            {"observation": r.observation, "action": r.action} for r in self.records
        ]


class HistoryRecorder(Wrapper, RecordConstructorArgs):
    """Wrapper that records state-action history and returns it in info.

    This wrapper maintains a history of all observations and actions for each agent.
    The history is returned in the info dict under the key "history".

    The history structure is:
        info[agent_id]["history"] = [
            {"observation": Observation, "action": str | None},
            {"observation": Observation, "action": str | None},
            ...
        ]

    Note: The last record may have action=None if we just received an observation
    but haven't taken an action yet (e.g., after reset).
    """

    def __init__(
        self,
        env: Env,
        include_history_in_info: bool = True,
        max_turns: int | None = None,
        target_turns: int | None = None,
    ):
        """Initialize the history recorder wrapper.

        Args:
            env: The environment to wrap
            include_history_in_info: Whether to include history in info dict
            max_turns: Maximum turns before pruning is triggered (None = no limit)
            target_turns: Target turns to keep after pruning (defaults to max_turns)
        """
        RecordConstructorArgs.__init__(
            self,
            include_history_in_info=include_history_in_info,
            max_turns=max_turns,
            target_turns=target_turns,
        )
        Wrapper.__init__(self, env)
        self.include_history_in_info = include_history_in_info
        self.max_turns = max_turns
        self.target_turns = target_turns if target_turns is not None else max_turns
        self._history: dict[str, AgentHistory] = {}

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        """Reset the environment and clear history.

        Args:
            seed: Optional seed for the environment's random number generator
            options: Optional dictionary of reset options

        Returns:
            observation: Dictionary of initial observations {agent_id: Observation}
            info: Dictionary of infos {agent_id: dict}
        """
        self._history.clear()
        obs, info = self.env.reset(seed=seed, options=options)

        # Record initial observations
        for agent_id, observation in obs.items():
            self._history[agent_id] = AgentHistory()
            self._history[agent_id].add_observation(observation)

        # Add history to info if enabled
        if self.include_history_in_info:
            for agent_id in obs.keys():
                if agent_id not in info:
                    info[agent_id] = {}
                info[agent_id]["history"] = self._history[agent_id].to_dict()

        return obs, info

    def step(
        self, action: dict[str, str]
    ) -> tuple[
        dict[str, Observation],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, Any],
    ]:
        """Execute action and record state-action pair.

        Args:
            action: Dictionary of actions {agent_id: action_string}

        Returns:
            observation: Dictionary of observations {agent_id: Observation}
            reward: Dictionary of rewards {agent_id: float}
            terminated: Dictionary of terminated flags {agent_id: bool, "__all__": bool}
            truncated: Dictionary of truncated flags {agent_id: bool, "__all__": bool}
            info: Dictionary of infos with history {agent_id: dict}
        """
        # Record actions for agents that are acting this turn
        for agent_id, act in action.items():
            if agent_id in self._history:
                self._history[agent_id].add_action(act)

        obs, reward, terminated, truncated, info = self.env.step(action)

        # Record new observations
        for agent_id, observation in obs.items():
            if agent_id not in self._history:
                self._history[agent_id] = AgentHistory()
            self._history[agent_id].add_observation(observation)

        # Prune if exceeds max_turns
        if self.max_turns is not None:
            for _, history in self._history.items():
                if len(history) > self.max_turns:
                    history.prune(self.target_turns)

        # Add history to info if enabled
        if self.include_history_in_info:
            for agent_id in obs.keys():
                if agent_id not in info:
                    info[agent_id] = {}
                info[agent_id]["history"] = self._history[agent_id].to_dict()

        return obs, reward, terminated, truncated, info

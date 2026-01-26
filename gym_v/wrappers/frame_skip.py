"""Frame skip wrapper for gym-v environments."""

from __future__ import annotations

from typing import Any

import numpy as np

from gym_v.core import Env, Observation, Wrapper
from gym_v.utils import RecordConstructorArgs


class StochasticFrameSkip(Wrapper, RecordConstructorArgs):
    """Stochastic frame skip wrapper that repeats actions with sticky probability.

    This wrapper implements the stochastic frame skipping technique where:
    - Actions are repeated for n frames
    - With probability `stickprob`, the previous action "sticks" on the first substep
    - From the second substep onwards, the new action is always used

    This is commonly used in Atari environments to add stochasticity.
    """

    def __init__(self, env: Env, n: int, stickprob: float):
        """Initialize the stochastic frame skip wrapper.

        Args:
            env: The environment to wrap
            n: Number of frames to skip (action repeat count)
            stickprob: Probability that the previous action sticks on first substep
        """
        RecordConstructorArgs.__init__(self, n=n, stickprob=stickprob)
        Wrapper.__init__(self, env)
        self.n = n
        self.stickprob = stickprob
        self.curac: dict[str, str] | None = None
        self.rng = np.random.RandomState()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        """Reset the environment and clear sticky action state.

        Args:
            seed: Optional seed for the environment's random number generator
            options: Optional dictionary of reset options

        Returns:
            observation: Dictionary of initial observations {agent_id: Observation}
            info: Dictionary of infos {agent_id: dict}
        """
        self.curac = None
        return self.env.reset(seed=seed, options=options)

    def step(
        self, action: dict[str, str]
    ) -> tuple[
        dict[str, Observation],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, Any],
    ]:
        """Execute action with stochastic frame skipping.

        Args:
            action: Dictionary of actions {agent_id: action_string}

        Returns:
            observation: Dictionary of observations {agent_id: Observation}
            reward: Dictionary of accumulated rewards {agent_id: float}
            terminated: Dictionary of terminated flags {agent_id: bool, "__all__": bool}
            truncated: Dictionary of truncated flags {agent_id: bool, "__all__": bool}
            info: Dictionary of infos {agent_id: dict}
        """
        obs: dict[str, Observation] = {}
        terminated: dict[str, bool] = {}
        truncated: dict[str, bool] = {}
        info: dict[str, Any] = {}
        totrew: dict[str, float] = {}

        for i in range(self.n):
            # Determine the actual action to use
            if self.curac is None:
                # First step after reset, use the provided action
                self.curac = dict(action)
            elif i == 0:
                # First substep: with probability stickprob, keep previous action
                # Only apply sticky logic to agents that acted previously
                for agent_id, act in action.items():
                    if agent_id in self.curac:
                        if self.rng.rand() > self.stickprob:
                            self.curac[agent_id] = act
                    else:
                        # New agent that wasn't in previous action
                        self.curac[agent_id] = act
            elif i == 1:
                # Second substep: new action definitely kicks in
                self.curac.update(action)

            # Use current action dict, but only pass agents that are in this turn's action
            current_action = {
                agent_id: self.curac.get(agent_id, act)
                for agent_id, act in action.items()
            }

            obs, rew, terminated, truncated, info = self.env.step(current_action)

            # Accumulate rewards
            for agent_id, r in rew.items():
                totrew[agent_id] = totrew.get(agent_id, 0.0) + r

            # Break early if episode ends
            if terminated.get("__all__", False) or truncated.get("__all__", False):
                break

        return obs, totrew, terminated, truncated, info

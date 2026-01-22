"""Gin Rummy game using PettingZoo."""

from __future__ import annotations

from collections import defaultdict
from textwrap import dedent
from typing import Any

from PIL import Image
from typing_extensions import override

from gym_v import Env, Observation, get_logger
from gym_v.envs.multi_players.pettingzoo.utils import TerminateIllegalOutOfBoundsWrapper
from pettingzoo.classic import gin_rummy_v4

logger = get_logger()


class PettingZooGinRummy(Env):
    """
    Gin Rummy card game using PettingZoo's gin_rummy environment.

    Two players draw and discard cards to form melds and reduce deadwood.
    """

    is_deterministic = False

    def __init__(
        self,
        num_players: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if num_players != 2:
            raise ValueError(
                f"{self.__class__.__name__} only supports 2 players, got {num_players}"
            )

        env = gin_rummy_v4.raw_env(render_mode="rgb_array")
        env = TerminateIllegalOutOfBoundsWrapper(env)
        self._pz_env = env

        self._agent_ids = {"player_0", "player_1"}

    @override
    @property
    def description(self) -> dict[str, str]:
        base_description = dedent("""
            You are playing <<Gin Rummy>> as Player {player_id}.

            Draw and discard cards to form melds (sets or runs) and minimize deadwood.
            Knock when your deadwood is 10 or less, or go Gin with 0 deadwood.

            Action format: Provide an action number from 0 to 109.
            Actions encode draw source, discard choice, knock/gin decisions.
        """).strip()
        return {
            "player_0": base_description.format(player_id="0"),
            "player_1": base_description.format(player_id="1"),
        }

    def _get_current_observation(self) -> Observation:
        """Get observation for the current player."""
        return Observation(image=self.render(), text=None)

    def _get_pz_action(self, action: str) -> int:
        """Convert action string to PettingZoo action."""
        return int(action.strip())

    @override
    def inner_step(
        self, action: dict[str, str]
    ) -> tuple[
        dict[str, Observation],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, Any],
    ]:
        obs_map: dict[str, Observation] = {}
        reward_map: dict[str, float] = defaultdict(float)
        terminated_map: dict[str, bool] = defaultdict(bool)
        truncated_map: dict[str, bool] = defaultdict(bool)
        info_map: dict[str, dict[str, Any]] = defaultdict(dict)
        info_map["__all__"] = {}

        acting_player = self._pz_env.agent_selection
        action_str = action[acting_player]
        pz_action = self._get_pz_action(action_str)

        self._pz_env.step(pz_action)

        while self._pz_env.agents:
            _, reward, terminated, truncated, info = self._pz_env.last()
            current_player = self._pz_env.agent_selection
            obs_map[current_player] = self._get_current_observation()
            reward_map[current_player] = reward
            terminated_map[current_player] = terminated
            truncated_map[current_player] = truncated
            info_map[current_player] = info

            if terminated or truncated:
                self._pz_env.step(None)
            else:
                break

        all_gone = not self._pz_env.agents
        terminated_map["__all__"] = all_gone and all(terminated_map.values())
        truncated_map["__all__"] = all_gone and all(truncated_map.values())

        # Add text feedback based on game state
        made_invalid_action = False
        made_invalid_action_players = []
        for pid in obs_map:
            if info_map.get(pid, {}).get("invalid_action", False):
                made_invalid_action = True
                made_invalid_action_players.append(pid)

        for pid in obs_map:
            if made_invalid_action:
                if pid in made_invalid_action_players:
                    obs_map[pid] = Observation(
                        image=obs_map[pid].image,
                        text="You made an invalid action.",
                        metadata=obs_map[pid].metadata,
                    )
                elif terminated_map.get(pid, False) or truncated_map.get(pid, False):
                    obs_map[pid] = Observation(
                        image=obs_map[pid].image,
                        text=f"Game terminated due to {', '.join(made_invalid_action_players)} made invalid actions.",
                        metadata=obs_map[pid].metadata,
                    )
            else:
                if terminated_map.get(pid, False) or truncated_map.get(pid, False):
                    if reward_map[pid] > 0:
                        obs_map[pid] = Observation(
                            image=obs_map[pid].image,
                            text="You win!",
                            metadata=obs_map[pid].metadata,
                        )
                    elif reward_map[pid] < 0:
                        obs_map[pid] = Observation(
                            image=obs_map[pid].image,
                            text="You lose!",
                            metadata=obs_map[pid].metadata,
                        )
                    else:
                        obs_map[pid] = Observation(
                            image=obs_map[pid].image,
                            text="Draw!",
                            metadata=obs_map[pid].metadata,
                        )

        return obs_map, reward_map, terminated_map, truncated_map, info_map

    @override
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        super().reset(seed=seed)

        self._pz_env.reset(seed=seed, options=options)
        _, _, _, _, info = self._pz_env.last()

        obs_map = {self._pz_env.agent_selection: self._get_current_observation()}
        info_map = {self._pz_env.agent_selection: info}
        info_map["__all__"] = {}

        logger.info("Reset PettingZoo Gin Rummy")

        return obs_map, info_map

    @override
    def render(self) -> Image.Image | list[Image.Image] | None:
        """Render the Gin Rummy game state."""
        return Image.fromarray(self._pz_env.render())

    @override
    def close(self):
        """Clean up resources."""
        self._pz_env.close()

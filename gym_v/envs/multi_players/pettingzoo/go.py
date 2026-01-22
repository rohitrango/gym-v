"""Go game using PettingZoo."""

from __future__ import annotations

from collections import defaultdict
from textwrap import dedent
from typing import Any

from PIL import Image
from typing_extensions import override

from gym_v import Env, Observation, get_logger
from gym_v.envs.multi_players.pettingzoo.utils import TerminateIllegalOutOfBoundsWrapper
from pettingzoo.classic import go_v5

logger = get_logger()


class PettingZooGo(Env):
    """
    Go game using PettingZoo's go environment.

    Two players take turns placing stones on a board to surround territory.
    """

    is_deterministic = False

    def __init__(
        self,
        num_players: int = 2,
        board_size: int = 19,
        komi: float = 7.5,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if num_players != 2:
            raise ValueError(
                f"{self.__class__.__name__} only supports 2 players, got {num_players}"
            )

        self._board_size = board_size
        env = go_v5.raw_env(board_size=board_size, komi=komi, render_mode="rgb_array")
        env = TerminateIllegalOutOfBoundsWrapper(env)
        self._pz_env = env

        self._agent_ids = {"black_0", "white_0"}

    @override
    @property
    def description(self) -> dict[str, str]:
        # Generate column labels (A-T, skipping I)
        cols = "ABCDEFGHJKLMNOPQRST"[: self._board_size]
        base_description = dedent("""
            You are playing <<Go>> as {color} on a {board_size}x{board_size} board.

            Place stones to surround territory and capture opponent's stones.
            The game ends when both players pass consecutively.

            Action format: Provide coordinates like "{example}" or "pass".
            Columns: {cols} (left to right), Rows: 1-{board_size} (bottom to top).
        """).strip()
        return {
            "black_0": base_description.format(
                color="Black",
                board_size=self._board_size,
                cols=cols,
                example="D4" if self._board_size >= 4 else "A1",
            ),
            "white_0": base_description.format(
                color="White",
                board_size=self._board_size,
                cols=cols,
                example="D4" if self._board_size >= 4 else "A1",
            ),
        }

    def _get_current_observation(self) -> Observation:
        """Get observation for the current player."""
        return Observation(image=self.render(), text=None)

    def _get_pz_action(self, action: str) -> int:
        """Convert coordinate notation to PettingZoo action."""
        action = action.strip().lower()
        if action == "pass":
            return self._board_size * self._board_size

        # Parse coordinate like "D4" -> col=3, row=3
        col_letter = action[0].upper()
        row_num = int(action[1:])

        # Column mapping: A=0, B=1, ..., H=7, J=8, ... (skip I)
        cols = "ABCDEFGHJKLMNOPQRST"
        col = cols.index(col_letter)
        row = row_num - 1  # Convert 1-indexed to 0-indexed

        return row * self._board_size + col

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

        logger.info("Reset PettingZoo Go")

        return obs_map, info_map

    @override
    def render(self) -> Image.Image | list[Image.Image] | None:
        """Render the Go board."""
        return Image.fromarray(self._pz_env.render())

    @override
    def close(self):
        """Clean up resources."""
        self._pz_env.close()

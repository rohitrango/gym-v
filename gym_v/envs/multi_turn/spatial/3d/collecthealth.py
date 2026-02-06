"""MiniWorld CollectHealth environment wrapper."""

from __future__ import annotations

from .miniworld_base import MiniWorldBaseEnv


class CollectHealthEnv(MiniWorldBaseEnv):
    # Meta: source=MiniWorld, category=spatial, turn=multi
    # Overrides: visual_complexity=3d_rendering
    """Collect health packs while avoiding lava.

    The agent must navigate through an environment with lava obstacles to
    collect health packs. Touching lava causes damage, so careful navigation
    is required to maximize health pack collection while staying safe.

    Available actions: turn_left, turn_right, move_forward
    """

    def __init__(self, **kwargs):
        super().__init__(env_id="MiniWorld-CollectHealth-v0", **kwargs)

    def get_task_description(self) -> str:
        return (
            "GOAL: Collect as many HEALTH PACKS as possible while avoiding LAVA. "
            "The environment contains health packs (items to collect) and lava obstacles (dangerous areas that damage you). "
            "STRATEGY: (1) SCAN the area by turning left or right to identify safe paths and locate health packs, "
            "(2) Identify LAVA AREAS (typically red/orange colored dangerous zones) that you must NOT step on, "
            "(3) Plan a safe path to reach health packs by moving forward along areas that are NOT lava, "
            "(4) Walk over health packs to automatically collect them, "
            "(5) If you accidentally touch lava, you'll take damage, so navigate carefully around it, "
            "(6) Continue collecting health packs while maintaining your health above zero. "
            "SUCCESS: Maximize health pack collection while staying alive. The episode ends if your health reaches zero from lava damage."
        )

    def get_available_actions(self) -> list[str]:
        return ["turn_left", "turn_right", "move_forward"]

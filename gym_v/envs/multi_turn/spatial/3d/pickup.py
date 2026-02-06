"""MiniWorld PickupObjects environment wrapper."""

from __future__ import annotations

from .miniworld_base import MiniWorldBaseEnv


class PickupObjectsEnv(MiniWorldBaseEnv):
    # Meta: source=MiniWorld, category=spatial, turn=multi
    # Overrides: visual_complexity=3d_rendering
    """Navigate and pick up boxes of the target color.

    The agent must navigate through the environment to locate and collect
    boxes that match the target color. This combines navigation with object
    manipulation and requires identifying the correct objects.

    Available actions: turn_left, turn_right, move_forward, move_back, pickup
    """

    def __init__(self, **kwargs):
        super().__init__(env_id="MiniWorld-PickupObjects-v0", **kwargs)

    def get_task_description(self) -> str:
        return (
            "GOAL: Find and pick up ALL boxes that match the TARGET COLOR. "
            "The environment contains multiple colored boxes scattered around. You need to collect only the boxes of a specific target color. "
            "STRATEGY: (1) IDENTIFY the target color you need to collect (this will be specified), "
            "(2) EXPLORE by turning left or right to scan for boxes, moving forward or backward to navigate, "
            "(3) When you spot a box, check its color visually, "
            "(4) If it matches the target color, walk up close to it until you're standing directly next to it, "
            "(5) Use the PICKUP action to collect the box, "
            "(6) Continue exploring to find and collect all remaining target-colored boxes. "
            "SUCCESS: You win when you've picked up all boxes of the target color. Do NOT pick up boxes of other colors."
        )

    def get_available_actions(self) -> list[str]:
        return ["turn_left", "turn_right", "move_forward", "move_back", "pickup"]

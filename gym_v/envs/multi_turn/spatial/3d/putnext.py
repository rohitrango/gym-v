"""MiniWorld PutNext environment wrapper."""

from __future__ import annotations

from .miniworld_base import MiniWorldBaseEnv


class PutNextEnv(MiniWorldBaseEnv):
    # Meta: source=MiniWorld, category=spatial, turn=multi
    # Overrides: visual_complexity=3d_rendering
    """Pick up a box and place it next to the target box.

    The agent must locate a movable box, pick it up, navigate to the target
    box, and place the carried box next to it. This requires sequential
    manipulation, navigation, and spatial reasoning.

    Available actions: turn_left, turn_right, move_forward, move_back, pickup, drop, toggle, done
    """

    def __init__(self, **kwargs):
        super().__init__(env_id="MiniWorld-PutNext-v0", **kwargs)

    def get_task_description(self) -> str:
        return (
            "GOAL: Pick up a movable box and place it directly next to the TARGET BOX. "
            "The environment contains at least two boxes: one or more movable boxes you can pick up, and one target box. "
            "STRATEGY: (1) EXPLORE by turning and moving to locate a movable box (not the target), "
            "(2) Walk up close to the movable box until you're standing directly next to it, "
            "(3) Use the PICKUP action to pick up the box (you'll now be carrying it), "
            "(4) While carrying the box, navigate to find the TARGET BOX, "
            "(5) Position yourself right next to the target box, "
            "(6) Use the DROP action to place your carried box on the ground next to the target box, "
            "(7) Use the DONE action to signal task completion. "
            "SUCCESS: You win when you've placed a movable box directly adjacent to the target box and called done."
        )

    def get_available_actions(self) -> list[str]:
        return [
            "turn_left",
            "turn_right",
            "move_forward",
            "move_back",
            "pickup",
            "drop",
            "toggle",
            "done",
        ]

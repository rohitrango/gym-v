"""MiniWorld Hallway environment wrapper."""

from __future__ import annotations

from .miniworld_base import MiniWorldBaseEnv


class HallwayEnv(MiniWorldBaseEnv):
    # Meta: source=MiniWorld, category=spatial, turn=multi
    # Overrides: visual_complexity=3d_rendering
    """Navigate to the red box at the end of a hallway.

    The agent starts in a long hallway and must navigate forward to reach
    a red box placed at the end. This is a simple navigation task that tests
    basic forward movement and orientation.

    Available actions: turn_left, turn_right, move_forward
    """

    def __init__(self, **kwargs):
        super().__init__(env_id="MiniWorld-Hallway-v0", **kwargs)

    def get_task_description(self) -> str:
        return (
            "GOAL: Reach the RED BOX located at the end of the hallway. "
            "You are standing in a long, straight corridor. The red box (your target) is at the far end. "
            "STRATEGY: (1) Face down the hallway by turning left or right to align yourself, "
            "(2) Move forward repeatedly to walk down the hallway toward the red box, "
            "(3) Continue moving forward until you reach the box. "
            "SUCCESS: You win when you reach the red box at the end of the hallway."
        )

    def get_available_actions(self) -> list[str]:
        return ["turn_left", "turn_right", "move_forward"]

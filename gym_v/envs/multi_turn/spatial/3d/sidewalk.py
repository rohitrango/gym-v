"""MiniWorld Sidewalk environment wrapper."""

from __future__ import annotations

from .miniworld_base import MiniWorldBaseEnv


class SidewalkEnv(MiniWorldBaseEnv):
    # Meta: source=MiniWorld, category=spatial, turn=multi
    # Overrides: visual_complexity=3d_rendering
    """Navigate along a sidewalk while avoiding obstacles.

    The agent must follow a sidewalk path to reach the goal while avoiding
    obstacles placed along the route. This tests path following and obstacle
    avoidance in a constrained navigation scenario.

    Available actions: turn_left, turn_right, move_forward
    """

    def __init__(self, **kwargs):
        super().__init__(env_id="MiniWorld-Sidewalk-v0", **kwargs)

    def get_task_description(self) -> str:
        return (
            "GOAL: Follow the SIDEWALK PATH to reach the end while avoiding OBSTACLES. "
            "You are standing on a sidewalk (a defined path). The sidewalk leads to the goal, but there are obstacles placed along it that you must avoid. "
            "STRATEGY: (1) Stay ON the sidewalk path (do not step off into non-sidewalk areas), "
            "(2) Move forward to progress along the sidewalk toward the goal, "
            "(3) When you encounter an OBSTACLE blocking the path, turn left or right to maneuver around it, "
            "(4) Once you've navigated around an obstacle, realign yourself with the sidewalk path, "
            "(5) Continue moving forward along the sidewalk until you reach the end. "
            "SUCCESS: You win when you reach the end of the sidewalk path. Stepping off the sidewalk or hitting obstacles may result in failure."
        )

    def get_available_actions(self) -> list[str]:
        return ["turn_left", "turn_right", "move_forward"]

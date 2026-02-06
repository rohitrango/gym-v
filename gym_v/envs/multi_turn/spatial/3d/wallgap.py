"""MiniWorld WallGap environment wrapper."""

from __future__ import annotations

from .miniworld_base import MiniWorldBaseEnv


class WallGapEnv(MiniWorldBaseEnv):
    # Meta: source=MiniWorld, category=spatial, turn=multi
    # Overrides: visual_complexity=3d_rendering
    """Find and navigate through a gap in a wall.

    The agent starts on one side of a wall with a gap and must locate the
    opening, then navigate through it to reach the goal on the other side.
    This tests exploration and precision navigation through narrow passages.

    Available actions: turn_left, turn_right, move_forward
    """

    def __init__(self, **kwargs):
        super().__init__(env_id="MiniWorld-WallGap-v0", **kwargs)

    def get_task_description(self) -> str:
        return (
            "GOAL: Find and navigate through the GAP in the wall to reach the goal on the other side. "
            "There is a WALL blocking your path with a GAP (opening) somewhere in it. The gap may not be visible from your starting position. "
            "STRATEGY: (1) EXPLORE by turning left or right to scan along the wall and find where the gap is located, "
            "(2) Move forward along the wall if needed to get a better view of different sections, "
            "(3) Once you locate the gap, POSITION yourself to face the opening directly, "
            "(4) Carefully WALK THROUGH the gap by moving forward, "
            "(5) Continue to the other side of the wall. "
            "SUCCESS: You win when you successfully pass through the gap to the other side. The gap requires exploration and precise navigation."
        )

    def get_available_actions(self) -> list[str]:
        return ["turn_left", "turn_right", "move_forward"]

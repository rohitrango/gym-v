"""MiniWorld OneRoom environment wrappers."""

from __future__ import annotations

from .miniworld_base import MiniWorldBaseEnv


class OneRoomEnv(MiniWorldBaseEnv):
    # Meta: source=MiniWorld, category=spatial, turn=multi
    # Overrides: visual_complexity=3d_rendering
    """Navigate to the goal in a single room.

    The agent starts in a single room and must navigate to reach the goal
    location. This tests basic navigation and spatial orientation in an
    open environment.

    Available actions: turn_left, turn_right, move_forward
    """

    def __init__(self, **kwargs):
        super().__init__(env_id="MiniWorld-OneRoom-v0", **kwargs)

    def get_task_description(self) -> str:
        return (
            "GOAL: Reach the RED BOX located somewhere in this single room. "
            "You are standing in an open room with four walls. The red box is placed somewhere on the floor. "
            "STRATEGY: (1) TURN left or right to scan the room and locate the red box, "
            "(2) Once you spot the red box, turn to face it directly, "
            "(3) Move forward repeatedly to walk straight toward the red box until you reach it. "
            "SUCCESS: You win when you reach the red box. The box may not be visible from your starting position, so you may need to turn around first."
        )

    def get_available_actions(self) -> list[str]:
        return ["turn_left", "turn_right", "move_forward"]


class OneRoomS6Env(MiniWorldBaseEnv):
    # Meta: source=MiniWorld, category=spatial, turn=multi
    # Overrides: visual_complexity=3d_rendering
    """Navigate to the goal in a 6x6 room.

    A larger 6x6 variant of the single room environment. The increased
    size requires longer navigation paths and better spatial reasoning.

    Available actions: turn_left, turn_right, move_forward
    """

    def __init__(self, **kwargs):
        super().__init__(env_id="MiniWorld-OneRoomS6-v0", **kwargs)

    def get_task_description(self) -> str:
        return (
            "GOAL: Reach the RED BOX located somewhere in this large 6x6 room. "
            "You are standing in a large, open room (6x6 units). The red box is placed somewhere on the floor. "
            "STRATEGY: (1) TURN left or right to scan the room and locate the red box, "
            "(2) Due to the larger room size, the box might be quite far away, "
            "(3) Once you spot it, turn to face it directly and move forward repeatedly, "
            "(4) Keep moving forward until you reach the box. "
            "SUCCESS: You win when you reach the red box. The larger room means longer travel distances."
        )

    def get_available_actions(self) -> list[str]:
        return ["turn_left", "turn_right", "move_forward"]


class OneRoomS6FastEnv(MiniWorldBaseEnv):
    # Meta: source=MiniWorld, category=spatial, turn=multi
    # Overrides: visual_complexity=3d_rendering
    """Navigate to the goal in a 6x6 room with faster movement.

    A 6x6 room variant with increased movement speed. The faster pace
    requires quicker decision-making while navigating the larger space.

    Available actions: turn_left, turn_right, move_forward
    """

    def __init__(self, **kwargs):
        super().__init__(env_id="MiniWorld-OneRoomS6Fast-v0", **kwargs)

    def get_task_description(self) -> str:
        return (
            "GOAL: Reach the RED BOX located somewhere in this large 6x6 room. "
            "You are in a large, open room (6x6 units) with FASTER MOVEMENT SPEED than normal. "
            "STRATEGY: (1) TURN left or right to quickly scan the room and locate the red box, "
            "(2) Once you spot it, turn to face it directly, "
            "(3) Move forward repeatedly to walk toward the box (movement is faster than normal), "
            "(4) The increased speed makes you cover ground quickly but requires responsive adjustments. "
            "SUCCESS: You win when you reach the red box."
        )

    def get_available_actions(self) -> list[str]:
        return ["turn_left", "turn_right", "move_forward"]

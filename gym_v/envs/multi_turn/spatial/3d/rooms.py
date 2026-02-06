"""MiniWorld multi-room environment wrappers."""

from __future__ import annotations

from .miniworld_base import MiniWorldBaseEnv


class FourRoomsEnv(MiniWorldBaseEnv):
    # Meta: source=MiniWorld, category=spatial, turn=multi
    # Overrides: visual_complexity=3d_rendering
    """Navigate through four connected rooms to reach the goal.

    The agent starts in one of four interconnected rooms and must navigate
    through doorways to explore the rooms and find the goal. This tests
    multi-room navigation and spatial memory.

    Available actions: turn_left, turn_right, move_forward
    """

    def __init__(self, **kwargs):
        super().__init__(env_id="MiniWorld-FourRooms-v0", **kwargs)

    def get_task_description(self) -> str:
        return (
            "GOAL: Find and reach the RED BOX hidden somewhere in the four connected rooms. "
            "You are in a layout with FOUR ROOMS connected by DOORWAYS. The red box is located in one of these rooms. "
            "STRATEGY: (1) EXPLORE your current room by turning to scan for the red box or doorways to other rooms, "
            "(2) Look for DOORWAYS (openings in walls) that connect rooms to each other, "
            "(3) Move forward through doorways to enter adjacent rooms, "
            "(4) Systematically check each of the four rooms by navigating through doorways, "
            "(5) When you find the red box, move forward to reach it. "
            "SUCCESS: You win when you reach the red box. You may need to explore multiple rooms through doorways before finding it."
        )

    def get_available_actions(self) -> list[str]:
        return ["turn_left", "turn_right", "move_forward"]


class ThreeRoomsEnv(MiniWorldBaseEnv):
    # Meta: source=MiniWorld, category=spatial, turn=multi
    # Overrides: visual_complexity=3d_rendering
    """Navigate through three connected rooms to reach the goal.

    The agent must explore three interconnected rooms to locate the goal.
    The layout requires strategic exploration to efficiently find the
    target location.

    Available actions: turn_left, turn_right, move_forward
    """

    def __init__(self, **kwargs):
        super().__init__(env_id="MiniWorld-ThreeRooms-v0", **kwargs)

    def get_task_description(self) -> str:
        return (
            "GOAL: Find and reach the RED BOX hidden somewhere in the three connected rooms. "
            "You are in a layout with THREE ROOMS connected by DOORWAYS. The red box is located in one of these rooms. "
            "STRATEGY: (1) EXPLORE your current room by turning to scan for the red box or doorways to other rooms, "
            "(2) Look for DOORWAYS (openings in walls) that connect rooms to each other, "
            "(3) Move forward through doorways to enter adjacent rooms, "
            "(4) Systematically check each of the three rooms by navigating through doorways, "
            "(5) When you find the red box, move forward to reach it. "
            "SUCCESS: You win when you reach the red box. You may need to explore multiple rooms before finding it."
        )

    def get_available_actions(self) -> list[str]:
        return ["turn_left", "turn_right", "move_forward"]


class RoomObjectsEnv(MiniWorldBaseEnv):
    # Meta: source=MiniWorld, category=spatial, turn=multi
    # Overrides: visual_complexity=3d_rendering
    """Navigate to the goal in a room with objects.

    The agent must navigate to the goal in a room containing various
    objects. The objects act as obstacles that must be navigated around,
    testing path planning in cluttered environments.

    Available actions: turn_left, turn_right, move_forward
    """

    def __init__(self, **kwargs):
        super().__init__(env_id="MiniWorld-RoomObjects-v0", **kwargs)

    def get_task_description(self) -> str:
        return (
            "GOAL: Reach the RED BOX in this room filled with obstacles. "
            "You are in a single room containing the red box (your target) and various OBJECTS that act as obstacles blocking direct paths. "
            "STRATEGY: (1) TURN to scan the room and locate both the red box and the obstacle objects, "
            "(2) Identify objects blocking your path (boxes, furniture, or other items you cannot walk through), "
            "(3) Plan a path around the obstacles to reach the red box, "
            "(4) Turn and move forward to navigate around obstacles, adjusting your direction as needed, "
            "(5) Keep moving toward the red box while avoiding collisions with obstacles. "
            "SUCCESS: You win when you reach the red box. The cluttered room requires careful path planning to avoid obstacles."
        )

    def get_available_actions(self) -> list[str]:
        return ["turn_left", "turn_right", "move_forward"]

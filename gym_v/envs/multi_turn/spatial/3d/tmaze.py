"""MiniWorld TMaze environment wrappers."""

from __future__ import annotations

from .miniworld_base import MiniWorldBaseEnv


class TMazeEnv(MiniWorldBaseEnv):
    # Meta: source=MiniWorld, category=spatial, turn=multi
    # Overrides: visual_complexity=3d_rendering
    """Navigate through a T-shaped maze to reach the goal.

    The agent starts at the bottom of a T-shaped maze and must navigate
    forward to the intersection, then turn left or right to find the goal.
    This tests basic decision-making at maze junctions.

    Available actions: turn_left, turn_right, move_forward
    """

    def __init__(self, **kwargs):
        super().__init__(env_id="MiniWorld-TMaze-v0", **kwargs)

    def get_task_description(self) -> str:
        return (
            "GOAL: Find and reach the RED BOX located in one of the branches of this T-shaped maze. "
            "You are at the bottom of a T-SHAPED corridor. The maze has a vertical hallway leading to a T-junction, where it splits into left and right branches. The red box is in either the left OR right branch. "
            "STRATEGY: (1) Move forward down the initial corridor toward the T-junction, "
            "(2) When you reach the junction, you must CHOOSE: turn left or turn right, "
            "(3) Explore the chosen branch by moving forward to see if the red box is there, "
            "(4) If the red box is not in that branch, backtrack to the junction and try the other branch. "
            "SUCCESS: You win when you reach the red box. The goal location varies, so you may need to explore both branches."
        )

    def get_available_actions(self) -> list[str]:
        return ["turn_left", "turn_right", "move_forward"]


class TMazeLeftEnv(MiniWorldBaseEnv):
    # Meta: source=MiniWorld, category=spatial, turn=multi
    # Overrides: visual_complexity=3d_rendering
    """Navigate through a T-shaped maze with the goal on the left.

    A variant of the T-maze where the goal is always located on the left
    branch. The agent must learn to consistently choose the left path at
    the junction.

    Available actions: turn_left, turn_right, move_forward
    """

    def __init__(self, **kwargs):
        super().__init__(env_id="MiniWorld-TMazeLeft-v0", **kwargs)

    def get_task_description(self) -> str:
        return (
            "GOAL: Reach the RED BOX located in the LEFT BRANCH of this T-shaped maze. "
            "You are at the bottom of a T-SHAPED corridor. The maze has a vertical hallway leading to a T-junction. The red box is specifically in the LEFT branch. "
            "STRATEGY: (1) Move forward down the initial corridor toward the T-junction, "
            "(2) When you reach the junction, TURN LEFT, "
            "(3) Move forward down the left corridor to reach the red box. "
            "SUCCESS: You win when you reach the red box in the left branch. In this variant, the goal is always on the left."
        )

    def get_available_actions(self) -> list[str]:
        return ["turn_left", "turn_right", "move_forward"]


class TMazeRightEnv(MiniWorldBaseEnv):
    # Meta: source=MiniWorld, category=spatial, turn=multi
    # Overrides: visual_complexity=3d_rendering
    """Navigate through a T-shaped maze with the goal on the right.

    A variant of the T-maze where the goal is always located on the right
    branch. The agent must learn to consistently choose the right path at
    the junction.

    Available actions: turn_left, turn_right, move_forward
    """

    def __init__(self, **kwargs):
        super().__init__(env_id="MiniWorld-TMazeRight-v0", **kwargs)

    def get_task_description(self) -> str:
        return (
            "GOAL: Reach the RED BOX located in the RIGHT BRANCH of this T-shaped maze. "
            "You are at the bottom of a T-SHAPED corridor. The maze has a vertical hallway leading to a T-junction. The red box is specifically in the RIGHT branch. "
            "STRATEGY: (1) Move forward down the initial corridor toward the T-junction, "
            "(2) When you reach the junction, TURN RIGHT, "
            "(3) Move forward down the right corridor to reach the red box. "
            "SUCCESS: You win when you reach the red box in the right branch. In this variant, the goal is always on the right."
        )

    def get_available_actions(self) -> list[str]:
        return ["turn_left", "turn_right", "move_forward"]

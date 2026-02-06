"""MiniWorld YMaze environment wrappers."""

from __future__ import annotations

from .miniworld_base import MiniWorldBaseEnv


class YMazeEnv(MiniWorldBaseEnv):
    # Meta: source=MiniWorld, category=spatial, turn=multi
    # Overrides: visual_complexity=3d_rendering
    """Navigate through a Y-shaped maze to reach the goal.

    The agent starts at the bottom of a Y-shaped maze and must navigate
    to a three-way junction, then choose one of the branches to find the
    goal. This tests decision-making at multi-way intersections.

    Available actions: turn_left, turn_right, move_forward
    """

    def __init__(self, **kwargs):
        super().__init__(env_id="MiniWorld-YMaze-v0", **kwargs)

    def get_task_description(self) -> str:
        return (
            "GOAL: Find and reach the RED BOX located in one of the branches of this Y-shaped maze. "
            "You are at the bottom of a Y-SHAPED corridor system. The maze has one main path that splits into THREE branches at a junction (left, straight, or right). The red box is in one of these three branches. "
            "STRATEGY: (1) Move forward down the initial corridor toward the three-way junction, "
            "(2) When you reach the junction, you must CHOOSE which branch to explore (left, straight, or right), "
            "(3) Turn as needed and move forward down the chosen branch to see if the red box is there, "
            "(4) If the red box is not in that branch, backtrack to the junction and try a different branch. "
            "SUCCESS: You win when you reach the red box. You may need to explore multiple branches to find it."
        )

    def get_available_actions(self) -> list[str]:
        return ["turn_left", "turn_right", "move_forward"]


class YMazeLeftEnv(MiniWorldBaseEnv):
    # Meta: source=MiniWorld, category=spatial, turn=multi
    # Overrides: visual_complexity=3d_rendering
    """Navigate through a Y-shaped maze with the goal on the left.

    A variant of the Y-maze where the goal is always located on the left
    branch. The agent must learn to consistently choose the left path at
    the three-way junction.

    Available actions: turn_left, turn_right, move_forward
    """

    def __init__(self, **kwargs):
        super().__init__(env_id="MiniWorld-YMazeLeft-v0", **kwargs)

    def get_task_description(self) -> str:
        return (
            "GOAL: Reach the RED BOX located in the LEFT BRANCH of this Y-shaped maze. "
            "You are at the bottom of a Y-SHAPED corridor system. The maze has one main path that splits into three branches at a junction. The red box is specifically in the LEFT branch. "
            "STRATEGY: (1) Move forward down the initial corridor toward the three-way junction, "
            "(2) When you reach the junction, TURN LEFT, "
            "(3) Move forward down the left corridor to reach the red box. "
            "SUCCESS: You win when you reach the red box in the left branch. In this variant, the goal is always on the left."
        )

    def get_available_actions(self) -> list[str]:
        return ["turn_left", "turn_right", "move_forward"]


class YMazeRightEnv(MiniWorldBaseEnv):
    # Meta: source=MiniWorld, category=spatial, turn=multi
    # Overrides: visual_complexity=3d_rendering
    """Navigate through a Y-shaped maze with the goal on the right.

    A variant of the Y-maze where the goal is always located on the right
    branch. The agent must learn to consistently choose the right path at
    the three-way junction.

    Available actions: turn_left, turn_right, move_forward
    """

    def __init__(self, **kwargs):
        super().__init__(env_id="MiniWorld-YMazeRight-v0", **kwargs)

    def get_task_description(self) -> str:
        return (
            "GOAL: Reach the RED BOX located in the RIGHT BRANCH of this Y-shaped maze. "
            "You are at the bottom of a Y-SHAPED corridor system. The maze has one main path that splits into three branches at a junction. The red box is specifically in the RIGHT branch. "
            "STRATEGY: (1) Move forward down the initial corridor toward the three-way junction, "
            "(2) When you reach the junction, TURN RIGHT, "
            "(3) Move forward down the right corridor to reach the red box. "
            "SUCCESS: You win when you reach the red box in the right branch. In this variant, the goal is always on the right."
        )

    def get_available_actions(self) -> list[str]:
        return ["turn_left", "turn_right", "move_forward"]

"""MiniWorld Maze environment wrappers."""

from __future__ import annotations

from .miniworld_base import MiniWorldBaseEnv


class MazeEnv(MiniWorldBaseEnv):
    # Meta: source=MiniWorld, category=spatial, turn=multi
    # Overrides: visual_complexity=3d_rendering
    """Navigate through a maze to reach the goal.

    The agent must explore a procedurally generated maze and find the goal
    location. The maze requires spatial reasoning and memory to navigate
    efficiently.

    Available actions: turn_left, turn_right, move_forward
    """

    def __init__(self, **kwargs):
        super().__init__(env_id="MiniWorld-Maze-v0", **kwargs)

    def get_task_description(self) -> str:
        return (
            "GOAL: Find and reach the RED BOX hidden somewhere in the maze. "
            "You are inside a maze with multiple corridors and dead ends. The red box could be anywhere. "
            "STRATEGY: (1) EXPLORE systematically by turning left or right at intersections to check each path, "
            "(2) Move forward down corridors, backing out of dead ends when you hit walls, "
            "(3) Remember which paths you've already explored to avoid revisiting the same areas, "
            "(4) Continue exploring until you find the red box. "
            "SUCCESS: You win when you reach the red box. The maze is procedurally generated, so the solution differs each time."
        )

    def get_available_actions(self) -> list[str]:
        return ["turn_left", "turn_right", "move_forward"]


class MazeS2Env(MiniWorldBaseEnv):
    # Meta: source=MiniWorld, category=spatial, turn=multi
    # Overrides: visual_complexity=3d_rendering
    """Navigate through a 2x2 maze to reach the goal.

    A smaller 2x2 variant of the maze environment. The compact size makes
    exploration faster while still requiring basic navigation skills.

    Available actions: turn_left, turn_right, move_forward
    """

    def __init__(self, **kwargs):
        super().__init__(env_id="MiniWorld-MazeS2-v0", **kwargs)

    def get_task_description(self) -> str:
        return (
            "GOAL: Find and reach the RED BOX hidden in this compact 2x2 maze. "
            "You are inside a small maze (2x2 grid of rooms). The red box is hidden in one of the rooms. "
            "STRATEGY: (1) Turn left or right to explore each corridor systematically, "
            "(2) Move forward to navigate through the maze, "
            "(3) When you hit a dead end, turn around and try a different path, "
            "(4) The small size means fewer paths to check. "
            "SUCCESS: You win when you reach the red box."
        )

    def get_available_actions(self) -> list[str]:
        return ["turn_left", "turn_right", "move_forward"]


class MazeS3Env(MiniWorldBaseEnv):
    # Meta: source=MiniWorld, category=spatial, turn=multi
    # Overrides: visual_complexity=3d_rendering
    """Navigate through a 3x3 maze to reach the goal.

    A medium-sized 3x3 variant of the maze environment. The increased
    complexity requires more careful exploration and path planning.

    Available actions: turn_left, turn_right, move_forward
    """

    def __init__(self, **kwargs):
        super().__init__(env_id="MiniWorld-MazeS3-v0", **kwargs)

    def get_task_description(self) -> str:
        return (
            "GOAL: Find and reach the RED BOX hidden in this 3x3 maze. "
            "You are inside a medium-sized maze (3x3 grid of rooms) with more corridors and intersections than the smaller versions. "
            "STRATEGY: (1) Systematically explore by choosing a direction at each intersection (e.g., always try right first), "
            "(2) Move forward through corridors, turning when you encounter walls or intersections, "
            "(3) Keep mental track of areas you've explored to avoid wasting time revisiting them, "
            "(4) The larger size requires more careful exploration than the 2x2 version. "
            "SUCCESS: You win when you reach the red box."
        )

    def get_available_actions(self) -> list[str]:
        return ["turn_left", "turn_right", "move_forward"]


class MazeS3FastEnv(MiniWorldBaseEnv):
    # Meta: source=MiniWorld, category=spatial, turn=multi
    # Overrides: visual_complexity=3d_rendering
    """Navigate through a 3x3 maze with faster movement.

    A 3x3 maze variant with increased movement speed. The faster pace
    requires quicker decision-making and adaptation.

    Available actions: turn_left, turn_right, move_forward
    """

    def __init__(self, **kwargs):
        super().__init__(env_id="MiniWorld-MazeS3Fast-v0", **kwargs)

    def get_task_description(self) -> str:
        return (
            "GOAL: Find and reach the RED BOX hidden in this 3x3 maze. "
            "You are inside a medium-sized maze (3x3 grid) with FASTER MOVEMENT SPEED than normal. "
            "STRATEGY: (1) Explore systematically, but make decisions quickly due to the faster pace, "
            "(2) Move forward through corridors, turning at intersections to explore each path, "
            "(3) Be ready to react quickly when you spot the red box or need to change direction, "
            "(4) The increased speed makes navigation feel more responsive but requires quicker decision-making. "
            "SUCCESS: You win when you reach the red box."
        )

    def get_available_actions(self) -> list[str]:
        return ["turn_left", "turn_right", "move_forward"]

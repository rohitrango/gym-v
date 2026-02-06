"""MiniWorld Sign environment wrapper."""

from __future__ import annotations

from .miniworld_base import MiniWorldBaseEnv


class SignEnv(MiniWorldBaseEnv):
    # Meta: source=MiniWorld, category=spatial, turn=multi
    # Overrides: visual_complexity=3d_rendering
    """Read a sign and navigate to the indicated location.

    The agent must read text displayed on a sign that indicates where the
    goal is located, then navigate to that location. This tests visual
    understanding of text and following directional instructions.

    Available actions: turn_left, turn_right, move_forward
    """

    def __init__(self, **kwargs):
        super().__init__(env_id="MiniWorld-Sign-v0", **kwargs)

    def get_task_description(self) -> str:
        return (
            "GOAL: Read the DIRECTIONAL SIGN and navigate to the location it indicates. "
            "The environment contains a sign with TEXT that tells you where to go (e.g., 'Go left', 'Red box is ahead', etc.). "
            "STRATEGY: (1) LOCATE the sign by turning left or right to scan your surroundings, "
            "(2) Move close enough to the sign to READ the text on it clearly, "
            "(3) UNDERSTAND the directional instruction written on the sign (e.g., left, right, forward), "
            "(4) FOLLOW the sign's instructions by turning the appropriate direction and moving forward, "
            "(5) Navigate to the location indicated by the sign until you reach the goal. "
            "SUCCESS: You win when you reach the goal location indicated by the sign. This task requires visual text reading and following written directions."
        )

    def get_available_actions(self) -> list[str]:
        return ["turn_left", "turn_right", "move_forward"]

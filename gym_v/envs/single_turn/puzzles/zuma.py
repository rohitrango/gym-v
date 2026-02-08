"""Zuma Game Q&A environment."""

from __future__ import annotations

import math
import random
from textwrap import dedent
from typing import Any

import matplotlib

from gym_v.utils.gamerl_utils import build_description, score_exact

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np

from gym_v import Env, Observation, get_logger

logger = get_logger()


class ZumaQAEnv(Env):
    # Meta: source=GameRL, category=puzzles, turn=single
    # Overrides: interaction_mode=single_turn, action_format=open_ended
    """Zuma game Q&A environment.

    A puzzle game where a frog shoots colored marbles toward a winding track of marbles.
    The goal is to eliminate marbles by creating groups of 3+ same-colored marbles.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    QUESTION_TYPES = [
        {
            "id": "frog_ball_color",
            "name": "Frog Ball Color",
            "level": "Easy",
            "answer_format": "fill_in_blank",
            "qa_type": "Target Perception",
        },
        {
            "id": "color_count",
            "name": "Color Count",
            "level": "Easy",
            "answer_format": "fill_in_blank",
            "qa_type": "Target Perception",
        },
        {
            "id": "direction_groups",
            "name": "Direction Groups",
            "level": "Medium",
            "answer_format": "fill_in_blank",
            "qa_type": "Target Perception",
        },
        {
            "id": "angle_shot_color",
            "name": "Angle Shot Color",
            "level": "Medium",
            "answer_format": "fill_in_blank",
            "qa_type": "Target Perception",
        },
        {
            "id": "angle_shot_result",
            "name": "Angle Shot Result",
            "level": "Hard",
            "answer_format": "fill_in_blank",
            "qa_type": "State Prediction",
        },
        {
            "id": "optimal_strategy",
            "name": "Optimal Strategy",
            "level": "Hard",
            "answer_format": "fill_in_blank",
            "qa_type": "Strategy Optimization",
        },
    ]

    COLORS = ["red", "yellow", "blue", "green"]
    DIRECTIONS = [
        "up",
        "down",
        "left",
        "right",
        "up-left",
        "up-right",
        "down-left",
        "down-right",
    ]
    DIRECTION_RANGES = {
        "up": "67.5 degrees ~ 112.5 degrees",
        "down": "-112.5 degrees ~ -67.5 degrees",
        "left": "157.5 degrees ~ 180 degrees or -180 degrees ~ -157.5 degrees",
        "right": "-22.5 degrees ~ 22.5 degrees",
        "up-left": "112.5 degrees ~ 157.5 degrees",
        "up-right": "22.5 degrees ~ 67.5 degrees",
        "down-left": "-157.5 degrees ~ -112.5 degrees",
        "down-right": "-67.5 degrees ~ -22.5 degrees",
    }

    def __init__(
        self,
        question_type: int | None = None,
        curve_type: str | None = None,
        num_balls: int | None = None,
        ball_radius: float = 0.3,
        num_players: int = 1,
        **kwargs,
    ):
        """Initialize Zuma QA environment.

        Args:
            question_type: Type of question (default: random)
            curve_type: Track curve type: 'spiral', 'heart', 'ellipse' (default: random)
            num_balls: Number of balls on track (default: random 10-130)
            ball_radius: Radius of each ball
        """
        super().__init__(**kwargs)
        self._question_type_param = question_type
        self._curve_type = curve_type
        self._num_balls = num_balls
        self._ball_radius = ball_radius
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        # Game state
        self._frog_pos: dict[str, float] = {}
        self._frog_angle: float = 0
        self._frog_color: str = ""
        self._frog_next_color: str = ""
        self._balls: list[dict[str, Any]] = []
        self._hole_pos: dict[str, float] = {}
        self._track_data: tuple[np.ndarray, np.ndarray] = (np.array([]), np.array([]))
        self._plot_level: str = "Medium"

        # Question data (standard QA variables)
        self._question_type_idx: int = 0
        self._question: str = ""
        self._options: list[str] | None = None
        self._oracle_answer: str = ""
        self._answer_format: str | None = None
        self._analysis: str = ""
        self._selected_question_type: str = ""

    GAME_RULES = dedent("""
        This is a Zuma game.
        You need to control a frog to shoot colored marbles from its mouth toward a winding track of approaching marbles.
        Your goal is to clear all marbles before they reach the black hole at the end of the track.
        The marbles roll steadily along the track, and the player must fire marbles to create groups of three or more of the same color.
        These groups will disappear, reducing the number of marbles on the track.
        The frog will shoot marbles in a straight line.
        If there is no marble on the track, the shot marble will pass through the track.
        However, the marble it shoots cannot bypass marbles already in its direct line of fire.
        In the offered pictures, the frog is represented as a white triangle,
        with the circle on it representing the next marble it will shoot.
        The colored marbles are positioned on a gray track.
        Any directions or angles mentioned in questions are relative to the center of the circle on the frog,
        with its positive x-axis as the 0-degree reference line.
    """).strip()

    @property
    def description(self) -> str:
        """Return game rules + current question + answer format."""
        return build_description(
            game_name="Zuma",
            rules=self.GAME_RULES,
            question=self._question,
            options=self._options,
            oracle_answer=self._oracle_answer,
            answer_format=self._answer_format,
        )

    def _get_state_text(self) -> str:
        """Generate text description of current Zuma game state.

        Returns a text representation with ball positions and colors.
        """
        # List all balls in order along the track
        ball_sequence = [ball["color"] for ball in self._balls]

        # Create a visual representation
        if ball_sequence:
            balls_str = " -> ".join(ball_sequence)
        else:
            balls_str = "No balls on track"

        # Color counts
        color_counts = {}
        for ball in self._balls:
            color = ball["color"]
            color_counts[color] = color_counts.get(color, 0) + 1

        counts_str = ", ".join(
            [f"{color}: {count}" for color, count in sorted(color_counts.items())]
        )

        return f"""Zuma Game State
Total balls: {len(self._balls)}
Ball sequence (frog to hole): {balls_str}
Color counts: {counts_str if counts_str else "None"}
Frog position: ({self._frog_pos["x"]:.2f}, {self._frog_pos["y"]:.2f})
Hole position: ({self._hole_pos["x"]:.2f}, {self._hole_pos["y"]:.2f})"""

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        """Reset the environment and generate a new question."""
        super().reset(seed=seed)

        # Select question type
        if self._question_type_param is None:
            self._question_type_idx = int(self.np_random.integers(0, len(self.QUESTION_TYPES)))
        else:
            self._question_type_idx = self._question_type_param

        # Validate question type index
        if not (0 <= self._question_type_idx < len(self.QUESTION_TYPES)):
            raise ValueError(f"Invalid question type index: {self._question_type_idx}")

        # Get question type ID from index
        self._selected_question_type = self.QUESTION_TYPES[self._question_type_idx][
            "id"
        ]

        # Select curve type
        curve_type = (
            self._curve_type
            if self._curve_type is not None
            else random.choice(["spiral", "heart", "ellipse"])
        )

        # Select number of balls
        num_balls = (
            self._num_balls if self._num_balls is not None else random.randint(10, 130)
        )

        # Generate game state
        self._generate_game_state(curve_type, num_balls)

        # Generate question
        self._generate_question()

        text_state = self._get_state_text()

        obs = Observation(
            image=self.render(),
            text=None,
            metadata={
                "state_text": text_state,
                "text_prompt": f"{text_state}\n\n{self.description}",
                "question": self._question,
                "options": self._options,
                "question_type": self.QUESTION_TYPES[self._question_type_idx]["name"],
                "level": self.QUESTION_TYPES[self._question_type_idx]["level"],
            },
        )

        logger.info(
            f"Reset Zuma QA (balls={len(self._balls)}, curve={curve_type}, question: {self._selected_question_type})."
        )

        info = {
            "seed": seed,
            "oracle_answer": self._oracle_answer,
            "question_type": self._selected_question_type,
        }

        return {agent_id: obs for agent_id in self._agent_ids}, {
            agent_id: info for agent_id in self._agent_ids
        }

    def _generate_game_state(self, curve_type: str, num_balls: int):
        """Generate the game state with track, frog, and balls."""
        # Random frog and hole positions
        self._frog_pos = {
            "x": float(np.random.uniform(-1, 1)),
            "y": float(np.random.uniform(-1, 1)),
        }
        hole_x = float(np.random.uniform(1.5, 3) * np.random.choice([-1, 1]))
        hole_y = float(np.random.uniform(1.5, 3) * np.random.choice([-1, 1]))
        self._hole_pos = {"x": hole_x, "y": hole_y}

        # Random frog angle and colors
        self._frog_angle = float(np.random.randint(-179, 180))
        self._frog_color = random.choice(self.COLORS)
        self._frog_next_color = random.choice(self.COLORS)

        # Generate track curve
        if curve_type == "spiral":
            b = 0.2
            self._track_data = self._generate_spiral_curve(self._hole_pos, b, num_balls)
        elif curve_type == "heart":
            self._track_data = self._generate_heart_curve(num_balls)
        elif curve_type == "ellipse":
            a, b = 8.0, 5.0
            self._track_data = self._generate_ellipse_curve(a, b, num_balls)
        else:
            raise ValueError(f"Unknown curve type: {curve_type}")

        curve_x, curve_y = self._track_data

        # Determine plot level based on number of balls
        if 10 <= num_balls <= 15:
            self._plot_level = "Easy"
        elif 15 < num_balls <= 30:
            self._plot_level = "Easy"
        elif 30 < num_balls <= 55:
            self._plot_level = "Medium"
        elif 55 < num_balls <= 75:
            self._plot_level = "Medium"
        else:
            self._plot_level = "Hard"

        # Generate balls along the track
        self._balls = []
        ball_centers_x = [curve_x[-1]]
        ball_centers_y = [curve_y[-1]]

        k = len(curve_x) - 1
        for _ in range(1, num_balls):
            prev_x, prev_y = ball_centers_x[-1], ball_centers_y[-1]
            for j in range(k - 1, -1, -1):
                next_x, next_y = curve_x[j], curve_y[j]
                distance = np.sqrt((next_x - prev_x) ** 2 + (next_y - prev_y) ** 2)
                if distance >= 2 * self._ball_radius:
                    ball_centers_x.append(next_x)
                    ball_centers_y.append(next_y)
                    k = j
                    break

        for x, y in zip(ball_centers_x, ball_centers_y, strict=False):
            self._balls.append(
                {
                    "position": {"x": float(x), "y": float(y)},
                    "color": random.choice(self.COLORS),
                }
            )

    def _generate_spiral_curve(
        self, point: dict, b: float, num_balls: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate spiral curve."""
        # Determine curve length based on num_balls
        if 10 <= num_balls <= 15:
            length = np.random.uniform(0.5, 1)
        elif 15 < num_balls <= 30:
            length = np.random.uniform(1, 1.5)
        elif 30 < num_balls <= 55:
            length = np.random.uniform(1.5, 2)
        elif 55 < num_balls <= 75:
            length = np.random.uniform(2, 2.5)
        elif 75 < num_balls <= 100:
            length = np.random.uniform(2.5, 3)
        else:
            length = 3.0

        t = np.linspace(0, length, 2000)
        p, q = point["x"], point["y"]
        a = np.sqrt(p**2 + q**2) - b * np.arctan2(q, p)
        theta = t * 2 * np.pi + np.arctan2(q, p)
        r = a + b * theta
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return x, y

    def _generate_heart_curve(self, num_balls: int) -> tuple[np.ndarray, np.ndarray]:
        """Generate heart-shaped curve."""
        if 10 <= num_balls <= 15:
            length = np.random.uniform(0.5, 1)
        elif 15 < num_balls <= 30:
            length = np.random.uniform(1, 1.5)
        elif 30 < num_balls <= 55:
            length = np.random.uniform(1.5, 2)
        elif 55 < num_balls <= 75:
            length = np.random.uniform(2, 2.5)
        elif 75 < num_balls <= 100:
            length = np.random.uniform(2.5, 3)
        else:
            length = 3.0

        t = np.linspace(0, length, 2000)
        x = 16 * np.sin(t) ** 3
        y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
        return x, y

    def _generate_ellipse_curve(
        self, a: float, b: float, num_balls: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate ellipse curve."""
        if 10 <= num_balls <= 15:
            length = np.random.uniform(0.5, 1)
        elif 15 < num_balls <= 30:
            length = np.random.uniform(1, 1.5)
        elif 30 < num_balls <= 55:
            length = np.random.uniform(1.5, 2)
        elif 55 < num_balls <= 75:
            length = np.random.uniform(2, 2.5)
        elif 75 < num_balls <= 100:
            length = np.random.uniform(2.5, 3)
        else:
            length = 3.0

        t = np.linspace(0, length, 2000)
        x = a * np.cos(t)
        y = b * np.sin(t)
        return x, y

    def _generate_question(self):
        """Generate question based on question type."""
        if self._selected_question_type == "frog_ball_color":
            self._question_frog_ball_color()
        elif self._selected_question_type == "color_count":
            self._question_color_count()
        elif self._selected_question_type == "direction_groups":
            self._question_direction_groups()
        elif self._selected_question_type == "angle_shot_color":
            self._question_angle_shot_color()
        elif self._selected_question_type == "angle_shot_result":
            self._question_angle_shot_result()
        elif self._selected_question_type == "optimal_strategy":
            self._question_optimal_strategy()

    def _question_frog_ball_color(self):
        """Question 1: Color of marble frog will shoot."""
        self._question = (
            "What color is the marble that the frog is going to shoot? "
            "Answer in one of the following formats: 'red', 'yellow', 'blue', or 'green'."
        )
        self._options = None
        self._oracle_answer = self._frog_color
        self._answer_format = None
        self._analysis = (
            f"According to the color of the circle on the triangle (frog), "
            f"the answer is {self._frog_color}."
        )

    def _question_color_count(self):
        """Question 2: Count of specific color marbles."""
        target_color = random.choice(self.COLORS)
        count = sum(1 for ball in self._balls if ball["color"] == target_color)

        self._question = f"How many {target_color} marbles are there on the track?"
        self._options = None
        self._oracle_answer = str(count)
        self._answer_format = (
            "- Answer as a non-negative integer, such as '0', '1', '2', etc."
        )
        self._analysis = (
            f"By counting the marbles on the track, it can be determined that "
            f"there are {count} {target_color} marbles."
        )

    def _question_direction_groups(self):
        """Question 3: Number of same-color marble groups in a direction."""
        direction = random.choice(self.DIRECTIONS)
        ball_groups = self._get_ball_groups_in_direction(direction)

        self._question = (
            f"How many marble groups of two or more same-colored marbles are there at the {direction} side of the frog? "
            f"The direction '{direction}' refers to the region {self.DIRECTION_RANGES[direction]}, which is already divided by dashed lines. "
            f"Any marble group with at least one marble in this region is considered to be in the '{direction}' direction. "
        )
        self._options = None
        self._oracle_answer = str(len(ball_groups))
        self._answer_format = (
            "- Answer as a non-negative integer, such as '0', '1', '2', etc."
        )

        # Generate analysis
        color_stats = {}
        for start, end, color in ball_groups:
            group_size = end - start + 1
            if color not in color_stats:
                color_stats[color] = {"total": 0, "sizes": {}}
            color_stats[color]["total"] += 1
            if group_size not in color_stats[color]["sizes"]:
                color_stats[color]["sizes"][group_size] = 0
            color_stats[color]["sizes"][group_size] += 1

        analysis_parts = [
            f"At the {direction} side of the frog (triangle) ({self.DIRECTION_RANGES[direction]}), "
            f"there are {len(ball_groups)} groups of adjacent marbles with the same color"
        ]
        if len(ball_groups) > 0:
            analysis_parts.append(": ")
            for color, stats in color_stats.items():
                size_details = [
                    f"{count} group with {size} marbles"
                    for size, count in stats["sizes"].items()
                ]
                analysis_parts.append(
                    f"{stats['total']} {color} marble group including "
                    + ", ".join(size_details)
                    + ". "
                )
        else:
            analysis_parts.append(".")
        analysis_parts.append(f" So the answer is '{len(ball_groups)}'.")
        self._analysis = "".join(analysis_parts)

    def _question_angle_shot_color(self):
        """Question 4: Color of marble hit at specific angle."""
        angle = self._frog_angle
        color = self._angle_shot_color(angle)

        self._question = (
            f"If the frog shoots the marble at {angle} degrees, as shown in the picture, what color is the marble it hits? "
            "If it doesn't hit any marble, answer 'none'. "
            "Answer in one of the following formats: 'red', 'yellow', 'blue', 'green', or 'none'."
        )
        self._options = None
        self._oracle_answer = color
        self._answer_format = None

        if color == "none":
            self._analysis = (
                f"To determine which marble the frog hits, we draw a ray from the frog's position "
                f"(the center of the circle inside the triangle) at an angle of {angle} degrees. "
                f"We then check if there is any marble within a distance of the diameter of the marble from this ray. "
                f"For there is no such marble, the frog does not hit any marble. So the answer is none."
            )
        else:
            self._analysis = (
                f"To determine which marble the frog hits, we draw a ray from the frog's position "
                f"(the center of the circle inside the triangle) at an angle of {angle} degrees. "
                f"We then check if there is any marble within a distance of the diameter of the marble from this ray. "
                f"If there are marbles within this distance, we identify the closest marble to the frog, "
                f"which is of color {color}. Thus, the frog hits this marble. So the answer is {color}."
            )

    def _question_angle_shot_result(self):
        """Question 5: Result of shooting at specific angle."""
        angle = self._frog_angle
        color = self._frog_color
        ball_color = self._angle_shot_color(angle)
        is_hit, can_remove, removal_count = self._angle_shot_result()
        is_same_color = (color == ball_color) if is_hit else False

        self._question = (
            f"If the frog shoots the marble at {angle} degrees, as shown in the picture, "
            "what will happen?"
        )
        self._options = None
        self._answer_format = (
            "- Answer with one of the following options: 'The marble left the field.', "
            "'The marble stayed on the track.', or '[X] marbles on the track were removed.', "
            "where [X] is a positive integer."
        )

        if not is_hit:
            self._oracle_answer = "The marble left the field."
            self._analysis = (
                "The marble shot by the frog did not hit any marbles on the track, "
                "so the answer is 'The marble left the field.'."
            )
        elif is_hit and not can_remove:
            self._oracle_answer = "The marble stayed on the track."
            if is_same_color:
                self._analysis = (
                    f"The marble shot by the frog hit a {ball_color} marble on the track. "
                    "Although the hit marble is the same color as the shot marble, it does not belong to "
                    "any group of two or more same-colored marbles, "
                    "so the answer is 'The marble stayed on the track.'."
                )
            else:
                self._analysis = (
                    f"The marble shot by the frog hit a {ball_color} marble on the track. "
                    f"Firstly, the hit marble is a different color from the shot marble. "
                    f"Secondly, there is not a marble group of two or more marbles which is the same color as the shot marble "
                    f"on the side that the shot marble was inserted into. "
                    f"So the answer is 'The marble stayed on the track.'."
                )
        else:
            self._oracle_answer = f"{removal_count} marbles on the track were removed."
            if is_same_color:
                self._analysis = (
                    f"The marble shot by the frog hit a {ball_color} marble on the track. "
                    f"Since the hit marble is the same color as the shot marble, and it belongs to "
                    f"a group of {removal_count} same-colored marbles on the track, these marbles "
                    f"were removed from the track. "
                    f"So the answer is '{removal_count} marbles on the track were removed.'."
                )
            else:
                self._analysis = (
                    f"The marble shot by the frog hit a {ball_color} marble on the track. "
                    f"Although the hit marble is a different color from the shot marble, the side "
                    f"the shot marble was inserted into has a group of {removal_count} same-colored marbles, "
                    f"leading to the removal of {removal_count} marbles. "
                    f"So the answer is '{removal_count} marbles on the track were removed.'."
                )

    def _question_optimal_strategy(self):
        """Question 6: Optimal strategy for eliminating marbles."""
        can_remove, directions, removal_count = self._simulate_shot()
        optimal_solution_count = len(directions)

        self._question = (
            "Can the frog eliminate some marbles on the track by shooting the marble? "
            "If yes, how many marbles (excluding the one being shot) can be eliminated in the best case? "
            "How many distinct optimal solutions (counting different groups in the same direction separately) exist?"
        )
        self._options = None
        self._answer_format = (
            "- Answer in the format: '<Yes/No>, <number of eliminated marbles>, <number of optimal solutions>'. "
            "For example, 'Yes, 3, 2' or 'No, 0, 0'."
        )

        if can_remove:
            self._oracle_answer = f"Yes, {removal_count}, {optimal_solution_count}"
            direction_details = [
                f"{d} ({self.DIRECTION_RANGES[d]})" for d in set(directions)
            ]
            direction_text = ", ".join(direction_details)
            self._analysis = (
                f"By searching for groups of marbles on the track that the frog can hit and eliminate, "
                f"we find that the best case allows for the elimination of {removal_count} consecutive marbles of color "
                f"{self._frog_color} in the directions {directions}. "
                f"This results in {optimal_solution_count} distinct optimal solutions. "
                f"The specific angle range for each direction is as follows: {direction_text}. "
                f"Any marble group with at least one marble in the region of the direction is considered to be in the region. "
                f"These regions have been divided by gray dashed lines in the image. "
                f"So, the answer is '{self._oracle_answer}'."
            )
        else:
            self._oracle_answer = "No, 0, 0"
            self._analysis = (
                f"By searching for groups of marbles on the track that the frog can hit and eliminate, "
                f"we find that there are no groups of marbles on the track that the frog can hit and eliminate, "
                f"so the frog cannot eliminate any marbles. "
                f"So, the answer is '{self._oracle_answer}'."
            )

    def _get_direction(self, angle: float) -> str:
        """Get direction from angle."""
        if -22.5 <= angle < 22.5:
            return "right"
        elif 22.5 <= angle < 67.5:
            return "up-right"
        elif 67.5 <= angle < 112.5:
            return "up"
        elif 112.5 <= angle < 157.5:
            return "up-left"
        elif 157.5 <= angle or angle < -157.5:
            return "left"
        elif -157.5 <= angle < -112.5:
            return "down-left"
        elif -112.5 <= angle < -67.5:
            return "down"
        elif -67.5 <= angle < -22.5:
            return "down-right"
        else:
            return "unknown"

    def _get_ball_groups_in_direction(
        self, direction: str
    ) -> list[tuple[int, int, str]]:
        """Get ball groups in a specific direction."""

        def is_in_direction(ball_pos):
            angle = math.degrees(
                math.atan2(
                    ball_pos["y"] - self._frog_pos["y"],
                    ball_pos["x"] - self._frog_pos["x"],
                )
            )
            return self._get_direction(angle) == direction

        ball_groups = []
        n = len(self._balls)
        i = 0

        while i < n:
            ball = self._balls[i]
            color = ball["color"]
            group_start = i
            group_end = i

            for j in range(i + 1, n):
                if self._balls[j]["color"] == color:
                    group_end = j
                else:
                    break

            if group_end > group_start and (
                is_in_direction(ball["position"])
                or is_in_direction(self._balls[group_end]["position"])
            ):
                ball_groups.append((group_start, group_end, color))

            i = group_end + 1

        return ball_groups

    def _can_hit_by_angle(self, ball_pos: dict, direction_vector: dict) -> bool:
        """Check if frog can hit ball at given angle."""
        dx = ball_pos["x"] - self._frog_pos["x"]
        dy = ball_pos["y"] - self._frog_pos["y"]

        cross = abs(dx * direction_vector["y"] - dy * direction_vector["x"])
        dot = dx * direction_vector["x"] + dy * direction_vector["y"]

        return cross < 2 * self._ball_radius and dot > 0

    def _angle_shot_color(self, angle: float) -> str:
        """Get color of marble hit at specific angle."""
        radian = math.radians(angle)
        direction_vector = {"x": math.cos(radian), "y": math.sin(radian)}
        hit_balls = []

        for ball in self._balls:
            if self._can_hit_by_angle(ball["position"], direction_vector):
                hit_balls.append(ball)

        if not hit_balls:
            return "none"

        # Find nearest ball
        min_dist = float("inf")
        nearest_color = "none"
        for ball in hit_balls:
            dist = (ball["position"]["x"] - self._frog_pos["x"]) ** 2 + (
                ball["position"]["y"] - self._frog_pos["y"]
            ) ** 2
            if dist < min_dist:
                min_dist = dist
                nearest_color = ball["color"]

        return nearest_color

    def _angle_shot_result(self) -> tuple[bool, bool, int]:
        """Simulate shooting at frog's angle."""
        angle = self._frog_angle
        color = self._frog_color
        radian = math.radians(angle)
        direction_vector = {"x": math.cos(radian), "y": math.sin(radian)}

        hit_balls = []
        for idx, ball in enumerate(self._balls):
            if self._can_hit_by_angle(ball["position"], direction_vector):
                hit_balls.append((idx, ball))

        if not hit_balls:
            return False, False, 0

        # Find nearest ball
        min_dist = float("inf")
        nearest_idx = None
        for idx, ball in hit_balls:
            dist = (ball["position"]["x"] - self._frog_pos["x"]) ** 2 + (
                ball["position"]["y"] - self._frog_pos["y"]
            ) ** 2
            if dist < min_dist:
                min_dist = dist
                nearest_idx = idx

        if nearest_idx is None:
            return False, False, 0

        # Simulate insertion and removal
        O1 = self._balls[nearest_idx]["position"]
        dx = O1["x"] - self._frog_pos["x"]
        dy = O1["y"] - self._frog_pos["y"]
        proj_length = dx * direction_vector["x"] + dy * direction_vector["y"]

        # Determine insertion side
        if nearest_idx == len(self._balls) - 1:
            if nearest_idx == 0:
                return True, False, 0
            check_idx = nearest_idx - 1
        else:
            O2 = self._balls[nearest_idx + 1]["position"]
            v2_x = O2["x"] - O1["x"]
            v2_y = O2["y"] - O1["y"]

            P_x = self._frog_pos["x"] + proj_length * direction_vector["x"]
            P_y = self._frog_pos["y"] + proj_length * direction_vector["y"]
            d = math.sqrt((O1["x"] - P_x) ** 2 + (O1["y"] - P_y) ** 2)
            seg_length = (
                math.sqrt(4 * self._ball_radius**2 - d**2)
                if d < 2 * self._ball_radius
                else 0
            )
            A_x = P_x - seg_length * direction_vector["x"]
            A_y = P_y - seg_length * direction_vector["y"]
            v1_x = A_x - O1["x"]
            v1_y = A_y - O1["y"]

            dot_product = v1_x * v2_x + v1_y * v2_y
            check_idx = (
                nearest_idx + 1
                if dot_product > 0
                else (nearest_idx - 1 if nearest_idx > 0 else None)
            )

        # Calculate removal count
        removal_count = 0
        if check_idx is None:
            idx = nearest_idx
            if nearest_idx == 0:
                while idx < len(self._balls) and self._balls[idx]["color"] == color:
                    removal_count += 1
                    idx += 1
            else:
                while idx >= 0 and self._balls[idx]["color"] == color:
                    removal_count += 1
                    idx -= 1
            if removal_count >= 2:
                return True, True, removal_count
            else:
                return True, False, 0

        left_idx = min(nearest_idx, check_idx)
        right_idx = max(nearest_idx, check_idx)
        ball1_color = self._balls[left_idx]["color"]
        ball2_color = self._balls[right_idx]["color"]

        if ball1_color != color and ball2_color != color:
            return True, False, 0

        if ball1_color == color and ball2_color == color:
            removal_count = 2
            idx = left_idx - 1
            while idx >= 0 and self._balls[idx]["color"] == color:
                removal_count += 1
                idx -= 1
            idx = right_idx + 1
            while idx < len(self._balls) and self._balls[idx]["color"] == color:
                removal_count += 1
                idx += 1
        else:
            removal_count = 1
            if ball1_color == color:
                idx = left_idx - 1
                while idx >= 0 and self._balls[idx]["color"] == color:
                    removal_count += 1
                    idx -= 1
            else:
                idx = right_idx + 1
                while idx < len(self._balls) and self._balls[idx]["color"] == color:
                    removal_count += 1
                    idx += 1

        if removal_count >= 2:
            return True, True, removal_count

        return True, False, 0

    def _simulate_shot(self) -> tuple[bool, list[str], int]:
        """Simulate optimal shot strategy."""
        color = self._frog_color
        best_directions = []
        max_removal = 0

        n = len(self._balls)
        ball_idx = 0
        while ball_idx < n:
            ball = self._balls[ball_idx]
            if ball["color"] == color:
                # Check if adjacent balls have same color
                has_adjacent = False
                if ball_idx > 0 and self._balls[ball_idx - 1]["color"] == color:
                    has_adjacent = True
                if ball_idx < n - 1 and self._balls[ball_idx + 1]["color"] == color:
                    has_adjacent = True

                if has_adjacent:
                    angle = math.degrees(
                        math.atan2(
                            ball["position"]["y"] - self._frog_pos["y"],
                            ball["position"]["x"] - self._frog_pos["x"],
                        )
                    )
                    direction = self._get_direction(angle)

                    removal_count = 1
                    left_idx, right_idx = ball_idx - 1, ball_idx + 1

                    while left_idx >= 0 and self._balls[left_idx]["color"] == color:
                        removal_count += 1
                        left_idx -= 1

                    while right_idx < n and self._balls[right_idx]["color"] == color:
                        removal_count += 1
                        right_idx += 1

                    if removal_count > max_removal:
                        max_removal = removal_count
                        best_directions = [direction]
                    elif removal_count == max_removal:
                        best_directions.append(direction)

                    ball_idx = right_idx - 1

            ball_idx += 1

        if max_removal > 0:
            return True, best_directions, max_removal
        return False, [], 0

    def _score_answer(self, answer: str) -> float:
        """Score the user's answer.

        Args:
            answer: User's answer string

        Returns:
            1.0 if correct, 0.0 otherwise
        """
        return score_exact(answer, self._oracle_answer)

    def inner_step(
        self, action: dict[str, str]
    ) -> tuple[
        dict[str, Observation],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, Any],
    ]:
        """Process the answer."""
        agent_id = next(iter(self._agent_ids))
        action_str = action[agent_id]

        action_str = action_str.strip()
        reward = self._score_answer(action_str)
        correct = reward == 1.0

        if correct:
            response = f"Correct! {self._analysis}"
        else:
            response = f"Incorrect. The correct answer is: {self._oracle_answer}\n\n{self._analysis}"

        obs = Observation(
            image=self.render(),
            text=response,
        )

        info = {
            "correct": correct,
            "user_answer": action_str,
            "oracle_answer": self._oracle_answer,
        }

        terminated = True
        truncated = False

        return (
            {agent_id: obs for agent_id in self._agent_ids},
            {agent_id: reward for agent_id in self._agent_ids},
            {
                **{agent_id: terminated for agent_id in self._agent_ids},
                "__all__": terminated,
            },
            {
                **{agent_id: truncated for agent_id in self._agent_ids},
                "__all__": truncated,
            },
            {agent_id: info for agent_id in self._agent_ids},
        )

    def render(self) -> np.ndarray | None:
        """Render the game state as an image using Matplotlib."""
        curve_x, curve_y = self._track_data

        fig, ax = plt.subplots()
        ax.set_aspect("equal")
        ax.axis("off")

        # Draw track using PathPatch
        dx = np.gradient(curve_x)
        dy = np.gradient(curve_y)
        normal_x = -dy / np.sqrt(dx**2 + dy**2)
        normal_y = dx / np.sqrt(dx**2 + dy**2)
        left_x = curve_x + normal_x * (2 * self._ball_radius) / 2
        left_y = curve_y + normal_y * (2 * self._ball_radius) / 2
        right_x = curve_x - normal_x * (2 * self._ball_radius) / 2
        right_y = curve_y - normal_y * (2 * self._ball_radius) / 2

        path_data = [
            (mpath.Path.MOVETO, (left_x[0], left_y[0])),
            *[
                (mpath.Path.LINETO, (lx, ly))
                for lx, ly in zip(left_x[1:], left_y[1:], strict=False)
            ],
            (mpath.Path.LINETO, (right_x[-1], right_y[-1])),
            *[
                (mpath.Path.LINETO, (rx, ry))
                for rx, ry in zip(
                    right_x[:-1][::-1],
                    right_y[:-1][::-1],
                    strict=False,
                )
            ],
            (mpath.Path.CLOSEPOLY, (left_x[0], left_y[0])),
        ]
        codes, verts = zip(*path_data, strict=False)
        path = mpath.Path(verts, codes)
        track_patch = patches.PathPatch(
            path,
            facecolor="lightgray",
            edgecolor="black",
            linewidth=1,
        )
        ax.add_patch(track_patch)

        # Draw direction dividers (dashed lines)
        direction_angles = [22.5, -22.5, 67.5, -67.5, 112.5, -112.5, 157.5, -157.5]
        line_length = 20
        for angle in direction_angles:
            rad = math.radians(angle)
            end_x = self._frog_pos["x"] + line_length * math.cos(rad)
            end_y = self._frog_pos["y"] + line_length * math.sin(rad)
            ax.plot(
                [self._frog_pos["x"], end_x],
                [self._frog_pos["y"], end_y],
                "k--",
                alpha=0.3,
            )

        # Draw hole
        hole = patches.Circle(
            (self._hole_pos["x"], self._hole_pos["y"]),
            2 * self._ball_radius,
            facecolor="black",
            edgecolor="black",
        )
        ax.add_patch(hole)

        # Draw balls
        for ball in self._balls:
            circle = patches.Circle(
                (ball["position"]["x"], ball["position"]["y"]),
                self._ball_radius,
                facecolor=ball["color"],
                edgecolor="black",
            )
            ax.add_patch(circle)

        # Draw frog (white triangle)
        angle_rad = math.radians(self._frog_angle)
        base = 4 * self._ball_radius
        height = 6 * self._ball_radius
        top_point = (
            self._frog_pos["x"] + height * 2 / 3 * math.cos(angle_rad),
            self._frog_pos["y"] + height * 2 / 3 * math.sin(angle_rad),
        )
        left_point = (
            self._frog_pos["x"]
            - base / 2 * math.cos(angle_rad - math.pi / 2)
            - height * 1 / 3 * math.cos(angle_rad),
            self._frog_pos["y"]
            - base / 2 * math.sin(angle_rad - math.pi / 2)
            - height * 1 / 3 * math.sin(angle_rad),
        )
        right_point = (
            self._frog_pos["x"]
            + base / 2 * math.cos(angle_rad - math.pi / 2)
            - height * 1 / 3 * math.cos(angle_rad),
            self._frog_pos["y"]
            + base / 2 * math.sin(angle_rad - math.pi / 2)
            - height * 1 / 3 * math.sin(angle_rad),
        )
        triangle = patches.Polygon(
            [top_point, left_point, right_point],
            facecolor="white",
            edgecolor="black",
        )
        ax.add_patch(triangle)

        # Draw frog's next ball color
        frog_ball = patches.Circle(
            (self._frog_pos["x"], self._frog_pos["y"]),
            self._ball_radius,
            facecolor=self._frog_color,
            edgecolor="black",
        )
        ax.add_patch(frog_ball)

        margin = 1.5 * self._ball_radius
        x_vals = [
            float(curve_x.min()),
            float(curve_x.max()),
            self._hole_pos["x"] - 2 * self._ball_radius,
            self._hole_pos["x"] + 2 * self._ball_radius,
            top_point[0],
            left_point[0],
            right_point[0],
            self._frog_pos["x"] - self._ball_radius,
            self._frog_pos["x"] + self._ball_radius,
        ]
        y_vals = [
            float(curve_y.min()),
            float(curve_y.max()),
            self._hole_pos["y"] - 2 * self._ball_radius,
            self._hole_pos["y"] + 2 * self._ball_radius,
            top_point[1],
            left_point[1],
            right_point[1],
            self._frog_pos["y"] - self._ball_radius,
            self._frog_pos["y"] + self._ball_radius,
        ]
        if self._balls:
            ball_x = [ball["position"]["x"] for ball in self._balls]
            ball_y = [ball["position"]["y"] for ball in self._balls]
            x_vals.extend([min(ball_x), max(ball_x)])
            y_vals.extend([min(ball_y), max(ball_y)])

        min_x = min(x_vals)
        max_x = max(x_vals)
        min_y = min(y_vals)
        max_y = max(y_vals)
        ax.set_xlim(min_x - margin, max_x + margin)
        ax.set_ylim(min_y - margin, max_y + margin)

        fig.canvas.draw()
        buffer = np.asarray(fig.canvas.buffer_rgba())
        image = buffer[:, :, :3].copy()
        plt.close(fig)
        return image

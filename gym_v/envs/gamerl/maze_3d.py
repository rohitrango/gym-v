"""3D Maze Q&A environment."""

from __future__ import annotations

from dataclasses import dataclass
import random
from textwrap import dedent
from typing import Any

import numpy as np
from PIL import Image

from gym_v import Env, Observation, get_logger
from gym_v.envs.gamerl.utils import build_description, score_number_choice

logger = get_logger()


@dataclass(frozen=True)
class Position:
    """3D position in the maze."""

    x: int
    y: int
    z: int

    def __add__(self, other):
        return Position(self.x + other.x, self.y + other.y, self.z + other.z)

    def to_tuple(self):
        return (self.x, self.y, self.z)


@dataclass
class PathSegment:
    """A segment of the path."""

    start: Position
    end: Position
    type: str  # 'walk' or 'ladder'


@dataclass
class Branch:
    """A branch point in the maze."""

    pos: Position
    branch_id: int


@dataclass
class SequencePoint:
    """A numbered checkpoint in the maze."""

    pos: Position
    label: int


class GameRL3dMazeQAEnv(Env):
    """3D Maze game Q&A environment.

    A 3D maze where the player must navigate from start to goal by walking on cubes
    and climbing ladders. Features branching paths and numbered checkpoints.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    QUESTION_TYPES = [
        {
            "id": "path_finding",
            "name": "Path Finding",
            "level": "Hard",
            "answer_format": "multiple_choice",
            "qa_type": "State Prediction",
        },
        {
            "id": "sequence_finding",
            "name": "Sequence Finding",
            "level": "Medium",
            "answer_format": "multiple_choice",
            "qa_type": "State Prediction",
        },
        {
            "id": "height_comparison",
            "name": "Height Comparison",
            "level": "Easy",
            "answer_format": "multiple_choice",
            "qa_type": "Target Perception",
        },
        {
            "id": "main_path",
            "name": "Main Path",
            "level": "Medium",
            "answer_format": "multiple_choice",
            "qa_type": "State Prediction",
        },
    ]

    def __init__(
        self,
        question_type: int | None = None,
        grid_size: tuple[int, int, int] = (8, 8, 7),
        num_players: int = 1,
        **kwargs,
    ):
        """Initialize 3D Maze QA environment.

        Args:
            question_type: Question type index (0-based, default: random)
            grid_size: Size of the 3D grid (width, depth, height)
        """
        super().__init__(**kwargs)
        self._question_type_param = question_type
        self._grid_size = grid_size
        self._start_pos = Position(grid_size[0] - 1, grid_size[1] - 1, 0)
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        # Game state
        self._cubes: set[Position] = set()
        self._goal_pos: Position | None = None
        self._path: list[PathSegment] = []
        self._branches: list[Branch] = []
        self._sequence_points: list[SequencePoint] = []

        # Question data (standard QA variables)
        self._question_type_idx: int = 0
        self._question: str = ""
        self._options: list[str] | None = None
        self._oracle_answer: str = ""
        self._analysis: str = ""
        self._selected_question_type: str = ""

    GAME_RULES = dedent("""
        3D Maze Game Rules:

        1. Player can only walk on top of cubes
        2. Player can climb ladders if they can reach the cube under the ladder
        3. From a ladder, player can reach the top of the last cube with the ladder
        4. Blue cube is start position, red cube is goal position
        5. Green cubes are special points (branches or checkpoints)
        6. Numbered cubes indicate branch points or checkpoints

        The player must navigate from the start (blue) to the goal (red) by walking on cubes
        and climbing ladders when available.
    """).strip()

    @property
    def description(self) -> str:
        """Return game rules + current question + answer format."""
        return build_description(
            game_name="3D Maze",
            rules=self.GAME_RULES,
            question=self._question,
            options=self._options,
            oracle_answer=self._oracle_answer,
        )

    def _get_state_text(self) -> str:
        """Generate text description of current 3D Maze game state.

        Returns a multi-layer representation of the 3D maze.
        """
        # Determine bounds
        if not self._cubes:
            return "Empty maze"

        all_positions = list(self._cubes)
        min_x = min(p.x for p in all_positions)
        max_x = max(p.x for p in all_positions)
        min_y = min(p.y for p in all_positions)
        max_y = max(p.y for p in all_positions)
        min_z = min(p.z for p in all_positions)
        max_z = max(p.z for p in all_positions)

        layers = []
        for z in range(min_z, max_z + 1):
            layer_lines = [f"Layer z={z}:"]
            for y in range(min_y, max_y + 1):
                row_chars = []
                for x in range(min_x, max_x + 1):
                    pos = Position(x, y, z)
                    if pos in self._cubes:
                        if pos == self._start_pos:
                            row_chars.append("S")
                        elif pos == self._goal_pos:
                            row_chars.append("G")
                        else:
                            row_chars.append("#")
                    else:
                        row_chars.append(".")
                layer_lines.append("".join(row_chars))
            layers.append("\n".join(layer_lines))

        return f"""3D Maze (S=start, G=goal, #=path cube, .=empty)
Start: ({self._start_pos.x}, {self._start_pos.y}, {self._start_pos.z})
Goal: ({self._goal_pos.x if self._goal_pos else 'N/A'}, {self._goal_pos.y if self._goal_pos else 'N/A'}, {self._goal_pos.z if self._goal_pos else 'N/A'})

{chr(10).join(layers)}"""

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        """Reset the environment and generate a new question."""
        super().reset(seed=seed)

        # Select question type (convert 0-based index to question type ID)
        if self._question_type_param is None:
            self._question_type_idx = random.randint(0, len(self.QUESTION_TYPES) - 1)
        else:
            # Validate question type index
            if not (0 <= self._question_type_param < len(self.QUESTION_TYPES)):
                raise ValueError(
                    f"Invalid question type index: {self._question_type_param}"
                )
            self._question_type_idx = self._question_type_param
        # Convert 0-based index to question type ID
        self._selected_question_type = self.QUESTION_TYPES[self._question_type_idx][
            "id"
        ]

        # Generate maze and question
        self._generate_maze_and_question()

        text_state = self._get_state_text()

        obs = Observation(
            image=self.render(),
            text=text_state,
            metadata={
                "text_state": text_state,
                "text_prompt": f"{text_state}\n\n{self.description}",
                "question": self._question,
                "options": self._options,
                "question_type": self.QUESTION_TYPES[self._question_type_idx]["name"],
                "level": self.QUESTION_TYPES[self._question_type_idx]["level"],
            },
        )

        logger.info(
            f"Reset 3D Maze QA (cubes={len(self._cubes)}, question: {self._selected_question_type})."
        )

        info = {
            "seed": seed,
            "oracle_answer": self._oracle_answer,
            "question_type": self._selected_question_type,
        }

        return {agent_id: obs for agent_id in self._agent_ids}, {
            agent_id: info for agent_id in self._agent_ids
        }

    def _generate_maze_and_question(self) -> None:
        """Generate the maze structure and corresponding question based on question type."""
        if self._selected_question_type == "path_finding":
            self._generate_path_finding_question()
        elif self._selected_question_type == "sequence_finding":
            self._generate_sequence_finding_question()
        elif self._selected_question_type == "height_comparison":
            self._generate_height_comparison_question()
        elif self._selected_question_type == "main_path":
            self._generate_main_path_question()

    def _generate_valid_path(
        self,
        start: Position,
        existing_cubes: set[Position],
        seg_range: tuple[int, int] = (5, 7),
    ) -> tuple[list[PathSegment], set[Position]]:
        """Generate a valid path from start position."""
        current_pos = start
        path_segments = []
        path_cubes = {start}
        all_cubes = existing_cubes | path_cubes

        num_segments = random.randint(*seg_range)

        for _ in range(num_segments):
            possible_moves = []

            # Check possible moves
            if current_pos.y - 2 > 0:
                delta = Position(0, -2, 0)
                if self._is_move_valid(current_pos, delta, all_cubes):
                    possible_moves.append(("walk", delta))

            if current_pos.x - 2 > 0:
                delta = Position(-2, 0, 0)
                if self._is_move_valid(current_pos, delta, all_cubes):
                    possible_moves.append(("walk", delta))

            if current_pos.z + 3 < self._grid_size[2]:
                delta = Position(0, 0, 3)
                if self._is_move_valid(current_pos, delta, all_cubes):
                    possible_moves.append(("ladder", delta))

            if not possible_moves:
                break

            move_type, delta = random.choice(possible_moves)
            new_pos = current_pos + delta

            segment = PathSegment(current_pos, new_pos, move_type)
            path_segments.append(segment)

            if move_type == "walk":
                intermediate_pos = Position(
                    current_pos.x + delta.x // 2,
                    current_pos.y + delta.y // 2,
                    current_pos.z,
                )
                path_cubes.add(intermediate_pos)
                all_cubes.add(intermediate_pos)

            path_cubes.add(new_pos)
            all_cubes.add(new_pos)
            current_pos = new_pos

        return path_segments, path_cubes

    def _is_move_valid(
        self, start_pos: Position, delta: Position, existing_cubes: set[Position]
    ) -> bool:
        """Check if a move is valid."""
        new_pos = start_pos + delta

        # Check grid boundaries
        if not (
            0 <= new_pos.x < self._grid_size[0]
            and 0 <= new_pos.y < self._grid_size[1]
            and 0 <= new_pos.z < self._grid_size[2]
        ):
            return False

        # Handle walking paths
        if delta.z == 0:
            intermediate_pos = Position(
                start_pos.x + delta.x // 2, start_pos.y + delta.y // 2, start_pos.z
            )
            if intermediate_pos in existing_cubes:
                return False

        # Check end position
        if new_pos in existing_cubes:
            return False

        return True

    def _get_ordered_path_cubes(self, path: list[PathSegment]) -> list[Position]:
        """Convert path segments into an ordered list of cubes."""
        ordered_cubes = []

        if path:
            ordered_cubes.append(path[0].start)

        for segment in path:
            if segment.type == "walk":
                delta = Position(
                    segment.end.x - segment.start.x,
                    segment.end.y - segment.start.y,
                    segment.end.z - segment.start.z,
                )
                intermediate_pos = Position(
                    segment.start.x + delta.x // 2,
                    segment.start.y + delta.y // 2,
                    segment.start.z,
                )
                ordered_cubes.append(intermediate_pos)
            ordered_cubes.append(segment.end)

        return ordered_cubes

    def _get_segment_direction(self, segment: PathSegment) -> str:
        """Determine the direction of a path segment."""
        if segment.type == "ladder":
            return "up"

        delta_x = segment.end.x - segment.start.x
        delta_y = segment.end.y - segment.start.y

        if delta_x == 0:
            return "left-forward"
        elif delta_y == 0:
            return "right-forward"
        return "unknown"

    def _generate_path_finding_question(self):
        """Generate a path-finding question with branches."""
        # Clear previous state
        self._branches = []
        self._sequence_points = []

        # Generate main path
        main_path, main_cubes = self._generate_valid_path(self._start_pos, set())
        self._goal_pos = main_path[-1].end
        self._path = main_path
        self._cubes = main_cubes.copy()

        # Generate branches
        ordered_cubes = self._get_ordered_path_cubes(main_path)
        valid_branch_positions = [
            pos for pos in ordered_cubes[1:-3] if pos.x % 2 == 1 and pos.y % 2 == 1
        ]

        num_branches = min(len(valid_branch_positions), random.randint(2, 3))
        random.shuffle(valid_branch_positions)
        selected_positions = valid_branch_positions[:num_branches]

        for i, branch_pos in enumerate(selected_positions, 1):
            self._branches.append(Branch(branch_pos, i))
            # Generate side path
            side_path, side_cubes = self._generate_valid_path(
                branch_pos, self._cubes, seg_range=(1, 2)
            )
            if side_path:
                self._cubes.update(side_cubes)

        # Build correct path
        correct_path = ""
        for b in self._branches:
            direction = self._get_main_path_direction_at_branch(b.pos)
            correct_path += f"{b.branch_id}-{direction}, "
        correct_path = correct_path[:-2]

        # Generate options
        self._options = [correct_path]
        for _ in range(7):
            wrong_path = ""
            for b in self._branches:
                directions = ["left-forward", "right-forward", "up"]
                direction = random.choice(directions)
                wrong_path += f"{b.branch_id}-{direction}, "
            wrong_path = wrong_path[:-2]
            if wrong_path not in self._options:
                self._options.append(wrong_path)

        random.shuffle(self._options)
        correct_answer = self._options.index(correct_path) + 1

        self._question = "Which combination of path choices leads to the goal?"
        self._options = [f"{i+1}. {opt}" for i, opt in enumerate(self._options)]
        self._oracle_answer = str(correct_answer)
        self._analysis = (
            f"Analyzing each branch point, the correct path choices are: {correct_path}. "
            f"This makes the answer Option {correct_answer}."
        )

    def _get_main_path_direction_at_branch(self, branch_pos: Position) -> str:
        """Get the direction of the main path at a branch position."""
        for segment in self._path:
            if segment.start == branch_pos:
                return self._get_segment_direction(segment)

            if segment.type == "walk":
                delta = Position(
                    segment.end.x - segment.start.x,
                    segment.end.y - segment.start.y,
                    segment.end.z - segment.start.z,
                )
                intermediate_pos = Position(
                    segment.start.x + delta.x // 2,
                    segment.start.y + delta.y // 2,
                    segment.start.z,
                )
                if intermediate_pos == branch_pos:
                    for next_segment in self._path:
                        if next_segment.start == segment.end:
                            return self._get_segment_direction(next_segment)
        return "unknown"

    def _generate_sequence_finding_question(self):
        """Generate a sequence-finding question with numbered checkpoints."""
        # Clear previous state
        self._branches = []
        self._sequence_points = []

        # Generate main path
        main_path, main_cubes = self._generate_valid_path(self._start_pos, set())
        self._goal_pos = main_path[-1].end
        self._path = main_path
        self._cubes = main_cubes

        # Choose sequence points
        ordered_cubes = self._get_ordered_path_cubes(main_path)
        valid_positions = ordered_cubes[1:-1]

        num_labels = min(len(valid_positions), random.randint(3, 4))
        selected_positions = random.sample(valid_positions, num_labels)

        # Sort by order in path
        for pos in ordered_cubes:
            if pos in selected_positions:
                self._sequence_points.append(
                    SequencePoint(pos, len(self._sequence_points) + 1)
                )

        # Get correct sequence
        correct_sequence = (
            "Start -> "
            + " -> ".join(str(sp.label) for sp in self._sequence_points)
            + " -> Goal"
        )

        # Generate wrong options
        labels = [sp.label for sp in self._sequence_points]
        self._options = [correct_sequence]

        max_attempts = 100
        attempts = 0
        while len(self._options) < 8 and attempts < max_attempts:
            attempts += 1
            shuffled_labels = labels.copy()
            random.shuffle(shuffled_labels)
            wrong_sequence = (
                "Start -> "
                + " -> ".join(str(label) for label in shuffled_labels)
                + " -> Goal"
            )
            if wrong_sequence not in self._options:
                self._options.append(wrong_sequence)

        random.shuffle(self._options)
        correct_answer = self._options.index(correct_sequence) + 1

        self._question = "What is the correct sequence of numbered checkpoints when following the path from start to goal?"
        self._options = [f"{i+1}. {opt}" for i, opt in enumerate(self._options)]
        self._oracle_answer = str(correct_answer)
        self._analysis = (
            f"Following the path from start to goal, the correct sequence is: {correct_sequence}. "
            f"This makes the answer Option {correct_answer}."
        )

    def _generate_height_comparison_question(self):
        """Generate a height comparison question."""
        # Clear previous state
        self._branches = []
        self._sequence_points = []

        # Generate maze with 3 checkpoints
        main_path, main_cubes = self._generate_valid_path(self._start_pos, set())
        self._goal_pos = main_path[-1].end
        self._path = main_path
        self._cubes = main_cubes

        # Choose 3 sequence points
        ordered_cubes = self._get_ordered_path_cubes(main_path)
        valid_positions = ordered_cubes[1:-1]

        num_points = min(3, len(valid_positions))
        selected_positions = random.sample(valid_positions, num_points)

        for i, pos in enumerate(selected_positions, 1):
            self._sequence_points.append(SequencePoint(pos, i))

        # Get heights
        heights = [(sp.label, sp.pos.z) for sp in self._sequence_points]
        heights.sort(key=lambda x: (x[1], x[0]))

        # Generate correct relation
        correct_relation = self._normalize_height_relation(heights)

        # Possible relations for 3 points
        possible_relations = [
            "1 < 2 < 3",
            "1 < 3 < 2",
            "2 < 1 < 3",
            "2 < 3 < 1",
            "3 < 1 < 2",
            "3 < 2 < 1",
            "1 < 2 = 3",
            "2 < 1 = 3",
            "3 < 1 = 2",
            "1 = 2 < 3",
            "1 = 3 < 2",
            "2 = 3 < 1",
            "1 = 2 = 3",
        ]

        self._options = [correct_relation]
        while len(self._options) < 8:
            wrong_relation = random.choice(possible_relations)
            if wrong_relation not in self._options:
                self._options.append(wrong_relation)

        random.shuffle(self._options)
        correct_answer = self._options.index(correct_relation) + 1

        self._question = (
            "What is the correct height relationship between the three numbered points? "
            "Use '<' for 'lower than' and '=' for 'same height as'."
        )
        self._options = [f"{i+1}. {opt}" for i, opt in enumerate(self._options)]
        self._oracle_answer = str(correct_answer)
        self._analysis = (
            f"Analyzing the heights of the numbered points, the correct relationship is: {correct_relation}. "
            f"This makes the answer Option {correct_answer}."
        )

    def _normalize_height_relation(self, heights: list[tuple[int, int]]) -> str:
        """Convert heights into a relation string."""
        sorted_points = sorted(heights, key=lambda x: (x[1], x[0]))

        relations = []
        current_height = None
        current_group = []

        for label, height in sorted_points:
            if height != current_height:
                if current_group:
                    relations.append(current_group)
                current_group = [label]
                current_height = height
            else:
                current_group.append(label)
        relations.append(current_group)

        result = []
        for group in relations:
            group.sort()
            result.append(" = ".join(str(x) for x in group))

        return " < ".join(result)

    def _generate_main_path_question(self):
        """Generate a question about which blocks are on the main path."""
        # Clear previous state
        self._branches = []
        self._sequence_points = []

        # Generate main path
        main_path, main_cubes = self._generate_valid_path(self._start_pos, set())
        self._goal_pos = main_path[-1].end
        self._path = main_path
        self._cubes = main_cubes.copy()

        # Add some labeled cubes
        ordered_cubes = self._get_ordered_path_cubes(main_path)
        all_cubes = list(self._cubes - {self._start_pos, self._goal_pos})

        num_labels = min(len(all_cubes), random.randint(3, 4))
        labeled_cubes = random.sample(all_cubes, num_labels)

        main_path_labels = []
        side_path_labels = []

        for i, pos in enumerate(labeled_cubes, 1):
            self._branches.append(Branch(pos, i))
            if pos in ordered_cubes:
                main_path_labels.append(str(i))
            else:
                side_path_labels.append(str(i))

        # Generate correct answer
        correct_answer = (
            ", ".join(sorted(main_path_labels, key=int)) if main_path_labels else "None"
        )

        # Generate wrong options
        all_labels = [str(b.branch_id) for b in self._branches]
        self._options = [correct_answer]

        while len(self._options) < 8:
            num_choices = random.randint(0, len(all_labels))
            if num_choices == 0:
                wrong_answer = "None"
            else:
                wrong_labels = random.sample(all_labels, num_choices)
                wrong_answer = ", ".join(sorted(wrong_labels, key=int))
            if wrong_answer not in self._options:
                self._options.append(wrong_answer)

        random.shuffle(self._options)
        correct_option = self._options.index(correct_answer) + 1

        self._question = "Which numbered blocks are passed through when following the most direct path from start to goal?"
        self._options = [f"{i+1}. {opt}" for i, opt in enumerate(self._options)]
        self._oracle_answer = str(correct_option)
        self._analysis = (
            f"Following the main path, the numbered blocks passed through are: {correct_answer}. "
            f"This makes the answer Option {correct_option}."
        )

    def _score_answer(self, answer: str) -> float:
        """Score the user's answer.

        Args:
            answer: User's answer string

        Returns:
            1.0 if correct, 0.0 otherwise
        """
        return score_number_choice(answer, self._oracle_answer)

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

    def render(self) -> Image.Image | list[Image.Image] | None:
        """Render the 3D maze as an image using matplotlib 3D visualization."""
        import io

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        # Create figure with tight layout
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")

        # Common view and styling settings
        ax.view_init(elev=25, azim=30)
        ax.grid(False)
        ax.set_facecolor("white")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_axis_off()

        margin = 0.5
        ax.set_xlim(-margin, self._grid_size[0] + margin)
        ax.set_ylim(-margin, self._grid_size[1] + margin)
        ax.set_zlim(-margin, self._grid_size[2] + margin)

        # Draw base grid
        x = np.arange(-margin, self._grid_size[0] + margin, 1)
        y = np.arange(-margin, self._grid_size[1] + margin, 1)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        ax.plot_surface(X, Y, Z, alpha=0.1, color="gray", zorder=1)

        # Helper function to create cube vertices (only visible faces)
        def create_cube_verts(pos: Position, cubes_set: set) -> list:
            """Create vertices for a cube at given position, omitting hidden faces."""
            x, y, z = pos.x, pos.y, pos.z
            verts = []

            # Only add top face if there's no cube above
            if Position(x, y, z + 1) not in cubes_set:
                verts.append(
                    [
                        (x, y, z + 1),
                        (x + 1, y, z + 1),
                        (x + 1, y + 1, z + 1),
                        (x, y + 1, z + 1),
                    ]
                )

            # Only add right face if there's no cube to the right
            if Position(x + 1, y, z) not in cubes_set:
                verts.append(
                    [
                        (x + 1, y, z),
                        (x + 1, y + 1, z),
                        (x + 1, y + 1, z + 1),
                        (x + 1, y, z + 1),
                    ]
                )

            # Only add back face if there's no cube behind
            if Position(x, y + 1, z) not in cubes_set:
                verts.append(
                    [
                        (x, y + 1, z),
                        (x + 1, y + 1, z),
                        (x + 1, y + 1, z + 1),
                        (x, y + 1, z + 1),
                    ]
                )

            return verts

        # Sort cubes by distance from camera for proper transparency
        camera_pos = np.array(
            [self._grid_size[0] + 2, self._grid_size[1] + 2, self._grid_size[2] + 2]
        )
        sorted_cubes = sorted(
            self._cubes,
            key=lambda pos: -np.linalg.norm(
                np.array([pos.x, pos.y, pos.z]) - camera_pos
            ),
        )

        cubes_set = set(self._cubes)

        # Draw cubes
        for cube_pos in sorted_cubes:
            verts = create_cube_verts(cube_pos, cubes_set)

            alpha = 0.8

            # Determine cube color based on type
            if cube_pos == self._start_pos:
                color = "#FF4444"  # Red for start (note: swapped in original)
            elif cube_pos == self._goal_pos:
                color = "#4444FF"  # Blue for goal (note: swapped in original)
            elif any(sp.pos == cube_pos for sp in self._sequence_points):
                color = "#44FF44"  # Green for sequence points
            elif any(b.pos == cube_pos for b in self._branches):
                color = "#44FF44"  # Green for branches
            else:
                color = "#888888"  # Gray for regular cubes

            pc = Poly3DCollection(verts, alpha=alpha, zorder=2)
            pc.set_facecolor(color)
            pc.set_edgecolor("black")
            pc.set_linewidth(1.0)
            ax.add_collection3d(pc)

        # Draw ladders
        for segment in self._path:
            if segment.type == "ladder":
                base = segment.start
                height = segment.end.z - segment.start.z
                x, y, z = base.x + 0.5, base.y, base.z + 0.5
                ax.plot([x, x], [y, y], [z, z + height], "k-", linewidth=3, zorder=300)
                ax.plot(
                    [x, x],
                    [y + 0.2, y + 0.2],
                    [z, z + height],
                    "k-",
                    linewidth=3,
                    zorder=300,
                )
                for h in np.linspace(z, z + height, 6):
                    ax.plot([x, x], [y, y + 0.2], [h, h], "k-", linewidth=2, zorder=300)

        # Draw sequence point numbers or branch numbers
        for sp in self._sequence_points:
            # Draw white outline
            for dx, dy in [
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
                (-1, -1),
                (-1, 1),
                (1, -1),
                (1, 1),
            ]:
                ax.text(
                    sp.pos.x + 0.5 + dx * 0.01,
                    sp.pos.y + 0.5 + dy * 0.01,
                    sp.pos.z + 1.1,
                    str(sp.label),
                    size=24,
                    ha="center",
                    va="center",
                    weight="bold",
                    color="white",
                    zorder=400,
                    clip_on=True,
                )

            # Draw main text in black
            ax.text(
                sp.pos.x + 0.5,
                sp.pos.y + 0.5,
                sp.pos.z + 1.1,
                str(sp.label),
                size=24,
                ha="center",
                va="center",
                weight="bold",
                color="black",
                zorder=400,
                clip_on=True,
            )

        for b in self._branches:
            # Draw white outline
            for dx, dy in [
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
                (-1, -1),
                (-1, 1),
                (1, -1),
                (1, 1),
            ]:
                ax.text(
                    b.pos.x + 0.5 + dx * 0.01,
                    b.pos.y + 0.5 + dy * 0.01,
                    b.pos.z + 1.1,
                    str(b.branch_id),
                    size=24,
                    ha="center",
                    va="center",
                    weight="bold",
                    color="white",
                    zorder=400,
                    clip_on=True,
                )

            # Draw main text in black
            ax.text(
                b.pos.x + 0.5,
                b.pos.y + 0.5,
                b.pos.z + 1.1,
                str(b.branch_id),
                size=24,
                ha="center",
                va="center",
                weight="bold",
                color="black",
                zorder=400,
                clip_on=True,
            )

        # Set axis limits (set twice - first with margin, then without)
        # This matches the original implementation behavior
        ax.set_xlim(0, self._grid_size[0])
        ax.set_ylim(0, self._grid_size[1])
        ax.set_zlim(0, self._grid_size[2])

        # Remove extra white space
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

        # Save with tight bbox (matching original pad_inches=-0.3)
        buf = io.BytesIO()
        plt.savefig(
            buf,
            format="png",
            bbox_inches="tight",
            pad_inches=-0.3,
            facecolor="white",
            dpi=100,
        )
        plt.close(fig)
        buf.seek(0)

        return Image.open(buf).convert("RGB")

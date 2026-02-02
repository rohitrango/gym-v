"""Rubik's Cube QA environment based on GameRL."""

from __future__ import annotations

from importlib import resources
import random
from textwrap import dedent
from typing import Any

import numpy as np
from PIL import Image

from gym_v import Env, Observation, get_logger
from gym_v.envs.gamerl.parameter_controllers import get_controller_for_env
from gym_v.envs.gamerl.utils import build_description, score_exact

logger = get_logger()


class GameRLRubiksCubeQAEnv(Env):
    """Rubik's Cube QA environment.

    A 3x3x3 Rubik's Cube puzzle with 6 faces that can be rotated.

    Question Types:
    - Face Recognition: Identify color at a specific position
    - Color Count: Count how many squares of a specific color on a face
    - Move Prediction: Predict cube state after moves

    Args:
        num_moves: Number of random moves to scramble the cube
        question_type: Type of question to ask
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    QUESTION_TYPES = [
        {
            "id": "face_recognition",
            "name": "Face Recognition",
            "level": "Easy",
            "answer_format": "multiple_choice",
            "qa_type": "StateInfo",
        },
        {
            "id": "color_count",
            "name": "Color Count",
            "level": "Easy",
            "answer_format": "multiple_choice",
            "qa_type": "StateInfo",
        },
        {
            "id": "move_prediction",
            "name": "Move Prediction",
            "level": "Medium",
            "answer_format": "multiple_choice",
            "qa_type": "ActionOutcome",
        },
    ]

    # Color mappings
    COLORS = {
        0: ("yellow", (255, 255, 0)),
        1: ("white", (255, 255, 255)),
        2: ("orange", (255, 165, 0)),
        3: ("red", (255, 0, 0)),
        4: ("blue", (0, 0, 255)),
        5: ("green", (0, 255, 0)),
    }

    FACE_NAMES = {
        "U": "Upper face",
        "D": "Down face",
        "L": "Left face",
        "R": "Right face",
        "F": "Front face",
        "B": "Back face",
    }

    RUBIKS_RULES = dedent("""
        The Rubik's cube has six faces: Upper (U), Down (D), Left (L), Right (R), Front (F), and Back (B).
        Each face is a 3x3 grid with coordinates (row, col) where row and col go from 0 to 2.
        For each face, coordinates are determined: column increases from left to right (0,1,2) and row increases from bottom to top (0,1,2). The bottom-left cell is (0, 0).
        Moves: An uppercase letter indicates which face to rotate, with a prime symbol (') denoting counterclockwise rotation.
    """).strip()

    def __init__(
        self,
        num_moves: int | None = None,
        question_type: int | None = None,
        num_players: int = 1,
        difficulty: int | None = None,
        **kwargs,
    ):
        super().__init__(difficulty=difficulty, **kwargs)

        self._question_type_param = question_type
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        # Check if explicit params or difficulty is provided
        self._use_explicit_params = num_moves is not None
        self._use_difficulty = difficulty is not None

        self._parameter_controller = get_controller_for_env(
            self.__class__.__name__,
            self._difficulty if self._difficulty is not None else 0,
        )

        # Initialize num_moves based on priority
        if self._use_explicit_params:
            self._num_moves = num_moves
        elif self._use_difficulty:
            self._apply_difficulty_parameters()
        else:
            self._num_moves = random.randint(1, 3)

    def _apply_difficulty_parameters(self) -> None:
        """Apply parameters from the controller."""
        if self._use_difficulty and self._parameter_controller is not None:
            params = self._parameter_controller.get_parameters()
            if "scramble_depth" in params:
                self._num_moves = params["scramble_depth"]
            else:
                self._num_moves = random.randint(1, 3)
        else:
            self._num_moves = random.randint(1, 3)

    @property
    def description(self) -> str:
        """Return game rules + current question + answer format."""
        return build_description(
            game_name="Rubik's Cube",
            rules=self.RUBIKS_RULES,
            question=self._question,
            options=self._options,
            oracle_answer=self._oracle_answer,
        )

    def _get_state_text(self) -> str:
        """Generate text description of current Rubik's Cube state."""
        text = "Rubik's Cube State\n\n"

        for face_name, face_array in self._faces.items():
            text += f"{face_name} Face ({self.FACE_NAMES[face_name]}):\n"
            for row in face_array:
                color_names = [self.COLORS[c][0] for c in row]
                text += f"  {' '.join(color_names)}\n"
            text += "\n"

        text += "Color Legend:\n"
        for idx in sorted(self.COLORS.keys()):
            color_name, _ = self.COLORS[idx]
            text += f"  {idx}: {color_name}\n"

        return text.strip()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        super().reset(seed=seed)

        # Reset cube to solved state
        self._faces = {
            "U": np.zeros((3, 3), dtype=int),
            "D": np.ones((3, 3), dtype=int),
            "L": np.full((3, 3), 2, dtype=int),
            "R": np.full((3, 3), 3, dtype=int),
            "F": np.full((3, 3), 4, dtype=int),
            "B": np.full((3, 3), 5, dtype=int),
        }

        # Scramble the cube
        moves = ["F", "B", "L", "R", "U", "D", "F'", "B'", "L'", "R'", "U'", "D'"]
        for _ in range(self._num_moves):
            move = random.choice(moves)
            self._make_move(move)

        # Select question type
        if self._question_type_param is None:
            self._question_type_idx = random.randint(0, len(self.QUESTION_TYPES) - 1)
        else:
            self._question_type_idx = self._question_type_param

        # Validate question type index
        if not (0 <= self._question_type_idx < len(self.QUESTION_TYPES)):
            raise ValueError(f"Invalid question type index: {self._question_type_idx}")

        q_type = self.QUESTION_TYPES[self._question_type_idx]

        # Generate question
        if q_type["id"] == "face_recognition":
            result = self._generate_face_recognition_question()
        elif q_type["id"] == "color_count":
            result = self._generate_color_count_question()
        elif q_type["id"] == "move_prediction":
            result = self._generate_move_prediction_question()
        else:
            raise ValueError(f"Unknown question type: {q_type['id']}")

        # Extract to instance variables
        self._question = result["question"]
        self._options = result.get("options")
        self._oracle_answer = result["answer"]

        logger.info(
            f"Reset Rubik's Cube QA (num_moves={self._num_moves}, question: {q_type['id']})."
        )

        # Generate text state
        text_state = self._get_state_text()

        obs = Observation(
            image=self.render(),
            text=None,
            metadata={
                "state_text": text_state,
                "text_prompt": f"{text_state}\n\n{self.description}",
                "question": self._question,
                "options": self._options,
                "question_type": q_type["name"],
                "level": q_type["level"],
            },
        )

        info = {
            "seed": seed,
            "oracle_answer": self._oracle_answer,
            "question_type": q_type["id"],
        }

        return {agent_id: obs for agent_id in self._agent_ids}, {
            agent_id: info for agent_id in self._agent_ids
        }

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
        agent_id = next(iter(self._agent_ids))
        action_str = action[agent_id]

        info: dict[str, Any] = {}
        terminated = True
        truncated = False

        # Check answer
        reward = self._score_answer(action_str)
        correct = reward == 1.0

        if correct:
            response = "Correct!"
        else:
            response = f"Incorrect. The correct answer is: {self._oracle_answer}"

        info = {
            "correct": correct,
            "user_answer": action_str.strip(),
            "oracle_answer": self._oracle_answer,
        }

        obs = Observation(image=self.render(), text=response)

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
        """Render the Rubik's Cube with 3D views using matplotlib."""
        import io

        import matplotlib.pyplot as plt

        # Create figure with improved size ratio
        fig = plt.figure(figsize=(12, 9))

        # Create grid with better proportions - left side split into 70% top, 30% bottom
        grid = plt.GridSpec(10, 2, width_ratios=[2, 1], hspace=0.3, wspace=0.15)

        # Draw main unfolded cube (top left, larger size)
        ax_main = fig.add_subplot(grid[0:7, 0])  # Takes up first 7 units vertically
        self._draw_unfolded_cube_improved(ax_main)

        # Draw coordinate reference (bottom left)
        ax_coord = fig.add_subplot(grid[7:, 0])  # Takes up remaining 3 units vertically
        self._draw_coordinate_reference(ax_coord)

        # Draw 3D views (right column)
        ax_front = fig.add_subplot(grid[0:5, 1], projection="3d")
        self._draw_3d_cube_improved(ax_front, angle_front=30, angle_side=45)
        ax_front.set_title("Front view (F, R, U faces)", pad=8, fontsize=16)

        ax_back = fig.add_subplot(grid[5:, 1], projection="3d")
        self._draw_3d_cube_improved(ax_back, angle_front=-30, angle_side=225)
        ax_back.set_title("Back view (L, D, B faces)", pad=8, fontsize=16)

        # Save with higher resolution
        buf = io.BytesIO()
        plt.savefig(buf, format="PNG", bbox_inches="tight", dpi=65)
        plt.close(fig)
        buf.seek(0)

        return Image.open(buf).convert("RGB")

    def _draw_unfolded_cube_improved(self, ax):
        """Draw improved unfolded cube layout with thicker borders but no coordinates."""
        ax.set_aspect("equal")
        ax.axis("off")

        # Standard layout positions
        layouts = {
            "U": (0, 3),  # Upper face at top
            "L": (3, 0),  # Left face on left
            "F": (3, 3),  # Front face in center
            "R": (3, 6),  # Right face on right
            "B": (3, 9),  # Back face on far right
            "D": (6, 3),  # Down face at bottom
        }

        # Draw faces with thicker borders
        for face, (row_offset, col_offset) in layouts.items():
            # Draw face outline with thicker border
            import matplotlib.pyplot as plt

            rect = plt.Rectangle(
                (col_offset, row_offset),
                3,
                3,
                fill=False,
                edgecolor="black",
                linewidth=2,
            )
            ax.add_patch(rect)

            # Draw individual cells
            for i in range(3):
                for j in range(3):
                    color_name, _ = self.COLORS[self._faces[face][i, j]]
                    cell = plt.Rectangle(
                        (col_offset + j, row_offset + i),
                        1,
                        1,
                        facecolor=color_name,
                        edgecolor="black",
                        linewidth=1,
                    )
                    ax.add_patch(cell)

            # Add face labels with better positioning
            label_offset = 1.0  # Reduced offset for more compact layout
            if face in ["U", "D", "L", "B", "R"]:
                ax.annotate(
                    f"{face}",
                    xy=(col_offset + 1.5, row_offset + 1.5),
                    xytext=(
                        col_offset + 1.5,
                        row_offset + (-label_offset if face == "U" else 4),
                    ),
                    ha="center",
                    va="center",
                    fontsize=20,
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
                    arrowprops=dict(arrowstyle="->", linewidth=1.5),
                )
            else:
                ax.annotate(
                    f"{face}",
                    xy=(col_offset + 1.5, row_offset + 1.5),
                    xytext=(col_offset + 4.5, row_offset - label_offset),
                    va="center",
                    fontsize=20,
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
                    arrowprops=dict(arrowstyle="->", linewidth=1.5),
                )

        ax.set_xlim(-1, 13)
        ax.set_ylim(-1, 10)

    def _draw_coordinate_reference(self, ax):
        """Draw a reference grid showing the coordinate system with pink background."""
        import matplotlib.pyplot as plt

        ax.set_aspect("equal")
        ax.axis("off")

        # Draw title first
        ax.text(
            1.5,
            3.2,
            "Coordinate Reference",
            ha="center",
            va="bottom",
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
        )

        # Draw 3x3 grid with coordinates
        coords = [
            (2, 0),
            (2, 1),
            (2, 2),
            (1, 0),
            (1, 1),
            (1, 2),
            (0, 0),
            (0, 1),
            (0, 2),
        ]

        for idx, (i, j) in enumerate(coords):
            row = idx // 3
            col = idx % 3
            # Draw cell with pink background
            rect = plt.Rectangle(
                (col, 2 - row), 1, 1, facecolor="pink", edgecolor="black", linewidth=1.5
            )
            ax.add_patch(rect)

            # Add coordinates with larger font
            ax.text(
                col + 0.5,
                2 - row + 0.5,
                f"({i},{j})",
                ha="center",
                va="center",
                fontsize=12,
                weight="bold",
            )

        ax.set_xlim(-0.2, 3.2)
        ax.set_ylim(-0.2, 3.5)

    def _draw_3d_cube_improved(self, ax, angle_front: int, angle_side: int):
        """Draw improved 3D view with better visibility."""
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        scale = 3
        cell_size = scale / 3

        def draw_cell(vertices, color):
            """Draw a single colored cell with thicker edges"""
            x = [v[0] for v in vertices]
            y = [v[1] for v in vertices]
            z = [v[2] for v in vertices]
            verts = [list(zip(x, y, z, strict=False))]
            poly = Poly3DCollection(verts)
            poly.set_color(color)
            poly.set_alpha(1.0)
            poly.set_edgecolor("black")
            poly.set_linewidth(1.5)
            ax.add_collection3d(poly)

        def is_face_visible(face: str, elev: float, azim: float) -> bool:
            """Determine if a face should be visible from the current viewing angle."""
            elev_rad = np.radians(elev)
            azim_rad = np.radians(azim)

            view_vector = np.array(
                [
                    np.cos(elev_rad) * np.sin(azim_rad),
                    np.cos(elev_rad) * np.cos(azim_rad),
                    np.sin(elev_rad),
                ]
            )

            normals = {
                "F": np.array([0, 0, 1]),
                "B": np.array([0, 0, -1]),
                "R": np.array([1, 0, 0]),
                "L": np.array([-1, 0, 0]),
                "U": np.array([0, 1, 0]),
                "D": np.array([0, -1, 0]),
            }

            normal = normals[face]
            visibility = np.dot(view_vector, normal)

            return visibility > -0.2

        def map_coordinates(face, i, j):
            """Map coordinates from 2D to 3D space."""
            if face == "F":  # Front face
                return [
                    (j * cell_size, (2 - i) * cell_size, scale),
                    ((j + 1) * cell_size, (2 - i) * cell_size, scale),
                    ((j + 1) * cell_size, (3 - i) * cell_size, scale),
                    (j * cell_size, (3 - i) * cell_size, scale),
                ]
            elif face == "B":  # Back face
                return [
                    ((2 - j) * cell_size, (2 - i) * cell_size, 0),
                    ((3 - j) * cell_size, (2 - i) * cell_size, 0),
                    ((3 - j) * cell_size, (3 - i) * cell_size, 0),
                    ((2 - j) * cell_size, (3 - i) * cell_size, 0),
                ]
            elif face == "R":  # Right face
                return [
                    (scale, (2 - i) * cell_size, scale - j * cell_size),
                    (scale, (2 - i) * cell_size, scale - (j + 1) * cell_size),
                    (scale, (3 - i) * cell_size, scale - (j + 1) * cell_size),
                    (scale, (3 - i) * cell_size, scale - j * cell_size),
                ]
            elif face == "L":  # Left face
                return [
                    (0, (2 - i) * cell_size, j * cell_size),
                    (0, (2 - i) * cell_size, (j + 1) * cell_size),
                    (0, (3 - i) * cell_size, (j + 1) * cell_size),
                    (0, (3 - i) * cell_size, j * cell_size),
                ]
            elif face == "U":  # Upper face
                return [
                    (j * cell_size, scale, scale - (2 - i) * cell_size),
                    ((j + 1) * cell_size, scale, scale - (2 - i) * cell_size),
                    ((j + 1) * cell_size, scale, scale - (3 - i) * cell_size),
                    (j * cell_size, scale, scale - (3 - i) * cell_size),
                ]
            elif face == "D":  # Down face
                return [
                    (j * cell_size, 0, (2 - i) * cell_size),
                    ((j + 1) * cell_size, 0, (2 - i) * cell_size),
                    ((j + 1) * cell_size, 0, (3 - i) * cell_size),
                    (j * cell_size, 0, (3 - i) * cell_size),
                ]

        # Draw all faces
        faces = ["F", "B", "U", "D", "L", "R"]
        for face in faces:
            if is_face_visible(face, angle_front, angle_side):
                for i in range(3):
                    for j in range(3):
                        color_name, _ = self.COLORS[self._faces[face][i, j]]
                        vertices = map_coordinates(face, i, j)
                        draw_cell(vertices, color_name)

        # Update label positions
        label_positions = {
            "F": (scale / 2, scale / 2, scale + 0.01),
            "B": (scale / 2, scale / 2, -0.01),
            "R": (scale + 0.01, scale / 2, scale / 2),
            "L": (-0.01, scale / 2, scale / 2),
            "U": (scale / 2, scale + 0.01, scale / 2),
            "D": (scale / 2, -0.01, scale / 2),
        }

        # Draw labels with improved visibility
        for face, pos in label_positions.items():
            if is_face_visible(face, angle_front, angle_side):
                x, y, z = pos
                ax.text(
                    x,
                    y,
                    z,
                    face,
                    fontsize=30,
                    color="black",
                    ha="center",
                    va="center",
                    weight="bold",
                    bbox=dict(
                        boxstyle="square,pad=0.3",
                        facecolor="white",
                        edgecolor="black",
                        linewidth=2,
                        alpha=0.9,
                    ),
                )

        # Set view angle and appearance
        ax.view_init(elev=angle_front, azim=angle_side)
        ax.set_xlim([0, scale])
        ax.set_ylim([0, scale])
        ax.set_zlim([0, scale])
        ax.set_box_aspect([1, 1, 1])

        # Remove axes and grid
        ax.set_axis_off()
        ax.grid(False)

    def _make_move(self, move: str):
        """Execute a move on the cube."""
        if move.endswith("'"):
            # Counterclockwise - rotate 3 times clockwise
            face = move[0]
            for _ in range(3):
                self._rotate_face_clockwise(face)
        else:
            # Clockwise
            self._rotate_face_clockwise(move)

    def _rotate_face_clockwise(self, face: str):
        """Rotate a face clockwise."""
        # Rotate the face itself
        self._faces[face] = np.rot90(self._faces[face], k=-1)

        # Rotate adjacent edges
        if face == "F":
            temp = self._faces["U"][2, :].copy()
            self._faces["U"][2, :] = np.flip(self._faces["L"][:, 2])
            self._faces["L"][:, 2] = self._faces["D"][0, :]
            self._faces["D"][0, :] = np.flip(self._faces["R"][:, 0])
            self._faces["R"][:, 0] = temp

        elif face == "B":
            temp = self._faces["U"][0, :].copy()
            self._faces["U"][0, :] = self._faces["R"][:, 2]
            self._faces["R"][:, 2] = np.flip(self._faces["D"][2, :])
            self._faces["D"][2, :] = self._faces["L"][:, 0]
            self._faces["L"][:, 0] = np.flip(temp)

        elif face == "L":
            temp = self._faces["U"][:, 0].copy()
            self._faces["U"][:, 0] = np.flip(self._faces["B"][:, 2])
            self._faces["B"][:, 2] = np.flip(self._faces["D"][:, 0])
            self._faces["D"][:, 0] = self._faces["F"][:, 0]
            self._faces["F"][:, 0] = temp

        elif face == "R":
            temp = self._faces["U"][:, 2].copy()
            self._faces["U"][:, 2] = self._faces["F"][:, 2]
            self._faces["F"][:, 2] = self._faces["D"][:, 2]
            self._faces["D"][:, 2] = np.flip(self._faces["B"][:, 0])
            self._faces["B"][:, 0] = np.flip(temp)

        elif face == "U":
            temp = self._faces["F"][0, :].copy()
            self._faces["F"][0, :] = self._faces["R"][0, :]
            self._faces["R"][0, :] = self._faces["B"][0, :]
            self._faces["B"][0, :] = self._faces["L"][0, :]
            self._faces["L"][0, :] = temp

        elif face == "D":
            temp = self._faces["F"][2, :].copy()
            self._faces["F"][2, :] = self._faces["L"][2, :]
            self._faces["L"][2, :] = self._faces["B"][2, :]
            self._faces["B"][2, :] = self._faces["R"][2, :]
            self._faces["R"][2, :] = temp

    def _generate_face_recognition_question(self) -> dict[str, Any]:
        """Generate question about color at a specific position."""
        face = random.choice(list(self._faces.keys()))
        row = random.randint(0, 2)
        col = random.randint(0, 2)

        color_idx = self._faces[face][row, col]
        correct_color, _ = self.COLORS[color_idx]

        # Generate options
        all_colors = [name for name, _ in self.COLORS.values()]
        options = [correct_color]

        while len(options) < 6:
            color = random.choice(all_colors)
            if color not in options:
                options.append(color)

        random.shuffle(options)
        answer_index = options.index(correct_color) + 1
        options_text = "\n".join(
            [f"{i+1}. {opt.capitalize()}" for i, opt in enumerate(options)]
        )

        question = f"""{self.RUBIKS_RULES}

Question: What color is at position ({row}, {col}) on the {self.FACE_NAMES[face]}?

Options:
{options_text}"""

        return {"question": question, "answer": str(answer_index), "options": options}

    def _generate_color_count_question(self) -> dict[str, Any]:
        """Generate question about counting colors on a face."""
        face = random.choice(list(self._faces.keys()))

        # Count colors on this face
        color_counts = {}
        for i in range(3):
            for j in range(3):
                color_idx = self._faces[face][i, j]
                color_name, _ = self.COLORS[color_idx]
                color_counts[color_name] = color_counts.get(color_name, 0) + 1

        # Pick a color that exists on this face
        color = random.choice(list(color_counts.keys()))
        count = color_counts[color]

        # Generate options
        options = [count]
        possible_counts = list(range(max(0, count - 3), min(10, count + 4)))
        for c in possible_counts:
            if c != count and c not in options and len(options) < 8:
                options.append(c)

        random.shuffle(options)
        answer_index = options.index(count) + 1
        options_text = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)])

        question = f"""{self.RUBIKS_RULES}

Question: How many {color} squares are there on the {self.FACE_NAMES[face]}?

Options:
{options_text}"""

        return {"question": question, "answer": str(answer_index), "options": options}

    def _generate_move_prediction_question(self) -> dict[str, Any]:
        """Generate question about state after moves."""
        # Save current state
        original_faces = {k: v.copy() for k, v in self._faces.items()}

        # Pick random moves
        num_pred_moves = random.randint(1, 2)
        moves = ["F", "B", "L", "R", "U", "D", "F'", "B'", "L'", "R'", "U'", "D'"]
        selected_moves = [random.choice(moves) for _ in range(num_pred_moves)]

        # Apply moves
        for move in selected_moves:
            self._make_move(move)

        # Pick a random position to ask about
        face = random.choice(list(self._faces.keys()))
        row = random.randint(0, 2)
        col = random.randint(0, 2)

        color_idx = self._faces[face][row, col]
        correct_color, _ = self.COLORS[color_idx]

        # Restore original state for rendering
        self._faces = original_faces

        # Generate options
        all_colors = [name for name, _ in self.COLORS.values()]
        options = [correct_color]

        while len(options) < 6:
            color = random.choice(all_colors)
            if color not in options:
                options.append(color)

        random.shuffle(options)
        answer_index = options.index(correct_color) + 1
        options_text = "\n".join(
            [f"{i+1}. {opt.capitalize()}" for i, opt in enumerate(options)]
        )

        moves_str = " ".join(selected_moves)

        question = f"""{self.RUBIKS_RULES}

Question: After performing the moves "{moves_str}", what color will be at position ({row}, {col}) on the {self.FACE_NAMES[face]}?

Options:
{options_text}"""

        return {"question": question, "answer": str(answer_index), "options": options}

    def _check_answer(self, action: str) -> bool:
        """Check if answer is correct."""
        return action.strip().lower() == self._oracle_answer.strip().lower()

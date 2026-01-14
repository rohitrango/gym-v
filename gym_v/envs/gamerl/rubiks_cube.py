"""Rubik's Cube QA environment based on GameRL."""

from __future__ import annotations

from importlib import resources
import random
from textwrap import dedent
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation, get_logger

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
        For each face, coordinates are determined: column increases from left to right (0,1,2) and row increases from bottom to top (0,1,2).
        Moves: An uppercase letter indicates which face to rotate, with a prime symbol (') denoting counterclockwise rotation.
    """).strip()

    def __init__(
        self,
        num_moves: int | None = None,
        question_type: int | None = None,
        num_players: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if num_moves is None:
            num_moves = random.randint(1, 3)

        self._num_moves = num_moves
        self._question_type = question_type
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        # Initialize cube - each face is 3x3 array
        # U=yellow(0), D=white(1), L=orange(2), R=red(3), F=blue(4), B=green(5)
        self._faces = {
            "U": np.zeros((3, 3), dtype=int),
            "D": np.ones((3, 3), dtype=int),
            "L": np.full((3, 3), 2, dtype=int),
            "R": np.full((3, 3), 3, dtype=int),
            "F": np.full((3, 3), 4, dtype=int),
            "B": np.full((3, 3), 5, dtype=int),
        }
        self._current_question: dict[str, Any] = {}

    @property
    def description(self) -> str:
        base_desc = dedent(f"""
            This is a Rubik's Cube QA environment.

            {self.RUBIKS_RULES}

            Question Types:
            - Face Recognition: Identify the color at a specific position on a face
            - Color Count: Count how many squares of a specific color appear on a face
            - Move Prediction: Predict the cube state after performing moves

            The system will present you with a cube state and ask a specific question.
        """).strip()

        # Add question and answer format if question has been generated
        if hasattr(self, "_current_question") and self._current_question:
            desc = base_desc + "\n\n" + self._current_question["question"]
            desc += """

**Answer Format:**
Reply with only the answer (number or option number).

Examples:
- For multiple choice: 1, 2, 3, etc.
- For numbers: 42, 100, etc.

Do not include any explanation or extra text.
"""
            return desc.strip()

        return base_desc

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
        if self._question_type is None:
            question_type_idx = random.randint(0, len(self.QUESTION_TYPES) - 1)
        else:
            question_type_idx = self._question_type

        # Validate question type index
        if not (0 <= question_type_idx < len(self.QUESTION_TYPES)):
            raise ValueError(f"Invalid question type index: {question_type_idx}")

        # Get question type ID from index
        question_type = self.QUESTION_TYPES[question_type_idx]["id"]

        # Generate question
        if question_type == "face_recognition":
            self._current_question = self._generate_face_recognition_question()
        elif question_type == "color_count":
            self._current_question = self._generate_color_count_question()
        elif question_type == "move_prediction":
            self._current_question = self._generate_move_prediction_question()
        else:
            raise ValueError(f"Unknown question type: {question_type}")

        logger.info(
            f"Reset Rubik's Cube QA (num_moves={self._num_moves}, question: {question_type})."
        )

        # Generate text state
        text_state = self._get_state_text()

        obs = Observation(
            image=self.render(),
            text=text_state,
            metadata={
                "question": self._current_question["question"],
                "options": self._current_question.get("options"),
            },
        )

        info = {
            "oracle_answer": self._current_question["answer"],
            "question_type": question_type,
        }

        return {agent_id: obs for agent_id in self._agent_ids}, {
            agent_id: info for agent_id in self._agent_ids
        }

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
        reward = 0.0
        terminated = True
        truncated = False

        # Check answer
        correct = self._check_answer(action_str.strip())

        if correct:
            reward = 1.0
            response = "Correct!"
        else:
            reward = 0.0
            response = (
                f"Incorrect. The correct answer is: {self._current_question['answer']}"
            )

        info = {
            "correct": correct,
            "user_answer": action_str.strip(),
            "oracle_answer": self._current_question["answer"],
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
        """Render the Rubik's Cube in unfolded view."""
        cell_size = 40
        margin = 20

        # Layout: cross pattern
        #       U U U
        #       U U U
        #       U U U
        # L L L F F F R R R B B B
        # L L L F F F R R R B B B
        # L L L F F F R R R B B B
        #       D D D
        #       D D D
        #       D D D

        img_width = margin * 2 + 12 * cell_size
        img_height = margin * 2 + 9 * cell_size

        img = Image.new("RGB", (img_width, img_height), (240, 240, 240))
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype(str(self.assets_dir / "DejaVuSans.ttf"), 12)
            label_font = ImageFont.truetype(str(self.assets_dir / "DejaVuSans.ttf"), 16)
        except Exception:
            font = ImageFont.load_default()
            label_font = ImageFont.load_default()

        # Face positions in the unfolded layout (row_offset, col_offset)
        face_positions = {
            "U": (0, 3),
            "L": (3, 0),
            "F": (3, 3),
            "R": (3, 6),
            "B": (3, 9),
            "D": (6, 3),
        }

        # Draw each face
        for face_name, (row_offset, col_offset) in face_positions.items():
            face_data = self._faces[face_name]

            # Draw face label
            label_x = margin + col_offset * cell_size + cell_size * 1.5
            label_y = margin + row_offset * cell_size - 15
            draw.text(
                (label_x, label_y),
                face_name,
                fill=(0, 0, 0),
                font=label_font,
                anchor="mm",
            )

            # Draw face cells
            for i in range(3):
                for j in range(3):
                    color_idx = face_data[i, j]
                    _, rgb_color = self.COLORS[color_idx]

                    x = margin + (col_offset + j) * cell_size
                    y = margin + (row_offset + i) * cell_size

                    # Draw cell
                    draw.rectangle(
                        [x, y, x + cell_size - 2, y + cell_size - 2],
                        fill=rgb_color,
                        outline=(0, 0, 0),
                        width=2,
                    )

                    # Draw coordinates (row, col) - using bottom-to-top convention
                    coord_row = 2 - i  # Flip: 0 at bottom, 2 at top
                    coord_text = f"({coord_row},{j})"
                    draw.text(
                        (x + cell_size // 2, y + cell_size // 2),
                        coord_text,
                        fill=(0, 0, 0)
                        if rgb_color != (255, 255, 255)
                        else (100, 100, 100),
                        font=font,
                        anchor="mm",
                    )

        # Draw color legend
        legend_y = img_height - margin - 60
        legend_x = margin
        draw.text(
            (legend_x, legend_y - 20), "Color Legend:", fill=(0, 0, 0), font=label_font
        )

        for idx, (color_name, rgb) in enumerate(self.COLORS.values()):
            x = legend_x + (idx % 3) * 130
            y = legend_y + (idx // 3) * 30

            # Color swatch
            draw.rectangle([x, y, x + 20, y + 20], fill=rgb, outline=(0, 0, 0), width=1)
            draw.text(
                (x + 25, y + 10),
                color_name.capitalize(),
                fill=(0, 0, 0),
                font=font,
                anchor="lm",
            )

        return img

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
        correct_answer = self._current_question["answer"]
        return action.strip().lower() == correct_answer.strip().lower()

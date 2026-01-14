"""Hue color gradient puzzle QA game based on GameRL."""

from __future__ import annotations

import colorsys
from importlib import resources
import random
from textwrap import dedent
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation, get_logger

logger = get_logger()


class GameRLHueQAEnv(Env):
    """Hue color gradient puzzle QA environment.

    A puzzle where colors change gradually along rows or columns following gradient patterns.
    Players need to identify colors, describe gradient patterns, or match missing colors.

    Args:
        board_size: Size of the grid (5-8, default random)
        num_lines: Number of gradient lines to generate (default random 3-4)
        cell_size: Size of each cell in pixels for rendering (default 60)
        question_type: Type of question to ask
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    # Question types
    QUESTION_TYPES = [
        {
            "id": "color_description",
            "name": "Color Description",
            "level": "Easy",
            "answer_format": "multiple_choice",
            "qa_type": "Target Perception",
        },
        {
            "id": "gradient_pattern",
            "name": "Gradient Pattern",
            "level": "Medium",
            "answer_format": "multiple_choice",
            "qa_type": "Target Perception",
        },
        {
            "id": "color_matching",
            "name": "Color Matching",
            "level": "Hard",
            "answer_format": "multiple_choice",
            "qa_type": "State Prediction",
        },
    ]

    GAME_RULES = dedent("""
        Rules:
        1. Colors change gradually along rows or columns.
        2. A gradient transitions between two colors.
        3. Each row or column can have its own independent gradient pattern.
        4. Row and column indexes begin from 1 at the top-left corner.
    """).strip()

    def __init__(
        self,
        board_size: int | None = None,
        num_lines: int | None = None,
        cell_size: int = 60,
        question_type: str | int | None = None,
        num_players: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._board_size = (
            board_size if board_size is not None else random.randint(5, 8)
        )
        self._num_lines = num_lines if num_lines is not None else random.randint(3, 4)
        self._cell_size = cell_size
        self._question_type = question_type
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        # Game state (initialized in reset)
        self._board: np.ndarray = np.zeros(
            (self._board_size, self._board_size, 3), dtype=np.uint8
        )
        self._colored_cells: set[tuple[int, int]] = set()
        self._cells_to_fill: dict[tuple[int, int], str] = {}
        self._gradient_info: list[dict[str, Any]] = []
        self._removed_positions: list[tuple[int, int]] = []
        self._shuffled_colors: list[tuple[int, int, int]] = []
        self._color_mapping: dict[int, int] = {}
        self._current_question: dict[str, Any] = {}

    @property
    def description(self) -> str:
        base_desc = dedent(f"""
            This is a Hue color gradient puzzle QA environment.

            {self.GAME_RULES}

            Question Types:
            - Color Description: Identify the color of a specific cell
            - Gradient Pattern: Describe the gradient pattern in a row or column
            - Color Matching: Determine which color fits a specific empty cell

            The system will present you with a puzzle state and ask a specific question.
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
        """Generate text description of current puzzle state.

        Returns a text representation of the color gradient board state.
        """
        text = "Hue Color Gradient Puzzle\n"
        text += f"Board Size: {self._board_size}x{self._board_size}\n\n"

        # List colored cells
        if self._colored_cells:
            text += "Colored Cells:\n"
            for row in range(self._board_size):
                for col in range(self._board_size):
                    if (row, col) in self._colored_cells:
                        color_rgb = tuple(self._board[row, col])
                        color_name = self._get_color_name(color_rgb)
                        text += f"  Row {row + 1}, Col {col + 1}: {color_name}\n"

        # List cells to fill (if any)
        if self._cells_to_fill:
            text += "\nCells to Fill:\n"
            for (row, col), label in self._cells_to_fill.items():
                text += f"  Cell {label} at Row {row + 1}, Col {col + 1}\n"

        return text.strip()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        super().reset(seed=seed)

        # Generate puzzle
        self._generate_puzzle()

        # Select question type
        if self._question_type is None:
            question_type = random.choice(self.QUESTION_TYPES)["id"]
        elif isinstance(self._question_type, int):
            # Support integer indexing (0, 1, 2, ...)
            question_type = self.QUESTION_TYPES[self._question_type]["id"]
        else:
            question_type = self._question_type

        # Generate question based on type
        if question_type == "color_description":
            self._current_question = self._generate_color_description_question()
        elif question_type == "gradient_pattern":
            self._current_question = self._generate_gradient_pattern_question()
        elif question_type == "color_matching":
            self._current_question = self._generate_color_matching_question()
        else:
            raise ValueError(f"Unknown question type: {question_type}")

        logger.info(
            f"Reset Hue QA ({self._board_size}x{self._board_size}, "
            f"question: {question_type})."
        )

        # Generate text state representation
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

        # Parse answer
        action_str = action_str.strip()

        # Check if answer is correct
        correct = self._check_answer(action_str)

        if correct:
            reward = 1.0
            response = "Correct!"
        else:
            reward = 0.0
            response = (
                f"Incorrect. The correct answer is: {self._current_question['answer']}"
            )

        if "explanation" in self._current_question:
            response += f"\n\nExplanation:\n{self._current_question['explanation']}"

        info = {
            "correct": correct,
            "user_answer": action_str,
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
        """Render the current puzzle state as a PIL Image."""
        index_margin = 30
        options_height = 100 if self._shuffled_colors else 0
        padding = 20

        board_size = self._board_size * self._cell_size
        img_width = board_size + index_margin
        img_height = board_size + index_margin + options_height + padding

        img = Image.new("RGB", (img_width, img_height), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Load font
        try:
            font = ImageFont.truetype(str(self.assets_dir / "DejaVuSans.ttf"), 14)
        except Exception:
            font = ImageFont.load_default()

        # Draw row indices
        for i in range(self._board_size):
            text = str(i + 1)
            draw.text(
                (10, index_margin + i * self._cell_size + self._cell_size // 2 - 5),
                text,
                fill=(0, 0, 0),
                font=font,
            )

        # Draw column indices
        for j in range(self._board_size):
            text = str(j + 1)
            draw.text(
                (index_margin + j * self._cell_size + self._cell_size // 3, 10),
                text,
                fill=(0, 0, 0),
                font=font,
            )

        # Draw the board
        for i in range(self._board_size):
            for j in range(self._board_size):
                pos = (i, j)
                x = index_margin + j * self._cell_size
                y = index_margin + i * self._cell_size

                # Determine cell appearance
                if pos in self._cells_to_fill:
                    # Empty cell with label
                    draw.rectangle(
                        [x, y, x + self._cell_size, y + self._cell_size],
                        fill=(255, 255, 255),
                        outline=(0, 0, 0),
                        width=1,
                    )
                    label = self._cells_to_fill[pos]
                    draw.text(
                        (x + self._cell_size // 2, y + self._cell_size // 2),
                        label,
                        fill=(0, 0, 0),
                        font=font,
                        anchor="mm",
                    )
                elif pos in self._colored_cells:
                    # Colored cell
                    color = tuple(self._board[i, j])
                    margin = 2
                    draw.rectangle(
                        [x, y, x + self._cell_size, y + self._cell_size],
                        fill=(255, 255, 255),
                        outline=(0, 0, 0),
                        width=1,
                    )
                    draw.rectangle(
                        [
                            x + margin,
                            y + margin,
                            x + self._cell_size - margin,
                            y + self._cell_size - margin,
                        ],
                        fill=color,
                        outline=None,
                    )
                else:
                    # Empty cell with diagonal lines
                    draw.rectangle(
                        [x, y, x + self._cell_size, y + self._cell_size],
                        fill=(245, 245, 245),
                        outline=(0, 0, 0),
                        width=1,
                    )
                    # Draw diagonal lines
                    for offset in range(-self._cell_size, self._cell_size * 2, 12):
                        draw.line(
                            [
                                (x + offset, y),
                                (x + offset + self._cell_size, y + self._cell_size),
                            ],
                            fill=(200, 200, 200),
                            width=2,
                        )

        # Draw color options if present
        if self._shuffled_colors:
            option_width = img_width // len(self._shuffled_colors)
            y_start = board_size + index_margin + padding + 20

            for idx, color in enumerate(self._shuffled_colors):
                x_start = idx * option_width + 5
                # Draw option number
                draw.text(
                    (x_start, y_start - 15), f"{idx + 1}", fill=(0, 0, 0), font=font
                )
                # Draw color swatch
                draw.rectangle(
                    [x_start, y_start, x_start + option_width - 10, y_start + 40],
                    fill=tuple(color),
                    outline=(0, 0, 0),
                    width=1,
                )

        return img

    def _generate_puzzle(self):
        """Generate a color gradient puzzle."""
        self._board = np.zeros((self._board_size, self._board_size, 3), dtype=np.uint8)
        self._colored_cells = set()
        self._gradient_info = []

        # Add gradient lines
        for _ in range(self._num_lines):
            self._add_gradient_line()

    def _add_gradient_line(self):
        """Add a row or column with color gradient."""
        for _ in range(10):  # Try multiple times
            is_row = random.choice([True, False])
            idx = random.randint(0, self._board_size - 1)

            # Check if adjacent parallel gradient exists
            has_adjacent = False
            for gradient in self._gradient_info:
                if gradient["type"] == ("row" if is_row else "column"):
                    if abs(gradient["index"] - idx) == 1:
                        has_adjacent = True
                        break

            if has_adjacent:
                continue

            # Generate gradient
            start_idx = random.randint(0, 1)
            end_idx = random.randint(self._board_size - 2, self._board_size - 1)

            color1 = np.array(
                [random.randint(0, 255) for _ in range(3)], dtype=np.uint8
            )
            color2 = np.array(
                [random.randint(0, 255) for _ in range(3)], dtype=np.uint8
            )

            for i in range(start_idx, end_idx + 1):
                pos = (idx, i) if is_row else (i, idx)
                ratio = (
                    (i - start_idx) / (end_idx - start_idx)
                    if end_idx > start_idx
                    else 0
                )
                self._board[pos[0], pos[1]] = self._mix_colors(color1, color2, ratio)
                self._colored_cells.add(pos)

            self._gradient_info.append(
                {
                    "type": "row" if is_row else "column",
                    "index": idx,
                    "start_color": color1,
                    "end_color": color2,
                }
            )
            return

    def _mix_colors(
        self, color1: np.ndarray, color2: np.ndarray, ratio: float
    ) -> np.ndarray:
        """Linear interpolation between two colors."""
        mix = np.clip((1 - ratio) * color1 + ratio * color2, 0, 255)
        return mix.astype(np.uint8)

    def _generate_color_description_question(self) -> dict[str, Any]:
        """Generate a question about a specific cell's color."""
        valid_cells = [
            pos for pos in self._colored_cells if pos not in self._cells_to_fill
        ]
        if not valid_cells:
            return {"question": "No valid cells", "answer": "1", "options": ["N/A"]}

        target_pos = random.choice(valid_cells)
        target_color = tuple(self._board[target_pos])
        correct_description = self._get_color_name(target_color)

        # Generate wrong options
        wrong_descriptions = set()
        while len(wrong_descriptions) < 7:
            wrong_color = self._generate_distant_color([])
            wrong_desc = self._get_color_name(tuple(wrong_color))
            if wrong_desc != correct_description:
                wrong_descriptions.add(wrong_desc)

        options = list(wrong_descriptions) + [correct_description]
        random.shuffle(options)

        question = f"""{self.GAME_RULES}

Question:
What color is the cell at row {target_pos[0] + 1}, column {target_pos[1] + 1}?

Options:
""" + "\n".join(f"{i+1}: {opt}" for i, opt in enumerate(options))

        return {
            "question": question,
            "answer": str(options.index(correct_description) + 1),
            "options": options,
        }

    def _generate_gradient_pattern_question(self) -> dict[str, Any]:
        """Generate a question about a gradient pattern."""
        if not self._gradient_info:
            return {"question": "No gradients", "answer": "1", "options": ["N/A"]}

        gradient = random.choice(self._gradient_info)
        correct_desc = self._describe_gradient(
            gradient["start_color"], gradient["end_color"]
        )

        # Generate wrong options
        wrong_descriptions = set()
        while len(wrong_descriptions) < 7:
            color1 = self._generate_distant_color([])
            color2 = self._generate_distant_color([color1])
            wrong_desc = self._describe_gradient(color1, color2)
            if wrong_desc != correct_desc:
                wrong_descriptions.add(wrong_desc)

        options = list(wrong_descriptions) + [correct_desc]
        random.shuffle(options)

        line_type = "row" if gradient["type"] == "row" else "column"

        question = f"""{self.GAME_RULES}

Question:
What is the gradient pattern in {line_type} {gradient["index"] + 1}?

Options:
""" + "\n".join(f"{i+1}: {opt}" for i, opt in enumerate(options))

        return {
            "question": question,
            "answer": str(options.index(correct_desc) + 1),
            "options": options,
        }

    def _generate_color_matching_question(self) -> dict[str, Any]:
        """Generate a color matching question."""
        # Remove some colors
        num_removed = random.randint(3, 6)
        if len(self._colored_cells) < num_removed:
            num_removed = len(self._colored_cells)

        self._removed_positions = random.sample(list(self._colored_cells), num_removed)
        removed_colors = []
        labels = list("ABCDEFGH")[:num_removed]

        for pos, label in zip(self._removed_positions, labels, strict=False):
            color = tuple(self._board[pos[0], pos[1]])
            removed_colors.append(color)
            self._board[pos[0], pos[1]] = (255, 255, 255)
            self._cells_to_fill[pos] = label

        # Generate extra distractor colors
        num_extra = 6 - num_removed
        extra_options = []
        existing_colors = removed_colors.copy()
        for _ in range(num_extra):
            new_color = self._generate_distant_color(existing_colors)
            extra_options.append(tuple(new_color))
            existing_colors.append(tuple(new_color))

        # Shuffle all colors
        all_colors = removed_colors + extra_options
        shuffled_indices = list(range(len(all_colors)))
        random.shuffle(shuffled_indices)

        self._color_mapping = {i: shuffled_indices[i] for i in range(len(all_colors))}
        inverse_mapping = {v: k for k, v in self._color_mapping.items()}
        self._shuffled_colors = [
            all_colors[inverse_mapping[i]] for i in range(len(all_colors))
        ]

        # Select a target cell
        target_pos = random.choice(self._removed_positions)
        target_label = self._cells_to_fill[target_pos]

        original_idx = self._removed_positions.index(target_pos)
        shuffled_answer = self._color_mapping[original_idx] + 1

        question = f"""{self.GAME_RULES}

Question:
Which color should be put in cell {target_label}?

Options:
Colors are numbered from 1 to 6 in the palette below"""

        return {
            "question": question,
            "answer": str(shuffled_answer),
            "options": list(range(1, len(self._shuffled_colors) + 1)),
        }

    def _get_color_name(self, rgb: tuple[int, int, int]) -> str:
        """Convert RGB color to human-readable name."""
        r, g, b = rgb
        h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
        h = h * 360

        if v < 0.2:
            return "black"
        if v > 0.9 and s < 0.1:
            return "white"
        if s < 0.15:
            return f"{'dark' if v < 0.5 else 'light'} gray"

        color_ranges = {
            (330, 30): "red",
            (30, 60): "orange",
            (60, 90): "yellow",
            (90, 150): "green",
            (150, 200): "cyan",
            (200, 240): "blue",
            (240, 270): "indigo",
            (270, 330): "purple",
        }

        base_color = None
        for (start, end), name in color_ranges.items():
            if start <= h <= end or (start > end and (h >= start or h <= end)):
                base_color = name
                break

        if not base_color:
            return "gray"

        modifiers = []
        if s < 0.4:
            modifiers.append("pale")
        elif s > 0.8:
            modifiers.append("vivid")

        if v < 0.4:
            modifiers.append("dark")
        elif v > 0.8:
            modifiers.append("bright")

        return f"{' '.join(modifiers)} {base_color}" if modifiers else base_color

    def _describe_gradient(self, start_color: np.ndarray, end_color: np.ndarray) -> str:
        """Create a human-readable description of a color gradient."""
        start_name = self._get_color_name(tuple(start_color))
        end_name = self._get_color_name(tuple(end_color))

        if start_name == end_name:
            start_hsv = colorsys.rgb_to_hsv(
                start_color[0] / 255, start_color[1] / 255, start_color[2] / 255
            )
            end_hsv = colorsys.rgb_to_hsv(
                end_color[0] / 255, end_color[1] / 255, end_color[2] / 255
            )
            return f"transitioning from {'darker' if end_hsv[2] > start_hsv[2] else 'lighter'} to {'lighter' if end_hsv[2] > start_hsv[2] else 'darker'} {start_name}"
        else:
            return f"transitioning from {start_name} to {end_name}"

    def _generate_distant_color(self, existing_colors: list) -> np.ndarray:
        """Generate a color that is maximally distant from existing colors."""

        def color_distance(c1, c2):
            hsv1 = colorsys.rgb_to_hsv(c1[0] / 255, c1[1] / 255, c1[2] / 255)
            hsv2 = colorsys.rgb_to_hsv(c2[0] / 255, c2[1] / 255, c2[2] / 255)
            h_diff = min(abs(hsv1[0] - hsv2[0]), 1 - abs(hsv1[0] - hsv2[0]))
            s_diff = abs(hsv1[1] - hsv2[1])
            v_diff = abs(hsv1[2] - hsv2[2])
            return h_diff + s_diff + v_diff

        max_min_distance = 0
        best_color = None

        for _ in range(100):
            color = np.array([random.randint(0, 255) for _ in range(3)])
            if not existing_colors:
                return color

            min_distance = min(
                color_distance(color, np.array(ec)) for ec in existing_colors
            )

            if min_distance > max_min_distance:
                max_min_distance = min_distance
                best_color = color

        return best_color if best_color is not None else np.array([128, 128, 128])

    def _check_answer(self, action: str) -> bool:
        """Check if the provided answer is correct."""
        correct_answer = self._current_question["answer"]
        return action.strip().lower() == correct_answer.strip().lower()

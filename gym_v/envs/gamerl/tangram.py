"""Tangram QA environment based on GameRL."""

from __future__ import annotations

import colorsys
from importlib import resources
import random
from textwrap import dedent
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation, get_logger
from gym_v.envs.gamerl.utils import build_description, score_exact

logger = get_logger()


class GameRLTangramQAEnv(Env):
    """Tangram QA environment.

    A Voronoi-based puzzle where pieces are formed by assigning cells to nearest seeds.
    Some pieces are removed and shown separately below the main board.

    Question Types:
    - Piece Count: Count how many pieces are on the board
    - Piece Area: Calculate the area of a specific piece
    - Piece Adjacency: Count adjacent pieces to a target piece
    - Rotation: Check if removed piece can fit back after rotation
    - Placement: Where to place a removed piece

    Args:
        grid_size: Size of the grid
        num_seeds: Number of seeds for Voronoi regions
        num_pieces_to_remove: Number of pieces to remove from board
        question_type: Type of question to ask
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    QUESTION_TYPES = [
        {
            "id": "piece_count",
            "name": "Piece Count",
            "level": "Easy",
            "answer_format": "multiple_choice",
            "qa_type": "Target Perception",
        },
        {
            "id": "piece_area",
            "name": "Piece Area",
            "level": "Medium",
            "answer_format": "multiple_choice",
            "qa_type": "Target Perception",
        },
        {
            "id": "adjacency",
            "name": "Piece Adjacency",
            "level": "Medium",
            "answer_format": "multiple_choice",
            "qa_type": "Target Perception",
        },
        {
            "id": "rotation",
            "name": "Piece Rotation",
            "level": "Medium",
            "answer_format": "multiple_choice",
            "qa_type": "State Prediction",
        },
        {
            "id": "placement",
            "name": "Piece Placement",
            "level": "Medium",
            "answer_format": "multiple_choice",
            "qa_type": "ActionOutcome",
        },
    ]

    # Fixed color mapping (RGB 0-255)
    COLOR_MAPPING = {
        0: (255, 255, 255),  # white for empty
        1: (31, 119, 180),  # blue
        2: (255, 127, 14),  # orange
        3: (44, 160, 44),  # green
        4: (214, 39, 40),  # red
        5: (148, 103, 189),  # purple
        6: (140, 86, 75),  # brown
        7: (227, 119, 194),  # pink
        8: (127, 127, 127),  # gray
        9: (188, 189, 34),  # yellow
        10: (23, 190, 207),  # cyan
        11: (125, 182, 234),  # light blue
        12: (255, 193, 121),  # light orange
        13: (132, 209, 132),  # light green
        14: (238, 132, 133),  # light red
        15: (194, 173, 225),  # light purple
        16: (186, 162, 156),  # light brown
        17: (244, 195, 229),  # light pink
        18: (181, 181, 181),  # light gray
        19: (227, 228, 117),  # light yellow
        20: (115, 220, 233),  # light cyan
    }

    GAME_RULES = dedent("""
        Rules:
        1. Each numbered region represents a piece on the board.
        2. Pieces are considered adjacent if they share at least one edge.
        3. Pieces that only touch at corners are not considered adjacent.
        4. Some pieces have been removed and are shown below the main board.
    """).strip()

    def __init__(
        self,
        grid_size: int | None = None,
        num_seeds: int | None = None,
        num_pieces_to_remove: int | None = None,
        question_type: int | None = None,
        num_players: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Random configuration if not specified
        if grid_size is None:
            grid_size = random.randint(5, 10)
        if num_seeds is None:
            num_seeds = random.randint(4, 8)
        if num_pieces_to_remove is None:
            num_pieces_to_remove = random.randint(1, 3)

        self._grid_size = grid_size
        self._num_seeds = num_seeds
        self._num_pieces_to_remove = num_pieces_to_remove
        self._question_type_param = question_type
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        # Game state
        self._grid: np.ndarray = np.zeros((grid_size, grid_size), dtype=int)
        self._seeds: list[tuple[int, int]] = []
        self._pieces: list[dict[str, Any]] = []
        self._removed_pieces: list[int] = []

        # Standard QA variables
        self._question_type_idx: int = 0
        self._question: str = ""
        self._options: list[str] | None = None
        self._oracle_answer: str = ""

    @property
    def description(self) -> str:
        """Return game rules + current question + answer format."""
        return build_description(
            game_name="Tangram Puzzle",
            rules=self.GAME_RULES,
            question=self._question,
            options=self._options,
            oracle_answer=self._oracle_answer,
        )

    def _get_state_text(self) -> str:
        """Generate text description of current Tangram state."""
        text = "Tangram Puzzle State\n"
        text += f"Grid Size: {self._grid_size}x{self._grid_size}\n\n"

        text += "Main Board:\n"
        text += f"  Pieces on board: {len(self._pieces)}\n"
        for piece_id, cells in enumerate(self._pieces, 1):
            text += f"  Piece {piece_id}: {len(cells)} cells\n"

        if self._removed_pieces:
            text += f"\nRemoved Pieces: {len(self._removed_pieces)}\n"
            for _idx, piece_id in enumerate(self._removed_pieces, 1):
                piece = self._pieces[piece_id - 1]
                text += f"  Removed Piece {piece_id}: {len(piece['cells'])} cells\n"

        return text.strip()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        super().reset(seed=seed)

        # Generate puzzle
        self._generate_puzzle()

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
        if q_type["id"] == "piece_count":
            result = self._generate_piece_count_question()
        elif q_type["id"] == "piece_area":
            result = self._generate_piece_area_question()
        elif q_type["id"] == "adjacency":
            result = self._generate_adjacency_question()
        elif q_type["id"] == "rotation":
            result = self._generate_rotation_question()
        elif q_type["id"] == "placement":
            result = self._generate_placement_question()
        else:
            raise ValueError(f"Unknown question type: {q_type['id']}")

        # Extract to instance variables
        self._question = result["question"]
        self._options = result.get("options")
        self._oracle_answer = result["answer"]

        logger.info(
            f"Reset Tangram QA ({self._grid_size}x{self._grid_size}, question: {q_type['id']})."
        )

        # Generate text state
        text_state = self._get_state_text()

        obs = Observation(
            image=self.render(),
            text=text_state,
            metadata={
                "text_state": text_state,
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
        """Render the Tangram puzzle."""
        cell_size = 50
        margin = 10

        # Main board size
        board_width = self._grid_size * cell_size
        board_height = self._grid_size * cell_size

        # Calculate removed pieces area size
        removed_height = 0
        removed_width = 0
        if self._removed_pieces:
            for piece_id in self._removed_pieces:
                piece = self._pieces[piece_id - 1]
                removed_width += piece["width"] * cell_size + margin
                removed_height = max(removed_height, piece["height"] * cell_size)
            removed_width -= margin  # Remove last margin

        # Total image size
        total_width = max(board_width, removed_width) + 2 * margin
        total_height = (
            board_height + removed_height + 4 * margin + 60
        )  # Extra for both titles (Main Board + Removed Pieces)

        img = Image.new("RGB", (total_width, total_height), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype(str(self.assets_dir / "DejaVuSans.ttf"), 14)
            title_font = ImageFont.truetype(str(self.assets_dir / "DejaVuSans.ttf"), 16)
        except Exception:
            font = ImageFont.load_default()
            title_font = ImageFont.load_default()

        # Draw main board title
        draw.text((margin, margin), "Main Board", fill=(0, 0, 0), font=title_font)

        # Draw main board
        y_offset = margin + 30
        for i in range(self._grid_size):
            for j in range(self._grid_size):
                piece_id = self._grid[i, j]
                color = self.COLOR_MAPPING.get(piece_id, (255, 255, 255))

                x = margin + j * cell_size
                y = y_offset + i * cell_size

                # Draw cell
                draw.rectangle(
                    [x, y, x + cell_size - 1, y + cell_size - 1],
                    fill=color,
                    outline=(0, 0, 0),
                    width=1,
                )

                # Draw number or shading for empty cells
                if piece_id != 0:
                    text = str(piece_id)
                    bbox = draw.textbbox((0, 0), text, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    draw.text(
                        (
                            x + cell_size // 2 - text_width // 2,
                            y + cell_size // 2 - text_height // 2,
                        ),
                        text,
                        fill=(0, 0, 0),
                        font=font,
                    )
                else:
                    # Draw shading for empty cells
                    self._draw_shaded_cell(draw, x, y, cell_size)

        # Draw removed pieces
        if self._removed_pieces:
            y_offset = board_height + 2 * margin + 40
            draw.text(
                (margin, y_offset), "Removed Pieces", fill=(0, 0, 0), font=title_font
            )
            y_offset += 30

            x_offset = margin
            for piece_id in self._removed_pieces:
                piece = self._pieces[piece_id - 1]
                piece_cells = piece["cells"]

                # Draw piece
                for i in range(piece["height"]):
                    for j in range(piece["width"]):
                        cell_value = piece_cells[i, j]
                        if cell_value > 0:
                            color = self.COLOR_MAPPING.get(cell_value, (255, 255, 255))
                        else:
                            color = (240, 240, 240)  # Light gray for empty

                        x = x_offset + j * cell_size
                        y = y_offset + i * cell_size

                        draw.rectangle(
                            [x, y, x + cell_size - 1, y + cell_size - 1],
                            fill=color,
                            outline=(0, 0, 0),
                            width=1,
                        )

                        if cell_value > 0:
                            text = str(piece_id)
                            bbox = draw.textbbox((0, 0), text, font=font)
                            text_width = bbox[2] - bbox[0]
                            text_height = bbox[3] - bbox[1]
                            draw.text(
                                (
                                    x + cell_size // 2 - text_width // 2,
                                    y + cell_size // 2 - text_height // 2,
                                ),
                                text,
                                fill=(0, 0, 0),
                                font=font,
                            )
                        else:
                            self._draw_shaded_cell(draw, x, y, cell_size)

                x_offset += piece["width"] * cell_size + margin

        return img

    def _draw_shaded_cell(self, draw: ImageDraw.ImageDraw, x: int, y: int, size: int):
        """Draw diagonal lines in empty cells."""
        line_num = 6
        for d in range(-line_num, line_num):
            d_norm = d / line_num
            start_x = x + max(0, d_norm * size)
            start_y = y + max(0, -d_norm * size)
            end_x = x + min(size, (d_norm + 1) * size)
            end_y = y + min(size, (-d_norm + 1) * size)
            draw.line([(start_x, start_y), (end_x, end_y)], fill=(0, 0, 0), width=1)

    def _generate_puzzle(self):
        """Generate a Tangram puzzle using Voronoi regions."""
        # Generate random seed points
        self._seeds = []
        points = []
        while len(points) < self._num_seeds:
            x = random.randint(0, self._grid_size - 1)
            y = random.randint(0, self._grid_size - 1)
            if (x, y) not in points:
                points.append((x, y))
        self._seeds = points

        # Assign each cell to nearest seed
        for i in range(self._grid_size):
            for j in range(self._grid_size):
                min_dist = float("inf")
                closest_seed = 0
                for idx, (seed_x, seed_y) in enumerate(self._seeds):
                    dist = (i - seed_x) ** 2 + (j - seed_y) ** 2
                    if dist < min_dist:
                        min_dist = dist
                        closest_seed = idx + 1
                self._grid[i, j] = closest_seed

        # Calculate pieces
        self._pieces = []
        for piece_id in range(1, self._num_seeds + 1):
            cells = np.argwhere(self._grid == piece_id)
            if len(cells) == 0:
                continue

            min_i = int(np.min(cells[:, 0]))
            min_j = int(np.min(cells[:, 1]))
            max_i = int(np.max(cells[:, 0]))
            max_j = int(np.max(cells[:, 1]))

            height = max_i - min_i + 1
            width = max_j - min_j + 1

            # Extract piece cells
            piece_cells = np.zeros((height, width), dtype=int)
            for cell in cells:
                piece_cells[cell[0] - min_i, cell[1] - min_j] = piece_id

            self._pieces.append(
                {
                    "id": piece_id,
                    "width": width,
                    "height": height,
                    "cells": piece_cells,
                }
            )

        # Remove random pieces
        available_pieces = list(range(1, self._num_seeds + 1))
        self._removed_pieces = random.sample(
            available_pieces, self._num_pieces_to_remove
        )
        for piece_id in self._removed_pieces:
            self._grid[self._grid == piece_id] = 0

    def _get_color_name(self, rgb: tuple[int, int, int]) -> str:
        """Convert RGB to color name."""
        r, g, b = rgb
        r, g, b = r / 255, g / 255, b / 255
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        h = h * 360

        if v < 0.2:
            return "black"
        if v > 0.9 and s < 0.1:
            return "white"
        if s < 0.15:
            return "gray" if v >= 0.5 else "dark gray"

        color_ranges = {
            (330, 360): "red",
            (0, 20): "red",
            (20, 45): "orange",
            (45, 80): "yellow",
            (80, 150): "green",
            (150, 200): "cyan",
            (200, 240): "blue",
            (240, 270): "indigo",
            (270, 330): "purple",
        }

        for (start, end), name in color_ranges.items():
            if start <= h <= end:
                return name

        return "gray"

    def _get_piece_colors(self) -> dict[int, str]:
        """Get color names for all pieces."""
        colors = {}
        for i in range(1, self._num_seeds + 1):
            rgb = self.COLOR_MAPPING.get(i, (255, 255, 255))
            colors[i] = self._get_color_name(rgb)
        return colors

    def _get_plot_level(self) -> str:
        """Determine difficulty level based on grid size."""
        if self._grid_size <= 5:
            return "Easy"
        elif self._grid_size <= 8:
            return "Medium"
        return "Hard"

    def _generate_piece_count_question(self) -> dict[str, Any]:
        """Generate piece counting question."""
        unique_pieces = len(np.unique(self._grid[self._grid != 0]))

        # Generate options
        options = list(range(max(0, unique_pieces - 4), unique_pieces + 5))
        while len(options) < 8:
            options.append(options[-1] + 1)
        random.shuffle(options)

        correct_answer = options.index(unique_pieces) + 1
        options_text = "\n".join([f"{i+1}: {opt}" for i, opt in enumerate(options)])

        question = f"""{self.GAME_RULES}

Question:
How many pieces are currently on the main board?

Options:
{options_text}"""

        return {"question": question, "answer": str(correct_answer), "options": options}

    def _generate_piece_area_question(self) -> dict[str, Any]:
        """Generate piece area question."""
        active_pieces = sorted(list(set(np.unique(self._grid)) - {0}))
        if not active_pieces:
            return self._generate_piece_count_question()

        target_piece = random.choice(active_pieces)
        target_area = int(np.sum(self._grid == target_piece))

        # Generate options
        possible_areas = list(range(max(1, target_area - 6), target_area + 6))
        options = random.sample(possible_areas, min(8, len(possible_areas)))
        if target_area not in options:
            options[0] = target_area
        options = sorted(options)

        correct_answer = options.index(target_area) + 1
        options_text = "\n".join([f"{i+1}: {opt}" for i, opt in enumerate(options)])

        question = f"""{self.GAME_RULES}

Question:
What is the area (number of cells) of Piece {target_piece}?

Options:
{options_text}"""

        return {"question": question, "answer": str(correct_answer), "options": options}

    def _generate_adjacency_question(self) -> dict[str, Any]:
        """Generate piece adjacency question."""
        active_pieces = sorted(list(set(np.unique(self._grid)) - {0}))

        if len(active_pieces) < 2:
            return self._generate_piece_count_question()

        target_piece = random.choice(active_pieces)

        # Find adjacent pieces
        adjacent_pieces = self._get_adjacent_pieces(target_piece)
        correct_count = len(adjacent_pieces)

        # Generate options
        options = list(
            range(max(0, correct_count - 4), min(len(active_pieces), correct_count + 3))
        )
        while len(options) < 8:
            next_val = options[-1] + 1
            if next_val <= len(active_pieces):
                options.append(next_val)
            else:
                break
        random.shuffle(options)

        correct_answer = options.index(correct_count) + 1
        options_text = "\n".join([f"{i+1}: {opt}" for i, opt in enumerate(options)])

        question = f"""{self.GAME_RULES}

Question:
How many different pieces are adjacent to Piece {target_piece}?

Options:
{options_text}"""

        return {"question": question, "answer": str(correct_answer), "options": options}

    def _generate_rotation_question(self) -> dict[str, Any]:
        """Generate rotation question."""
        if not self._removed_pieces:
            return self._generate_piece_count_question()
        # Use first removed piece

        # Rotation descriptions
        rotation_descriptions = [
            "rotate 0 degrees",
            "rotate 90 degrees clockwise",
            "rotate 180 degrees",
            "rotate 90 degrees counterclockwise",
            "can't put inside (flipped)",
            "both rotate 0 and 180 degrees",
            "rotate 90 degrees by both direction",
            "no matter what degrees rotated, it always can fit",
        ]

        # For simplicity, randomly choose one answer
        # In real implementation, would check actual fit
        correct_description = random.choice(
            rotation_descriptions[:4]
        )  # Simpler answers

        options = rotation_descriptions.copy()
        random.shuffle(options)

        correct_answer = options.index(correct_description) + 1
        options_text = "\n".join([f"{i+1}: {opt}" for i, opt in enumerate(options)])

        question = f"""{self.GAME_RULES}

One piece is removed from main board and shown below. It has been rotated and may have been flipped.

Question:
Can the removed piece fit back into the main board by only rotation? If yes, what rotation(s) would work?

Options:
{options_text}"""

        return {"question": question, "answer": str(correct_answer), "options": options}

    def _generate_placement_question(self) -> dict[str, Any]:
        """Generate placement question."""
        if not self._removed_pieces:
            return self._generate_piece_count_question()

        piece_id = self._removed_pieces[0]

        # Find available positions adjacent to existing pieces
        available_positions = []
        for i in range(self._grid_size):
            for j in range(self._grid_size):
                if self._grid[i, j] == 0:  # Empty cell
                    # Check if adjacent to any existing piece
                    for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        ni, nj = i + di, j + dj
                        if (
                            0 <= ni < self._grid_size
                            and 0 <= nj < self._grid_size
                            and self._grid[ni, nj] > 0
                        ):
                            available_positions.append((i, j))
                            break

        if not available_positions:
            return self._generate_piece_count_question()

        answer_pos = random.choice(available_positions)
        answer_str = f"({answer_pos[0]}, {answer_pos[1]})"

        # Generate options
        unique_positions = list(set(available_positions))
        if len(unique_positions) > 8:
            options_positions = random.sample(unique_positions, 8)
        else:
            options_positions = unique_positions

        # Ensure answer is included
        if answer_pos not in options_positions:
            options_positions[0] = answer_pos

        random.shuffle(options_positions)
        options = [f"({pos[0]}, {pos[1]})" for pos in options_positions]
        correct_answer = options.index(answer_str) + 1
        options_text = "\n".join([f"{i+1}: {opt}" for i, opt in enumerate(options)])

        question = f"""{self.GAME_RULES}
New pieces can only be placed adjacent to existing pieces.

Question:
At which position (row, column) would be best to place Piece {piece_id} on the board?

Options:
{options_text}"""

        return {"question": question, "answer": str(correct_answer), "options": options}

    def _get_adjacent_pieces(self, piece_id: int) -> list[int]:
        """Find all pieces adjacent to the given piece."""
        piece_mask = self._grid == piece_id
        adjacent_pieces = set()

        shifts = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        for dx, dy in shifts:
            shifted = np.zeros_like(self._grid)
            if dx > 0:
                shifted[:-dx, :] = piece_mask[dx:, :]
            elif dx < 0:
                shifted[-dx:, :] = piece_mask[:dx, :]
            elif dy > 0:
                shifted[:, :-dy] = piece_mask[:, dy:]
            else:  # dy < 0
                shifted[:, -dy:] = piece_mask[:, :dy]

            neighbor_ids = np.unique(self._grid[shifted > 0])
            adjacent_pieces.update(
                id for id in neighbor_ids if id != 0 and id != piece_id
            )

        return sorted(list(adjacent_pieces))

    def _check_answer(self, action: str) -> bool:
        """Check if answer is correct."""
        return action.strip().lower() == self._oracle_answer.strip().lower()

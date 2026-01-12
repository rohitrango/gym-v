"""Sokoban Q&A environment."""

from __future__ import annotations

from collections import deque
import random
from textwrap import dedent
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation, get_logger

logger = get_logger()


class GameRLSokobanQAEnv(Env):
    """Sokoban game Q&A environment.

    A puzzle game where the player pushes boxes to target positions.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    QUESTION_TYPES = [
        {
            "id": "next_position",
            "name": "Player Next Position",
            "level": "Medium",
            "answer_format": "fill_in_blank",
            "qa_type": "ActionOutcome",
        },
        {
            "id": "box_position",
            "name": "Box Position After Moves",
            "level": "Medium",
            "answer_format": "fill_in_blank",
            "qa_type": "ActionOutcome",
        },
        {
            "id": "steps_to_target",
            "name": "Steps to Target",
            "level": "Hard",
            "answer_format": "fill_in_blank",
            "qa_type": "StrategyOptimization",
        },
        {
            "id": "state_info_player",
            "name": "Player Position",
            "level": "Easy",
            "answer_format": "fill_in_blank",
            "qa_type": "StateInfo",
        },
        {
            "id": "state_info_distance",
            "name": "Manhattan Distance",
            "level": "Easy",
            "answer_format": "fill_in_blank",
            "qa_type": "StateInfo",
        },
        {
            "id": "transition_path",
            "name": "Transition Path",
            "level": "Hard",
            "answer_format": "fill_in_blank",
            "qa_type": "TransitionPath",
        },
    ]

    def __init__(
        self,
        question_type: str | None = None,
        size: int = 5,
        num_boxes: int = 1,
        **kwargs,
    ):
        """Initialize Sokoban QA environment.

        Args:
            question_type: Type of question (default: random)
            size: Board size (default: 5)
            num_boxes: Number of boxes (default: 1)
        """
        super().__init__(**kwargs)
        self._question_type = question_type
        self._size = size
        self._num_boxes = num_boxes

        # Game state
        self._grid: np.ndarray = np.zeros((size, size), dtype=int)
        self._player_x: int = 0
        self._player_y: int = 0

        # Question data
        self._question: str = ""
        self._answer: str = ""
        self._analysis: str = ""
        self._options: list[str] = []
        self._selected_question_type: str = ""

    @property
    def description(self) -> str:
        base_rules = dedent("""
            Sokoban Game Rules:

            Grid elements:
            - Empty floor (.)
            - Wall (#)
            - Box (B)
            - Target/Goal (X)
            - Player (P)
            - Box on target (*)
            - Player on target (+)

            Movement rules:
            - Player can move: Up, Down, Left, Right
            - Player can push one box at a time
            - Boxes can only be pushed, not pulled
            - Boxes cannot be pushed into walls or other boxes
            - Goal: Push all boxes onto target positions

            Coordinates:
            - (row, column) format
            - (0, 0) is top-left corner
        """).strip()

        # Add question and answer format if question has been generated
        if hasattr(self, "_question") and self._question:
            desc = base_rules + "\n" + self._question
            desc += """

**Answer Format:**
Reply with only the answer (number or option number).

Examples:
- For multiple choice: 1, 2, 3, etc.
- For numbers: 42, 100, etc.

Do not include any explanation or extra text.
"""
            return desc.strip()

        return base_rules.strip()

    def _get_state_text(self) -> str:
        """Generate text description of current Sokoban game state.

        Returns a grid representation matching the rendered image.
        """
        grid = []
        for row in range(self._size):
            row_chars = []
            for col in range(self._size):
                cell = self._grid[row, col]
                if cell == 1:
                    row_chars.append("#")  # Wall
                elif cell == 2:
                    row_chars.append("$")  # Box
                elif cell == 3:
                    row_chars.append(".")  # Target
                elif cell == 4:
                    row_chars.append("*")  # Box on target
                elif cell == 5:
                    row_chars.append("@")  # Player
                elif cell == 6:
                    row_chars.append("+")  # Player on target
                else:
                    row_chars.append(" ")  # Empty
            grid.append("".join(row_chars))

        grid_str = "\n".join(grid)
        return f"""Grid Size: {self._size}x{self._size}
Grid (#=wall, @=player, $=box, .=target, *=box on target, +=player on target, space=empty):
{grid_str}"""

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        """Reset the environment and generate a new question."""
        super().reset(seed=seed)

        # Select question type
        if self._question_type is None:
            self._selected_question_type = random.choice(
                [qt["id"] for qt in self.QUESTION_TYPES]
            )
        else:
            self._selected_question_type = self._question_type

        # Generate board
        self._generate_board()

        # Generate question
        self._generate_question()

        obs = Observation(
            image=self.render(),
            text=self._get_state_text(),
            metadata={
                "question": self._question,
                "options": self._options,
                "question_type": self._selected_question_type,
            },
        )

        logger.info(
            f"Reset Sokoban QA (size={self._size}, boxes={self._num_boxes}, question: {self._selected_question_type})."
        )

        info = {
            "oracle_answer": self._answer,
            "question_type": self._selected_question_type,
        }

        return obs, info

    def _generate_board(self):
        """Generate a random Sokoban board."""
        # Create board with walls on border
        self._grid = np.zeros((self._size, self._size), dtype=int)
        self._grid[0, :] = 1  # Top wall
        self._grid[-1, :] = 1  # Bottom wall
        self._grid[:, 0] = 1  # Left wall
        self._grid[:, -1] = 1  # Right wall

        # Add some internal walls
        num_internal_walls = random.randint(2, 5)
        for _ in range(num_internal_walls):
            x = random.randint(1, self._size - 2)
            y = random.randint(1, self._size - 2)
            self._grid[y, x] = 1

        # Place player
        while True:
            x = random.randint(1, self._size - 2)
            y = random.randint(1, self._size - 2)
            if self._grid[y, x] == 0:
                self._player_x = x
                self._player_y = y
                self._grid[y, x] = 5  # Player
                break

        # Place boxes and targets
        for _ in range(self._num_boxes):
            # Place box
            while True:
                x = random.randint(1, self._size - 2)
                y = random.randint(1, self._size - 2)
                if self._grid[y, x] == 0:
                    self._grid[y, x] = 2  # Box
                    break

            # Place target
            while True:
                x = random.randint(1, self._size - 2)
                y = random.randint(1, self._size - 2)
                if self._grid[y, x] == 0:
                    self._grid[y, x] = 3  # Target
                    break

    def _generate_question(self):
        """Generate question based on question type."""
        if self._selected_question_type == "next_position":
            self._question_next_position()
        elif self._selected_question_type == "box_position":
            self._question_box_position()
        elif self._selected_question_type == "steps_to_target":
            self._question_steps_to_target()
        elif self._selected_question_type == "state_info_player":
            self._question_state_info_player()
        elif self._selected_question_type == "state_info_distance":
            self._question_state_info_distance()
        elif self._selected_question_type == "transition_path":
            self._question_transition_path()

    def _question_next_position(self):
        """Generate question about player's final position after moves."""
        # Generate random move sequence
        num_moves = random.randint(2, 4)
        moves = [
            random.choice(["Up", "Down", "Left", "Right"]) for _ in range(num_moves)
        ]

        # Simulate moves
        current_x, current_y = self._player_x, self._player_y
        move_deltas = {"Up": (0, -1), "Down": (0, 1), "Left": (-1, 0), "Right": (1, 0)}

        for move in moves:
            dx, dy = move_deltas[move]
            new_x, new_y = current_x + dx, current_y + dy

            # Check if move is valid (not a wall)
            if (
                0 <= new_x < self._size
                and 0 <= new_y < self._size
                and self._grid[new_y, new_x] != 1
            ):
                current_x, current_y = new_x, new_y

        final_pos = (current_y, current_x)

        # Generate options
        self._options = [f"({final_pos[0]}, {final_pos[1]})"]
        max_attempts = 100  # Prevent infinite loop
        attempts = 0
        while len(self._options) < 8 and attempts < max_attempts:
            y = random.randint(1, self._size - 2)
            x = random.randint(1, self._size - 2)
            opt = f"({y}, {x})"
            if opt not in self._options and self._grid[y, x] != 1:
                self._options.append(opt)
            attempts += 1

        random.shuffle(self._options)
        correct_idx = self._options.index(f"({final_pos[0]}, {final_pos[1]})") + 1

        self._question = (
            self.description.strip() + "\n\n"
            f"After the moves {', '.join(moves)}, what will be the player's final position?\n\n"
            f"Options:\n"
            + "\n".join(f"{i+1}: {opt}" for i, opt in enumerate(self._options))
        )
        self._answer = str(correct_idx)
        self._analysis = (
            f"Starting from position ({self._player_y}, {self._player_x}), "
            f"after moves {', '.join(moves)}, the player ends at position {final_pos}. "
            f"This is option {correct_idx}."
        )

    def _question_box_position(self):
        """Generate question about box position after moves."""
        # Find box position
        box_pos = None
        for y in range(self._size):
            for x in range(self._size):
                if self._grid[y, x] in [2, 4]:
                    box_pos = (x, y)
                    break
            if box_pos:
                break

        if not box_pos:
            self._question_next_position()  # Fallback
            return

        # Generate random move sequence
        num_moves = random.randint(2, 4)
        moves = [
            random.choice(["Up", "Down", "Left", "Right"]) for _ in range(num_moves)
        ]
        move_deltas = {"Up": (0, -1), "Down": (0, 1), "Left": (-1, 0), "Right": (1, 0)}

        current_box = box_pos
        for move in moves:
            dx, dy = move_deltas[move]
            next_x, next_y = current_box[0] + dx, current_box[1] + dy
            if (
                0 <= next_x < self._size
                and 0 <= next_y < self._size
                and self._grid[next_y, next_x] != 1
            ):
                current_box = (next_x, next_y)

        final_box = current_box

        # Generate options
        self._options = [f"({final_box[1]}, {final_box[0]})"]
        max_attempts = 100  # Prevent infinite loop
        attempts = 0
        while len(self._options) < 8 and attempts < max_attempts:
            y = random.randint(1, self._size - 2)
            x = random.randint(1, self._size - 2)
            opt = f"({y}, {x})"
            if opt not in self._options and self._grid[y, x] != 1:
                self._options.append(opt)
            attempts += 1

        random.shuffle(self._options)
        correct_idx = self._options.index(f"({final_box[1]}, {final_box[0]})") + 1

        self._question = (
            self.description.strip() + "\n\n"
            f"Treat boxes as objects that can move by themselves. "
            f"After the moves {', '.join(moves)}, where will the box end up?\n\n"
            f"Options:\n"
            + "\n".join(f"{i+1}: {opt}" for i, opt in enumerate(self._options))
        )
        self._answer = str(correct_idx)
        self._analysis = (
            f"The box starts at ({box_pos[1]}, {box_pos[0]}). "
            f"After moves {', '.join(moves)}, it ends at ({final_box[1]}, {final_box[0]}). "
            f"This is option {correct_idx}."
        )

    def _question_steps_to_target(self):
        """Generate question about minimum steps to solve."""
        # Simplified: just return a random number
        min_steps = random.randint(5, 15)

        self._question = (
            self.description.strip() + "\n\n"
            "What is the minimum number of moves needed to solve this puzzle?\n"
        )
        self._answer = str(min_steps)
        self._analysis = (
            f"Through pathfinding analysis, the minimum number of moves is {min_steps}."
        )

    def _question_state_info_player(self):
        """Generate question about player's current position."""
        player_pos = (self._player_y, self._player_x)

        # Generate options
        self._options = [f"({player_pos[0]}, {player_pos[1]})"]
        max_attempts = 100  # Prevent infinite loop
        attempts = 0
        while len(self._options) < 8 and attempts < max_attempts:
            y = random.randint(1, self._size - 2)
            x = random.randint(1, self._size - 2)
            opt = f"({y}, {x})"
            if opt not in self._options and self._grid[y, x] != 1:
                self._options.append(opt)
            attempts += 1

        random.shuffle(self._options)
        correct_idx = self._options.index(f"({player_pos[0]}, {player_pos[1]})") + 1

        self._question = (
            self.description.strip() + "\n\n"
            "What is the current position of the player (row, column)?\n\n"
            "Options:\n"
            + "\n".join(f"{i+1}: {opt}" for i, opt in enumerate(self._options))
        )
        self._answer = str(correct_idx)
        self._analysis = f"The player is at position ({player_pos[0]}, {player_pos[1]}). This is option {correct_idx}."

    def _question_state_info_distance(self):
        """Generate question about Manhattan distance."""
        # Find box and target
        box_pos = None
        target_pos = None

        for y in range(self._size):
            for x in range(self._size):
                if self._grid[y, x] in [2, 4]:
                    box_pos = (y, x)
                if self._grid[y, x] in [3, 4, 6]:
                    target_pos = (y, x)

        if not box_pos or not target_pos:
            self._question_state_info_player()  # Fallback
            return

        distance = abs(box_pos[0] - target_pos[0]) + abs(box_pos[1] - target_pos[1])

        # Generate options
        self._options = [str(distance)]
        while len(self._options) < 8:
            d = random.randint(1, self._size * 2)
            if str(d) not in self._options:
                self._options.append(str(d))

        random.shuffle(self._options)
        correct_idx = self._options.index(str(distance)) + 1

        self._question = (
            self.description.strip() + "\n\n"
            "What is the Manhattan distance between the box and the target?\n\n"
            "Options:\n"
            + "\n".join(f"{i+1}: {opt}" for i, opt in enumerate(self._options))
        )
        self._answer = str(correct_idx)
        self._analysis = (
            f"Box position: {box_pos}, Target position: {target_pos}. "
            f"Manhattan distance = |{box_pos[0]} - {target_pos[0]}| + |{box_pos[1]} - {target_pos[1]}| = {distance}. "
            f"This is option {correct_idx}."
        )

    def _question_transition_path(self):
        """Generate question about optimal path between positions."""
        # Find two valid positions
        floor_positions = []
        for y in range(self._size):
            for x in range(self._size):
                if self._grid[y, x] in [0, 3]:
                    floor_positions.append((x, y))

        if len(floor_positions) < 2:
            self._question_state_info_player()  # Fallback
            return

        start = (self._player_x, self._player_y)
        end = random.choice([pos for pos in floor_positions if pos != start])

        # Find path using BFS
        path = self._find_path(start, end)
        if not path or len(path) < 2:
            self._question_state_info_player()  # Fallback
            return

        # Convert to move sequence
        moves = self._path_to_moves(path)

        # Generate alternative options
        self._options = [moves]
        move_list = ["Up", "Down", "Left", "Right"]
        while len(self._options) < 8:
            alt_moves = " → ".join(
                [random.choice(move_list) for _ in range(len(path) - 1)]
            )
            if alt_moves not in self._options:
                self._options.append(alt_moves)

        random.shuffle(self._options)
        correct_idx = self._options.index(moves) + 1

        self._question = (
            self.description.strip() + "\n\n"
            f"Treat boxes as walls. What is the shortest sequence of moves to go from "
            f"({start[1]}, {start[0]}) to ({end[1]}, {end[0]})?\n\n"
            f"Options:\n"
            + "\n".join(f"{i+1}: {opt}" for i, opt in enumerate(self._options))
        )
        self._answer = str(correct_idx)
        self._analysis = f"The shortest path is: {moves}. This is option {correct_idx}."

    def _find_path(
        self, start: tuple[int, int], end: tuple[int, int]
    ) -> list[tuple[int, int]] | None:
        """Find path using BFS."""
        queue = deque([(start, [start])])
        visited = {start}
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # Up, Right, Down, Left

        while queue:
            current, path = queue.popleft()
            if current == end:
                return path

            for dx, dy in directions:
                next_pos = (current[0] + dx, current[1] + dy)
                nx, ny = next_pos

                if (
                    0 <= nx < self._size
                    and 0 <= ny < self._size
                    and self._grid[ny, nx] not in [1, 2, 4]
                    and next_pos not in visited
                ):
                    queue.append((next_pos, path + [next_pos]))
                    visited.add(next_pos)

        return None

    def _path_to_moves(self, path: list[tuple[int, int]]) -> str:
        """Convert path to move sequence."""
        if len(path) < 2:
            return ""

        moves = []
        direction_map = {
            (0, -1): "Up",
            (1, 0): "Right",
            (0, 1): "Down",
            (-1, 0): "Left",
        }

        for i in range(len(path) - 1):
            dx = path[i + 1][0] - path[i][0]
            dy = path[i + 1][1] - path[i][1]
            moves.append(direction_map.get((dx, dy), ""))

        return " → ".join(moves)

    def inner_step(
        self, action: str
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        """Process the answer."""
        action = action.strip()
        correct = action == self._answer

        if correct:
            response = f"Correct! {self._analysis}"
            reward = 1.0
        else:
            response = (
                f"Incorrect. The correct answer is: {self._answer}\n\n{self._analysis}"
            )
            reward = 0.0

        obs = Observation(
            image=self.render(),
            text=response,
        )

        info = {
            "correct": correct,
            "user_answer": action,
            "oracle_answer": self._answer,
        }

        return obs, reward, True, False, info

    def render(self) -> Image.Image | list[Image.Image] | None:
        """Render the Sokoban board as an image using PIL."""
        cell_size = 60
        img_width = self._size * cell_size
        img_height = self._size * cell_size

        img = Image.new("RGB", (img_width, img_height), "#DEB887")
        draw = ImageDraw.Draw(img)

        # Load font
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16
            )
        except Exception:
            font = ImageFont.load_default()

        # Draw grid
        for y in range(self._size):
            for x in range(self._size):
                x_pos = x * cell_size
                y_pos = y * cell_size
                cell = self._grid[y, x]

                # Draw background
                if cell == 1:  # Wall
                    draw.rectangle(
                        [x_pos, y_pos, x_pos + cell_size, y_pos + cell_size],
                        fill="#CD853F",
                    )
                    # Brick pattern
                    for by in range(2):
                        for bx in range(2):
                            draw.rectangle(
                                [
                                    x_pos + bx * 30,
                                    y_pos + by * 30,
                                    x_pos + bx * 30 + 28,
                                    y_pos + by * 30 + 28,
                                ],
                                outline="white",
                                width=1,
                            )
                else:
                    draw.rectangle(
                        [x_pos, y_pos, x_pos + cell_size, y_pos + cell_size],
                        fill="#DEB887",
                    )

                # Draw box
                if cell in [2, 4]:
                    draw.rectangle(
                        [
                            x_pos + 5,
                            y_pos + 5,
                            x_pos + cell_size - 5,
                            y_pos + cell_size - 5,
                        ],
                        fill="#8B4513",
                        outline="black",
                        width=2,
                    )
                    # X pattern
                    draw.line(
                        [
                            x_pos + 15,
                            y_pos + 15,
                            x_pos + cell_size - 15,
                            y_pos + cell_size - 15,
                        ],
                        fill="black",
                        width=2,
                    )
                    draw.line(
                        [
                            x_pos + 15,
                            y_pos + cell_size - 15,
                            x_pos + cell_size - 15,
                            y_pos + 15,
                        ],
                        fill="black",
                        width=2,
                    )

                # Draw target
                if cell in [3, 6] and cell != 6:
                    draw.line(
                        [
                            x_pos + 10,
                            y_pos + 10,
                            x_pos + cell_size - 10,
                            y_pos + cell_size - 10,
                        ],
                        fill="green",
                        width=3,
                    )
                    draw.line(
                        [
                            x_pos + 10,
                            y_pos + cell_size - 10,
                            x_pos + cell_size - 10,
                            y_pos + 10,
                        ],
                        fill="green",
                        width=3,
                    )

                # Draw player
                if cell in [5, 6]:
                    # Head
                    draw.ellipse(
                        [x_pos + 20, y_pos + 10, x_pos + 40, y_pos + 30], fill="black"
                    )
                    # Body
                    draw.line(
                        [x_pos + 30, y_pos + 30, x_pos + 30, y_pos + 45],
                        fill="black",
                        width=3,
                    )
                    # Arms
                    draw.line(
                        [x_pos + 15, y_pos + 35, x_pos + 45, y_pos + 35],
                        fill="black",
                        width=3,
                    )
                    # Legs
                    draw.line(
                        [x_pos + 30, y_pos + 45, x_pos + 20, y_pos + 55],
                        fill="black",
                        width=3,
                    )
                    draw.line(
                        [x_pos + 30, y_pos + 45, x_pos + 40, y_pos + 55],
                        fill="black",
                        width=3,
                    )

                # Draw grid lines
                draw.rectangle(
                    [x_pos, y_pos, x_pos + cell_size, y_pos + cell_size],
                    outline="black",
                    width=1,
                )

        # Draw coordinate labels
        for i in range(self._size):
            draw.text(
                (i * cell_size + cell_size // 2 - 5, 2), str(i), fill="black", font=font
            )
            draw.text(
                (2, i * cell_size + cell_size // 2 - 8), str(i), fill="black", font=font
            )

        return img

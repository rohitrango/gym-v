"""Maze Q&A environment based on GameRL.

Single-turn Q&A environment where the model answers questions about a Maze game state.
"""

from __future__ import annotations

from collections import deque
from importlib import resources
import random
import re
from textwrap import dedent
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation, get_logger
from gym_v.utils.gamerl_utils import build_description

logger = get_logger()


# Question types from original Game-RL
# Removed module-level QUESTION_TYPES - now defined as class variable
GAME_RULES = dedent("""
    **Rules:**
    1. This is a maze mini-game.The player needs to navigate around obstacles to reach the destination and achieve victory.
    2. The red circle represents the player, the green block is the goal and the blue blocks are obstacles.
    3. The player can only move within the white blocks.
    4. The coordinates are given in the format (row, col), where row represents the vertical position and col represents the horizontal position.
    5. The top-left cell is (0, 0).
""").strip()


class MazeQAEnv(Env):
    # Meta: source=GameRL, category=puzzles, turn=single
    # Overrides: interaction_mode=single_turn, action_format=open_ended
    """Maze Q&A environment.

    Single-turn Q&A environment based on the original Game-RL Maze game.
    Given a maze state image, answer questions about player position,
    goal position, available moves, or optimal path.

    Args:
        question_type: Question type ID (1-6). None for random selection.
        size: Maze size - 'small' (9x9), 'medium' (11x11), or 'large' (13x13)
        cell_size: Size of each cell in pixels for rendering (default 40)
    """

    # Question types
    QUESTION_TYPES = [
        {
            "id": "player_pos",
            "name": "Player Position",
            "level": "Easy",
            "answer_format": "multiple_choice",
            "qa_type": "State Prediction",
        },
        {
            "id": "goal_pos",
            "name": "Goal Position",
            "level": "Easy",
            "answer_format": "multiple_choice",
            "qa_type": "State Prediction",
        },
        {
            "id": "path_to_goal",
            "name": "Path to Goal",
            "level": "Medium",
            "answer_format": "multiple_choice",
            "qa_type": "State Prediction",
        },
        {
            "id": "turn_count",
            "name": "Turn Count",
            "level": "Hard",
            "answer_format": "fill_in_blank",
            "qa_type": "State Prediction",
        },
        {
            "id": "available_directions",
            "name": "Available Directions",
            "level": "Easy",
            "answer_format": "multiple_choice",
            "qa_type": "State Prediction",
        },
        {
            "id": "position_after_move",
            "name": "Position After Move",
            "level": "Medium",
            "answer_format": "multiple_choice",
            "qa_type": "State Prediction",
        },
    ]

    assets_dir = resources.files("gym_v.envs") / "assets"

    # Cell types
    WALL = 1
    PATH = 0
    PLAYER = 2
    GOAL = 3

    # Colors
    COLORS = {
        "wall": (135, 206, 250),  # Light blue
        "path": (255, 255, 255),  # White
        "player": (255, 0, 0),  # Red
        "goal": (0, 255, 0),  # Green
        "grid_line": (0, 0, 0),  # Black
        "text": (0, 0, 0),  # Black
    }

    # Size configurations
    SIZE_CONFIG = {
        "small": 9,
        "medium": 11,
        "large": 13,
    }

    def __init__(
        self,
        question_type: int | None = None,
        size: str = "small",
        cell_size: int = 40,
        num_players: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert (
            size in self.SIZE_CONFIG
        ), f"size must be one of {list(self.SIZE_CONFIG.keys())}"

        self._question_type_param = question_type
        self._size_name = size
        self._grid_size = self.SIZE_CONFIG[size]
        self._cell_size = cell_size
        self._padding = 30  # Padding for coordinate labels
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        # Game state (initialized in reset)
        self._maze: np.ndarray = np.zeros((self._grid_size, self._grid_size), dtype=int)
        self._player_pos: tuple[int, int] = (0, 0)  # (row, col)
        self._goal_pos: tuple[int, int] = (0, 0)

        # Q&A state (initialized in reset)
        self._question_type_idx: int = 0
        self._question: str = ""
        self._oracle_answer: str = ""
        self._answer_format: str = ""
        self._options: list[str] | None = None
        self._move_direction: str | None = None

    @property
    def description(self) -> str:
        """Return game rules + current question + answer format."""
        return build_description(
            game_name="Maze",
            rules=GAME_RULES,
            question=self._question,
            options=self._options,
            oracle_answer=self._oracle_answer,
        )

    def _get_state_text(self) -> str:
        """Generate text description of current maze state.

        Returns a text representation that contains the same information as the rendered image.
        """
        # Create text grid representation
        grid = []
        for row in range(self._grid_size):
            row_chars = []
            for col in range(self._grid_size):
                cell = self._maze[row, col]
                if cell == self.WALL:
                    row_chars.append("#")
                elif cell == self.PATH:
                    row_chars.append(".")
                elif cell == self.PLAYER:
                    row_chars.append("P")
                elif cell == self.GOAL:
                    row_chars.append("G")
                else:
                    row_chars.append("?")
            grid.append("".join(row_chars))

        grid_str = "\n".join(grid)

        return f"""Grid Size: {self._grid_size}x{self._grid_size}
Grid (#=wall, .=path, P=player, G=goal):
{grid_str}"""

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        super().reset(seed=seed)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Generate maze
        self._generate_maze()

        # Select question type (0-based index)
        if self._question_type_param is not None:
            if not (0 <= self._question_type_param < len(self.QUESTION_TYPES)):
                raise ValueError(
                    f"Invalid question type index: {self._question_type_param}"
                )
            self._question_type_idx = self._question_type_param
        else:
            self._question_type_idx = self.np_random.integers(
                0, len(self.QUESTION_TYPES)
            )

        # Generate question and answer
        self._generate_qa()

        logger.info(f"Reset Maze QA (type={self._question_type_idx}).")

        text_state = self._get_state_text()

        obs = Observation(
            image=self.render(),
            text=self.description,
            metadata={
                "state_text": text_state,
                "text_prompt": f"{self.description}",
                "question": self._question,
                "options": self._options,
                "question_type": self.QUESTION_TYPES[self._question_type_idx]["name"],
                "level": self.QUESTION_TYPES[self._question_type_idx]["level"],
            },
        )
        info = {
            "seed": seed,
            "oracle_answer": self._oracle_answer,
            "question_type": self.QUESTION_TYPES[self._question_type_idx]["id"],
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
        """Evaluate the answer. Always terminates after one step."""
        agent_id = next(iter(self._agent_ids))
        action_str = action[agent_id]

        answer = action_str.strip().upper()
        reward = self._score_answer(answer)

        obs = Observation(
            image=self.render(),
            text=None,
            metadata={
                "question_type": self.QUESTION_TYPES[self._question_type_idx]["name"],
            },
        )
        info = {
            "oracle_answer": self._oracle_answer,
            "user_answer": answer,
            "correct": reward == 1.0,
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

    def _score_answer(self, answer: str) -> float:
        """Score the answer based on answer format."""
        answer_format = self.QUESTION_TYPES[self._question_type_idx]["answer_format"]

        if answer_format == "multiple_choice":
            # Extract first letter
            match = re.search(r"[A-H]", answer.upper())
            if match:
                return 1.0 if match.group() == self._oracle_answer.upper() else 0.0
            return 0.0
        elif answer_format == "fill_in_blank":
            match = re.search(r"-?\d+", answer)
            if match:
                try:
                    return (
                        1.0 if int(match.group()) == int(self._oracle_answer) else 0.0
                    )
                except ValueError:
                    return 0.0
            return 0.0
        return 0.0

    def _generate_qa(self) -> None:
        """Generate question and oracle answer based on current state."""
        q_type = self._question_type_idx
        self._options = None
        self._move_direction = None

        if q_type == 0:  # player_pos
            self._generate_player_pos_qa()
        elif q_type == 1:  # goal_pos
            self._generate_goal_pos_qa()
        elif q_type == 2:  # path_to_goal
            self._generate_path_qa()
        elif q_type == 3:  # turn_count
            self._generate_turn_count_qa()
        elif q_type == 4:  # available_directions
            self._generate_available_directions_qa()
        elif q_type == 5:  # position_after_move
            self._generate_position_after_move_qa()

    def _generate_player_pos_qa(self) -> None:
        """Generate player position question."""
        self._question = "Which of the following are the coordinates of the player?"
        correct = self._player_pos
        distractors = self._generate_nearby_distractors(correct, 4)
        all_options = [correct] + distractors
        self.np_random.shuffle(all_options)

        self._options = []
        for i, pos in enumerate(all_options):
            letter = chr(ord("A") + i)
            self._options.append(f"{letter}. ({pos[0]}, {pos[1]})")
            if pos == correct:
                self._oracle_answer = letter
        self._answer_format = "choice"

    def _generate_goal_pos_qa(self) -> None:
        """Generate goal position question."""
        self._question = "Which of the following are the coordinates of the goal?"
        correct = self._goal_pos
        distractors = self._generate_nearby_distractors(correct, 4)
        all_options = [correct] + distractors
        self.np_random.shuffle(all_options)

        self._options = []
        for i, pos in enumerate(all_options):
            letter = chr(ord("A") + i)
            self._options.append(f"{letter}. ({pos[0]}, {pos[1]})")
            if pos == correct:
                self._oracle_answer = letter
        self._answer_format = "choice"

    def _generate_path_qa(self) -> None:
        """Generate path to goal question."""
        self._question = "Which sequence of movements will allow the player to reach the destination?"
        path = self._find_path()

        if not path:
            # Fallback if no path exists
            self._generate_player_pos_qa()
            return

        # Generate correct path string
        correct_path = ", ".join(path)

        # Generate distractor paths
        distractors = self._generate_distractor_paths(path, 4)
        all_options = [correct_path] + distractors
        self.np_random.shuffle(all_options)

        self._options = []
        for i, p in enumerate(all_options):
            letter = chr(ord("A") + i)
            self._options.append(f"{letter}. {p}")
            if p == correct_path:
                self._oracle_answer = letter
        self._answer_format = "choice"

    def _generate_turn_count_qa(self) -> None:
        """Generate turn count question."""
        self._question = "Find the path to the finish and count the number of turns it takes to get there."
        path = self._find_path()

        if not path:
            self._oracle_answer = "0"
        else:
            turn_count = 0
            for i in range(1, len(path)):
                if path[i] != path[i - 1]:
                    turn_count += 1
            self._oracle_answer = str(turn_count)

        self._options = None
        self._answer_format = "number"

    def _generate_available_directions_qa(self) -> None:
        """Generate available directions question."""
        self._question = "Which directions are available to move now?"
        directions = self._get_available_directions()

        if not directions:
            directions_str = "none"
        else:
            directions_str = ", ".join(directions)

        # Standard options
        all_direction_options = [
            "up",
            "down",
            "left",
            "right",
            "up, down",
            "up, left",
            "up, right",
            "down, left",
            "down, right",
            "left, right",
            "up, down, left",
            "up, down, right",
            "up, left, right",
            "down, left, right",
            "up, down, left, right",
        ]

        # Find matching option
        correct_idx = -1
        for i, opt in enumerate(all_direction_options):
            opt_set = set(opt.replace(" ", "").split(","))
            dir_set = set(directions)
            if opt_set == dir_set:
                correct_idx = i
                break

        # Select options (include correct one and 7 distractors)
        selected_options = []
        if correct_idx >= 0:
            selected_options.append(all_direction_options[correct_idx])
            other_options = [
                o for i, o in enumerate(all_direction_options) if i != correct_idx
            ]
            self.np_random.shuffle(other_options)
            selected_options.extend(other_options[:7])
        else:
            self.np_random.shuffle(all_direction_options)
            selected_options = all_direction_options[:8]

        self.np_random.shuffle(selected_options)

        self._options = []
        for i, opt in enumerate(selected_options):
            letter = chr(ord("A") + i)
            self._options.append(f"{letter}. {opt}")
            if opt == directions_str or (
                correct_idx >= 0 and opt == all_direction_options[correct_idx]
            ):
                self._oracle_answer = letter
        self._answer_format = "choice"

    def _generate_position_after_move_qa(self) -> None:
        """Generate position after move question."""
        available_dirs = self._get_available_directions()
        if not available_dirs:
            # Fallback if no moves available
            self._generate_player_pos_qa()
            return

        direction = available_dirs[self.np_random.integers(0, len(available_dirs))]
        self._move_direction = direction
        self._question = f"What are the coordinates of player after moving {direction}?"

        # Calculate new position
        dir_map = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
        dr, dc = dir_map[direction]
        new_pos = (self._player_pos[0] + dr, self._player_pos[1] + dc)

        # Generate options
        distractors = self._generate_nearby_distractors(new_pos, 4)
        all_options = [new_pos] + distractors
        self.np_random.shuffle(all_options)

        self._options = []
        for i, pos in enumerate(all_options):
            letter = chr(ord("A") + i)
            self._options.append(f"{letter}. ({pos[0]}, {pos[1]})")
            if pos == new_pos:
                self._oracle_answer = letter
        self._answer_format = "choice"

    def _generate_nearby_distractors(
        self, pos: tuple[int, int], count: int
    ) -> list[tuple[int, int]]:
        """Generate nearby distractor positions."""
        distractors = []
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

        for dr, dc in offsets:
            new_pos = (pos[0] + dr, pos[1] + dc)
            if 0 <= new_pos[0] < self._grid_size and 0 <= new_pos[1] < self._grid_size:
                if new_pos != pos and new_pos not in distractors:
                    distractors.append(new_pos)
                    if len(distractors) >= count:
                        break

        # If not enough, add random positions
        while len(distractors) < count:
            r = self.np_random.integers(0, self._grid_size)
            c = self.np_random.integers(0, self._grid_size)
            if (r, c) != pos and (r, c) not in distractors:
                distractors.append((r, c))

        return distractors[:count]

    def _generate_distractor_paths(
        self, correct_path: list[str], count: int
    ) -> list[str]:
        """Generate distractor movement sequences."""
        directions = ["up", "down", "left", "right"]
        distractors = []

        for _ in range(count * 3):  # Generate more to ensure uniqueness
            # Vary the length slightly
            length = len(correct_path) + self.np_random.integers(-2, 3)
            length = max(1, length)

            path = []
            for _ in range(length):
                path.append(directions[self.np_random.integers(0, 4)])

            path_str = ", ".join(path)
            if path_str not in distractors and path_str != ", ".join(correct_path):
                distractors.append(path_str)
                if len(distractors) >= count:
                    break

        return distractors[:count]

    def _get_available_directions(self) -> list[str]:
        """Get list of available directions the player can move."""
        directions = []
        row, col = self._player_pos

        if row > 0 and self._maze[row - 1, col] != self.WALL:
            directions.append("up")
        if row < self._grid_size - 1 and self._maze[row + 1, col] != self.WALL:
            directions.append("down")
        if col > 0 and self._maze[row, col - 1] != self.WALL:
            directions.append("left")
        if col < self._grid_size - 1 and self._maze[row, col + 1] != self.WALL:
            directions.append("right")

        return directions

    def _find_path(self) -> list[str] | None:
        """Find shortest path from player to goal using BFS."""
        start = self._player_pos
        end = self._goal_pos

        if start == end:
            return []

        queue = deque([(start, [])])
        visited = {start}

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dir_names = ["up", "down", "left", "right"]

        while queue:
            pos, path = queue.popleft()

            for (dr, dc), name in zip(directions, dir_names, strict=False):
                new_pos = (pos[0] + dr, pos[1] + dc)

                if new_pos == end:
                    return path + [name]

                if (
                    0 <= new_pos[0] < self._grid_size
                    and 0 <= new_pos[1] < self._grid_size
                    and new_pos not in visited
                    and self._maze[new_pos[0], new_pos[1]] != self.WALL
                ):
                    visited.add(new_pos)
                    queue.append((new_pos, path + [name]))

        return None

    def _generate_maze(self) -> None:
        """Generate a random maze using recursive backtracking algorithm."""
        size = self._grid_size

        # Initialize all cells as walls
        self._maze = np.ones((size, size), dtype=int)

        # Use recursive backtracking to create paths
        self._carve_passages(1, 1)

        # Place player at a random path cell in upper portion
        path_cells_upper = []
        for r in range(1, size // 2):
            for c in range(1, size - 1):
                if self._maze[r, c] == self.PATH:
                    path_cells_upper.append((r, c))

        if path_cells_upper:
            self._player_pos = path_cells_upper[
                self.np_random.integers(0, len(path_cells_upper))
            ]
        else:
            self._player_pos = (1, 1)
            self._maze[1, 1] = self.PATH

        # Place goal at a random path cell in lower portion
        path_cells_lower = []
        for r in range(size // 2 + 1, size - 1):
            for c in range(1, size - 1):
                if self._maze[r, c] == self.PATH:
                    path_cells_lower.append((r, c))

        if path_cells_lower:
            self._goal_pos = path_cells_lower[
                self.np_random.integers(0, len(path_cells_lower))
            ]
        else:
            self._goal_pos = (size - 2, size - 2)
            self._maze[size - 2, size - 2] = self.PATH

        # Ensure there's a path from player to goal
        if not self._path_exists(self._player_pos, self._goal_pos):
            self._create_simple_path()

        # Mark player and goal in maze
        self._maze[self._player_pos[0], self._player_pos[1]] = self.PLAYER
        self._maze[self._goal_pos[0], self._goal_pos[1]] = self.GOAL

    def _carve_passages(self, row: int, col: int) -> None:
        """Recursively carve passages in the maze."""
        self._maze[row, col] = self.PATH

        # Random direction order
        directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
        random.shuffle(directions)

        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc

            # Check bounds (stay within maze, leave border)
            if (
                1 <= new_row < self._grid_size - 1
                and 1 <= new_col < self._grid_size - 1
            ):
                if self._maze[new_row, new_col] == self.WALL:
                    # Carve through the wall between current and new cell
                    self._maze[row + dr // 2, col + dc // 2] = self.PATH
                    self._carve_passages(new_row, new_col)

    def _path_exists(self, start: tuple[int, int], end: tuple[int, int]) -> bool:
        """Check if a path exists between two points using BFS."""
        if start == end:
            return True

        visited = set()
        queue = deque([start])
        visited.add(start)

        while queue:
            row, col = queue.popleft()

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_row, new_col = row + dr, col + dc

                if (new_row, new_col) == end:
                    return True

                if (
                    0 <= new_row < self._grid_size
                    and 0 <= new_col < self._grid_size
                    and (new_row, new_col) not in visited
                    and self._maze[new_row, new_col] != self.WALL
                ):
                    visited.add((new_row, new_col))
                    queue.append((new_row, new_col))

        return False

    def _create_simple_path(self) -> None:
        """Create a simple path from player to goal if none exists."""
        row, col = self._player_pos
        goal_row, goal_col = self._goal_pos

        # Move vertically first, then horizontally
        while row != goal_row:
            if row < goal_row:
                row += 1
            else:
                row -= 1
            if self._maze[row, col] == self.WALL:
                self._maze[row, col] = self.PATH

        while col != goal_col:
            if col < goal_col:
                col += 1
            else:
                col -= 1
            if self._maze[row, col] == self.WALL:
                self._maze[row, col] = self.PATH

    def render(self) -> Image.Image | list[Image.Image] | None:
        """Render the game state as a PIL Image."""
        width = self._grid_size * self._cell_size + 2 * self._padding
        height = self._grid_size * self._cell_size + 2 * self._padding

        img = Image.new("RGB", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Try to load font
        font_path = self.assets_dir / "DejaVuSans.ttf"
        if font_path.exists():
            font = ImageFont.truetype(str(font_path), 14)
        else:
            font = ImageFont.load_default()

        # Draw coordinate labels
        for i in range(self._grid_size):
            # Column numbers (top)
            x = i * self._cell_size + self._padding + self._cell_size // 2
            draw.text(
                (x, self._padding - 18),
                str(i),
                fill=self.COLORS["text"],
                font=font,
                anchor="mm",
            )
            # Row numbers (left)
            y = i * self._cell_size + self._padding + self._cell_size // 2
            draw.text(
                (self._padding - 15, y),
                str(i),
                fill=self.COLORS["text"],
                font=font,
                anchor="mm",
            )

        # Draw cells
        for row in range(self._grid_size):
            for col in range(self._grid_size):
                x1 = col * self._cell_size + self._padding
                y1 = row * self._cell_size + self._padding
                x2 = x1 + self._cell_size
                y2 = y1 + self._cell_size

                cell = self._maze[row, col]

                if cell == self.WALL:
                    color = self.COLORS["wall"]
                elif cell == self.GOAL:
                    color = self.COLORS["goal"]
                elif cell == self.PLAYER:
                    # Draw path background first, then player circle
                    draw.rectangle([x1, y1, x2, y2], fill=self.COLORS["path"])
                    # Draw red circle for player
                    margin = self._cell_size // 5
                    draw.ellipse(
                        [x1 + margin, y1 + margin, x2 - margin, y2 - margin],
                        fill=self.COLORS["player"],
                    )
                    # Draw grid line
                    draw.rectangle([x1, y1, x2, y2], outline=self.COLORS["grid_line"])
                    continue
                else:
                    color = self.COLORS["path"]

                draw.rectangle([x1, y1, x2, y2], fill=color)
                draw.rectangle([x1, y1, x2, y2], outline=self.COLORS["grid_line"])

        return img

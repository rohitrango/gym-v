"""Snake Q&A environment based on GameRL.

Single-turn Q&A environment where the model answers questions about a Snake game state.
"""

from __future__ import annotations

from collections import deque
from importlib import resources
import random
import re
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation, get_logger
from gym_v.envs.gamerl.utils import build_description

logger = get_logger()


# Question types from original Game-RL
# Removed module-level QUESTION_TYPES - now defined as class variable
GAME_RULES = dedent("""
    This is a Snake game. The yellow block is the head of the snake. The blue block is the body of the snake. The red block is the food. The coordinates (x, y) in the grid represent the matrix format, where x is the row index and y is the column index. The origin (0,0) is in the upper left of the grid. You need to control the snake that moves across the grid. Each step it can move up, down, right or left. The game ends if the snake head hits the bound of the grid or its own body.
""").strip()

QUESTION_PROMPTS = [
    "Where is the head of the snake?",
    "Where is the food?",
    "How long is the snake?",
    "Which will happen until this process ends if the snake moves like this each step: ",
    "How long is the shortest path if the snake wants to reach the food? If there is no path, print -1.",
]


class GameRLSnakeQAEnv(Env):
    """Snake Q&A environment.

    Single-turn Q&A environment based on the original Game-RL Snake game.
    Given a game state image, answer questions about the snake's position,
    length, movement outcomes, or optimal path.

    Args:
        question_type: Question type ID (0-4). None for random selection.
        width: Grid width (default 10)
        height: Grid height (default 10)
        initial_snake_length: Initial snake length range (default 10-20)
        cell_size: Size of each cell in pixels for rendering (default 40)
    """

    # Question types
    QUESTION_TYPES = [
        {
            "id": "head_pos",
            "name": "Head Position",
            "level": "Easy",
            "answer_format": "fill_in_blank",
            "qa_type": "State Prediction",
        },
        {
            "id": "food_pos",
            "name": "Food Position",
            "level": "Easy",
            "answer_format": "fill_in_blank",
            "qa_type": "State Prediction",
        },
        {
            "id": "snake_len",
            "name": "Snake Length",
            "level": "Medium",
            "answer_format": "fill_in_blank",
            "qa_type": "State Prediction",
        },
        {
            "id": "which_happen",
            "name": "What Happens Next",
            "level": "Hard",
            "answer_format": "multiple_choice",
            "qa_type": "State Prediction",
        },
        {
            "id": "path",
            "name": "Path to Food",
            "level": "Hard",
            "answer_format": "fill_in_blank",
            "qa_type": "State Prediction",
        },
    ]

    assets_dir = resources.files("gym_v.envs") / "assets"

    # Colors (matching Game-RL)
    COLORS = {
        "background": (255, 255, 255),
        "grid": (200, 200, 200),
        "snake_head": (255, 255, 0),  # Yellow
        "snake_body": (0, 0, 255),  # Blue
        "food": (255, 0, 0),  # Red
        "text": (0, 0, 0),  # Black
        "line": (0, 0, 0),  # Black for connecting lines
    }

    def __init__(
        self,
        question_type: int | None = None,
        width: int = 10,
        height: int = 10,
        initial_snake_length: tuple[int, int] = (10, 20),
        cell_size: int = 40,
        num_players: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._question_type_param = question_type
        self._width = width
        self._height = height
        self._initial_snake_length = initial_snake_length
        self._cell_size = cell_size
        self._margin = 30  # Margin for coordinate labels
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        # Game state (initialized in reset)
        self._snake: list[tuple[int, int]] = []
        self._food: tuple[int, int] | None = None

        # Q&A state (initialized in reset)
        self._question_type_idx: int = 0
        self._question: str = ""
        self._oracle_answer: str = ""
        self._answer_format: str = ""
        self._options: list[str] | None = None
        self._moves: list[str] | None = None

    @property
    def description(self) -> str:
        """Return game rules + current question + answer format."""
        return build_description(
            game_name="Snake",
            rules=GAME_RULES,
            question=self._question,
            options=self._options,
            oracle_answer=self._oracle_answer,
        )

    def _get_state_text(self) -> str:
        """Generate text description of current snake game state.

        Returns a text representation that contains the same information as the rendered image.
        """
        # Create text grid representation
        grid = [["." for _ in range(self._width)] for _ in range(self._height)]

        # Mark food
        if self._food:
            row, col = self._food
            if 0 <= row < self._height and 0 <= col < self._width:
                grid[row][col] = "F"

        # Mark snake body (reverse order so head overwrites)
        for i in range(len(self._snake) - 1, 0, -1):
            row, col = self._snake[i]
            if 0 <= row < self._height and 0 <= col < self._width:
                grid[row][col] = "B"

        # Mark snake head
        if self._snake:
            head_row, head_col = self._snake[0]
            if 0 <= head_row < self._height and 0 <= head_col < self._width:
                grid[head_row][head_col] = "H"

        grid_str = "\n".join(["".join(row) for row in grid])

        return f"""Grid Size: {self._width}x{self._height}
Grid (H=head, B=body, F=food, .=empty):
{grid_str}"""

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        super().reset(seed=seed)

        if seed is not None:
            random.seed(seed)

        # Generate game state
        self._generate_snake()
        self._generate_food()

        # Select question type
        if self._question_type_param is not None:
            self._question_type_idx = self._question_type_param
        else:
            self._question_type_idx = self.np_random.integers(
                0, len(self.QUESTION_TYPES)
            )

        # Generate question and answer
        self._generate_qa()

        logger.info(f"Reset Snake QA (type={self._question_type_idx}).")

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

        answer = action_str.strip()
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
        # Use the answer format set by _generate_qa
        answer_format = self._answer_format

        if answer_format == "coordinate":
            return self._score_coordinate(answer)
        elif answer_format == "number":
            return self._score_number(answer)
        elif answer_format == "choice":
            return self._score_choice(answer)
        return 0.0

    def _score_coordinate(self, answer: str) -> float:
        """Score coordinate answer like (3, 5)."""
        # Parse answer
        match = re.search(r"\(?\s*(\d+)\s*,\s*(\d+)\s*\)?", answer)
        if not match:
            return 0.0

        try:
            row, col = int(match.group(1)), int(match.group(2))
        except ValueError:
            return 0.0

        # Parse oracle
        oracle_match = re.search(r"\(?\s*(\d+)\s*,\s*(\d+)\s*\)?", self._oracle_answer)
        if not oracle_match:
            return 0.0

        oracle_row, oracle_col = int(oracle_match.group(1)), int(oracle_match.group(2))
        return 1.0 if (row, col) == (oracle_row, oracle_col) else 0.0

    def _score_number(self, answer: str) -> float:
        """Score numeric answer."""
        # Extract number from answer
        match = re.search(r"-?\d+", answer)
        if not match:
            return 0.0

        try:
            num = int(match.group())
            oracle = int(self._oracle_answer)
            return 1.0 if num == oracle else 0.0
        except ValueError:
            return 0.0

    def _score_choice(self, answer: str) -> float:
        """Score multiple choice answer (A-D)."""
        match = re.search(r"[A-Da-d]", answer)
        if not match:
            return 0.0

        choice = match.group().upper()
        oracle = self._oracle_answer.upper()
        return 1.0 if choice == oracle else 0.0

    def _generate_qa(self) -> None:
        """Generate question and oracle answer based on current state."""
        q_type = self._question_type_idx
        self._options = None
        self._moves = None

        if q_type == 0:  # head_pos
            self._question = QUESTION_PROMPTS[0]
            head_row, head_col = self._snake[0]
            self._oracle_answer = f"({int(head_row)}, {int(head_col)})"
            self._answer_format = "coordinate"

        elif q_type == 1:  # food_pos
            self._question = QUESTION_PROMPTS[1]
            food_row, food_col = self._food
            self._oracle_answer = f"({int(food_row)}, {int(food_col)})"
            self._answer_format = "coordinate"

        elif q_type == 2:  # snake_len
            self._question = QUESTION_PROMPTS[2]
            self._oracle_answer = str(len(self._snake))
            self._answer_format = "number"

        elif q_type == 3:  # which_happen
            self._generate_which_happen_qa()
            self._answer_format = "choice"

        elif q_type == 4:  # path
            self._question = QUESTION_PROMPTS[4]
            path = self._find_path()
            self._oracle_answer = str(len(path)) if path else "-1"
            self._answer_format = "number"

    def _generate_which_happen_qa(self) -> None:
        """Generate which_happen question with random moves."""
        # Generate valid moves (not reversing direction)
        move_len = self.np_random.integers(1, 11)
        moves = self._generate_moves(move_len)
        self._moves = moves

        # Simulate moves
        result = self._simulate_moves(moves)

        # Build question
        moves_str = ""
        for i, move in enumerate(moves):
            moves_str += f"\nstep {i+1}: {move}"

        self._question = QUESTION_PROMPTS[3] + moves_str
        options_list = [
            "The snake hits the bound of the grid.",
            "The snake hits its body.",
            "The snake reaches the food.",
            "Nothing happens.",
        ]
        self._options = [
            f"{chr(ord('A') + i)}. {opt}" for i, opt in enumerate(options_list)
        ]
        self._oracle_answer = chr(ord("A") + result)

    def _generate_moves(self, length: int) -> list[str]:
        """Generate valid moves that don't immediately reverse."""
        pos2dir = {
            (-1, 0): "up",
            (0, 1): "right",
            (1, 0): "down",
            (0, -1): "left",
        }
        directions = ["up", "right", "down", "left"]
        opposite = {"up": "down", "down": "up", "left": "right", "right": "left"}

        moves = []
        head, neck = self._snake[0], self._snake[1]
        last_move = pos2dir[(head[0] - neck[0], head[1] - neck[1])]

        for _ in range(length):
            valid_moves = [d for d in directions if d != opposite[last_move]]
            move = valid_moves[self.np_random.integers(0, len(valid_moves))]
            moves.append(move)
            last_move = move

        return moves

    def _simulate_moves(self, moves: list[str]) -> int:
        """Simulate moves and return outcome.

        Returns:
            0: Hit wall
            1: Hit self
            2: Reached food
            3: Nothing happens
        """
        if not moves:
            return 3

        directions = {"up": (-1, 0), "right": (0, 1), "down": (1, 0), "left": (0, -1)}
        current_snake = self._snake.copy()

        for move in moves:
            dr, dc = directions[move]
            head_row, head_col = current_snake[0]
            new_head = (head_row + dr, head_col + dc)

            # Update snake (move without growing)
            current_snake = [new_head] + current_snake[:-1]
            head = current_snake[0]

            # Check outcomes
            if (
                head[0] < 0
                or head[0] >= self._height
                or head[1] < 0
                or head[1] >= self._width
            ):
                return 0  # Hit wall
            if head in current_snake[1:]:
                return 1  # Hit self
            if head == self._food:
                return 2  # Reached food

        return 3  # Nothing happens

    def _find_path(self) -> list[str] | None:
        """Find shortest path from snake head to food using BFS."""
        if self._food is None:
            return None

        queue = deque([(self._snake[0], self._snake.copy(), [])])
        visited = {(self._snake[0], tuple(self._snake))}

        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        dir_symbols = ["up", "right", "down", "left"]

        while queue:
            head_pos, current_snake, path = queue.popleft()

            for (dr, dc), symbol in zip(directions, dir_symbols, strict=False):
                new_head = (head_pos[0] + dr, head_pos[1] + dc)

                # Check if reached food
                if new_head == self._food:
                    return path + [symbol]

                # Calculate new snake after move
                new_snake = [new_head] + current_snake[:-1]

                # Check if valid move
                if not self._is_valid_move(new_head, current_snake):
                    continue

                new_state = (new_head, tuple(new_snake))
                if new_state not in visited:
                    visited.add(new_state)
                    queue.append((new_head, new_snake, path + [symbol]))

        return None

    def _is_valid_move(
        self, head_pos: tuple[int, int], snake_body: list[tuple[int, int]]
    ) -> bool:
        """Check if move is valid (within bounds and not hitting self)."""
        row, col = head_pos
        return (
            0 <= row < self._height
            and 0 <= col < self._width
            and head_pos not in snake_body[:-1]
        )

    def _generate_snake(self) -> None:
        """Generate a random snake on the board."""
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        snake_len = self.np_random.integers(
            self._initial_snake_length[0], self._initial_snake_length[1] + 1
        )
        snake_len = min(snake_len, self._width * self._height - 1)

        max_attempts = 100
        for _ in range(max_attempts):
            # Random starting position for head
            head_row = self.np_random.integers(0, self._height)
            head_col = self.np_random.integers(0, self._width)
            self._snake = [(head_row, head_col)]

            success = True
            for _ in range(snake_len - 1):
                curr_row, curr_col = self._snake[-1]
                valid_dirs = directions.copy()
                self.np_random.shuffle(valid_dirs)

                found_valid = False
                for dr, dc in valid_dirs:
                    new_row, new_col = curr_row + dr, curr_col + dc

                    if (
                        0 <= new_row < self._height
                        and 0 <= new_col < self._width
                        and (new_row, new_col) not in self._snake
                    ):
                        self._snake.append((new_row, new_col))
                        found_valid = True
                        break

                if not found_valid:
                    success = False
                    break

            if success:
                return

        # Fallback: create a short snake
        head_row = self.np_random.integers(0, self._height)
        head_col = self.np_random.integers(0, self._width)
        self._snake = [(head_row, head_col)]

        for dr, dc in directions:
            new_row, new_col = head_row + dr, head_col + dc
            if 0 <= new_row < self._height and 0 <= new_col < self._width:
                self._snake.append((new_row, new_col))
                break

    def _generate_food(self) -> None:
        """Generate food at a random empty position."""
        empty_positions = [
            (r, c)
            for r in range(self._height)
            for c in range(self._width)
            if (r, c) not in self._snake
        ]

        if empty_positions:
            idx = self.np_random.integers(0, len(empty_positions))
            self._food = empty_positions[idx]
        else:
            self._food = None

    def render(self) -> Image.Image | list[Image.Image] | None:
        """Render the game state as a PIL Image."""
        img_width = self._width * self._cell_size + 2 * self._margin
        img_height = self._height * self._cell_size + 2 * self._margin

        img = Image.new("RGB", (img_width, img_height), self.COLORS["grid"])
        draw = ImageDraw.Draw(img)

        # Try to load font
        font_path = self.assets_dir / "DejaVuSans.ttf"
        if font_path.exists():
            font = ImageFont.truetype(str(font_path), self._cell_size // 2)
        else:
            logger.warning(f"Font file not found: {font_path}, using default font")
            font = ImageFont.load_default()

        # Draw coordinate labels
        # Row numbers (left side)
        for row in range(self._height):
            text = str(row)
            y = self._margin + row * self._cell_size + self._cell_size // 2
            draw.text(
                (self._margin // 2, y),
                text,
                fill=self.COLORS["text"],
                font=font,
                anchor="mm",
            )

        # Column numbers (top)
        for col in range(self._width):
            text = str(col)
            x = self._margin + col * self._cell_size + self._cell_size // 2
            draw.text(
                (x, self._margin // 2),
                text,
                fill=self.COLORS["text"],
                font=font,
                anchor="mm",
            )

        # Draw empty cells (white background)
        for row in range(self._height):
            for col in range(self._width):
                x0 = self._margin + col * self._cell_size + 1
                y0 = self._margin + row * self._cell_size + 1
                x1 = x0 + self._cell_size - 2
                y1 = y0 + self._cell_size - 2
                draw.rectangle([x0, y0, x1, y1], fill=self.COLORS["background"])

        # Draw food
        if self._food:
            row, col = self._food
            x0 = self._margin + col * self._cell_size + 1
            y0 = self._margin + row * self._cell_size + 1
            x1 = x0 + self._cell_size - 2
            y1 = y0 + self._cell_size - 2
            draw.rectangle([x0, y0, x1, y1], fill=self.COLORS["food"])

        # Draw snake
        for i, (row, col) in enumerate(self._snake):
            x0 = self._margin + col * self._cell_size + 1
            y0 = self._margin + row * self._cell_size + 1
            x1 = x0 + self._cell_size - 2
            y1 = y0 + self._cell_size - 2
            color = self.COLORS["snake_head"] if i == 0 else self.COLORS["snake_body"]
            draw.rectangle([x0, y0, x1, y1], fill=color)

        # Draw connecting lines between snake segments
        if len(self._snake) > 1:
            points = []
            for row, col in self._snake:
                cx = self._margin + col * self._cell_size + self._cell_size // 2
                cy = self._margin + row * self._cell_size + self._cell_size // 2
                points.append((cx, cy))
            draw.line(points, fill=self.COLORS["line"], width=4)

        return img

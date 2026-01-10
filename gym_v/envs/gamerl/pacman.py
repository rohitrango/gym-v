"""Pacman Q&A environment based on GameRL.

Single-turn Q&A environment where the model answers questions about a Pacman game state.
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

logger = get_logger()


# Question types from original Game-RL
# Removed module-level QUESTION_TYPES - now defined as class variable
GAME_RULES = dedent("""
    # Game Overview
    Pac-Man is a maze arcade game where the player controls Pac-Man to eat as many beans as possible while avoiding ghosts. If a ghost catches Pac-Man, the game ends.

    # Basic Elements
    - **Pac-Man**: The yellow circular character that the player controls
    - **Beans**: Yellow dots that Pac-Man can eat to score points
    - **Walls**: Blue barriers that restrict movement
    - **Ghosts**: Two ghosts (Pinky and Blinky) that chase Pac-Man

    # Game Rules
    - Pac-Man must eat beans while avoiding ghosts
    - Each bean eaten adds 1 point to the score
    - The game ends if a ghost catches Pac-Man
    - Movement is restricted by walls

    # Movement and Direction
    - Pac-Man's mouth opening indicates its current direction
    - The direction can be UP, DOWN, LEFT, or RIGHT
    - Neither Pac-Man nor ghosts can move through walls

    # **Ghost Behavior**
    - **Pinky** (Pink Ghost): Targets up to 4 spaces ahead of Pac-Man's current position and direction (stops at walls)
    - **Blinky** (Red Ghost): Directly targets Pac-Man's current position
    - Both ghosts follow a movement priority system based on the direction they are trying to move:
      - When moving in more than one direction is optimal, the priority order for both ghosts is **UP > DOWN > LEFT > RIGHT**.
      - This means if a ghost has multiple possible directions to move in, it will first attempt to move **UP** if possible, then **DOWN**, followed by **LEFT**, and finally **RIGHT** if all other directions are blocked.

    # Board Layout
    - The board is surrounded by walls on all four sides
    - Position (0,0) is located at the top-left corner wall
    - Movement grid uses (row, column) coordinates

    # Scoring
    The score equals the total number of beans eaten by Pac-Man
""").strip()

ANSWER_FORMAT_PROMPT = dedent("""
    **Answer Format:**
    - For multiple choice: Reply with only the letter (A, B, C, etc.)
    - For numbers: Reply with only the number

    Do not include any explanation or extra text.
""").strip()


class Ghost:
    """Class representing a ghost in the Pac-Man game."""

    def __init__(self, name: str, position: tuple[int, int], game: GameRLPacmanQAEnv):
        self.name = name
        self.game = game
        self.position = position
        self.path: list[tuple[int, int]] = []

    def update_direction(self) -> None:
        """Update the ghost's path based on its target using BFS."""
        if self.name == "Pinky":
            target = self.game._get_pinky_target()
        else:  # Blinky
            target = self.game._pacman_position

        if target:
            self.path = self.game._bfs(self.position, target)

    def move(self) -> None:
        """Move the ghost along the path towards its target."""
        if not self.path or len(self.path) < 2:
            return

        self.position = self.path[1]
        self.path.pop(0)


class GameRLPacmanQAEnv(Env):
    """Pacman Q&A environment.

    Single-turn Q&A environment based on the original Game-RL Pacman game.
    Given a game state image, answer questions about positions, movements,
    ghost behavior, or optimal strategies.

    Args:
        question_type: Question type ID (0-9). None for random selection.
        grid_size: Size of the grid (default 16)
        wall_ratio: Ratio of internal walls (default 0.1)
        cell_size: Size of each cell in pixels for rendering (default 25)
    """

    # Question types
    QUESTION_TYPES = [
        {
            "id": "type_0",
            "name": "pacman_position",
            "level": "Easy",
            "answer_format": "multiple_choice",
            "qa_type": "StateInfo",
        },
        {
            "id": "type_1",
            "name": "bean_count_5x5",
            "level": "Easy",
            "answer_format": "fill_in_blank",
            "qa_type": "StateInfo",
        },
        {
            "id": "type_2",
            "name": "closer_ghost",
            "level": "Easy",
            "answer_format": "multiple_choice",
            "qa_type": "StateInfo",
        },
        {
            "id": "type_3",
            "name": "beans_in_direction",
            "level": "Easy",
            "answer_format": "fill_in_blank",
            "qa_type": "ActionOutcome",
        },
        {
            "id": "type_4",
            "name": "movement_result",
            "level": "Hard",
            "answer_format": "multiple_choice",
            "qa_type": "ActionOutcome",
        },
        {
            "id": "type_5",
            "name": "pinky_direction_change",
            "level": "Medium",
            "answer_format": "multiple_choice",
            "qa_type": "ActionOutcome",
        },
        {
            "id": "type_6",
            "name": "blinky_direction_change",
            "level": "Medium",
            "answer_format": "multiple_choice",
            "qa_type": "ActionOutcome",
        },
        {
            "id": "type_7",
            "name": "pinky_next_move",
            "level": "Medium",
            "answer_format": "multiple_choice",
            "qa_type": "TransitionPath",
        },
        {
            "id": "type_8",
            "name": "blinky_next_move",
            "level": "Medium",
            "answer_format": "multiple_choice",
            "qa_type": "TransitionPath",
        },
        {
            "id": "type_9",
            "name": "optimal_direction",
            "level": "Hard",
            "answer_format": "multiple_choice",
            "qa_type": "StrategyOptimization",
        },
    ]

    gamerl_assets_dir = resources.files("gym_v.envs.gamerl") / "assets"
    assets_dir = resources.files("gym_v.envs") / "assets"

    DIRECTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
    DIR_MAP = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}

    COLORS = {
        "background": (0, 0, 0),
        "wall": (0, 0, 139),
        "bean": (255, 255, 0),
        "text": (255, 255, 255),
        "score": (255, 255, 0),
    }

    def __init__(
        self,
        question_type: int | None = None,
        grid_size: int = 16,
        wall_ratio: float = 0.1,
        cell_size: int = 25,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._question_type = question_type
        self._grid_size = grid_size
        self._wall_ratio = wall_ratio
        self._cell_size = cell_size
        self._margin = 30
        self._score_height = 20

        # Game state
        self._walls: set[tuple[int, int]] = set()
        self._beans: set[tuple[int, int]] = set()
        self._pacman_position: tuple[int, int] = (1, 1)
        self._direction: str = "RIGHT"
        self._ghosts: list[Ghost] = []
        self._score: int = 0

        # Q&A state
        self._current_question_type: int = 0
        self._current_question: str = ""
        self._oracle_answer: str = ""
        self._options: list[str] | None = None

        # Load images
        self._pacman_image = self._load_image("pacman.png")
        self._ghost_images = {
            "Pinky": self._load_image("Pinky.png"),
            "Blinky": self._load_image("Blinky.png"),
        }

    def _load_image(self, filename: str) -> Image.Image | None:
        """Load and scale an image from assets."""
        image_path = self.gamerl_assets_dir / filename
        try:
            if image_path.is_file():
                img = Image.open(str(image_path))
                return img.resize(
                    (self._cell_size, self._cell_size), Image.Resampling.LANCZOS
                )
        except Exception as e:
            logger.warning(f"Error loading image {filename}: {e}")
        return None

    @property
    def description(self) -> str:
        """Return game rules + current question + answer format."""
        desc = GAME_RULES + "\n\n**Question:** " + self._current_question

        if self._options:
            desc += "\n\n**Options:**\n" + "\n".join(self._options)

        desc += ANSWER_FORMAT_PROMPT
        return desc.strip()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)

        if seed is not None:
            random.seed(seed)

        self._score = 0
        self._direction = "RIGHT"

        # Create walls
        self._walls = self._create_outer_walls()
        self._add_internal_walls()

        # Initialize beans
        self._initialize_beans()

        # Initialize Pac-Man
        self._pacman_position = self._get_random_start_position()
        if self._pacman_position in self._beans:
            self._beans.remove(self._pacman_position)
            self._score += 1

        # Initialize ghosts
        self._ghosts = []
        self._add_ghosts()

        # Select question type
        if self._question_type is not None:
            self._current_question_type = self._question_type
        else:
            self._current_question_type = self.np_random.integers(
                0, len(self.QUESTION_TYPES)
            )

        # Generate Q&A
        self._generate_qa()

        logger.info(f"Reset Pacman QA (type={self._current_question_type}).")

        obs = Observation(
            image=self.render(),
            text=self._get_state_text(),
            metadata={
                "question": self._current_question,
                "options": self._options,
                "question_type": self.QUESTION_TYPES[self._current_question_type][
                    "name"
                ],
                "level": self.QUESTION_TYPES[self._current_question_type]["level"],
            },
        )
        info = {
            "oracle_answer": self._oracle_answer,
            "question_type": self._current_question_type,
        }
        return obs, info

    def inner_step(
        self, action: str
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        """Evaluate the answer. Always terminates after one step."""
        answer = action.strip().upper()
        reward = self._score_answer(answer)

        obs = Observation(
            image=self.render(),
            text=None,
            metadata={
                "question_type": self.QUESTION_TYPES[self._current_question_type][
                    "name"
                ],
            },
        )
        info = {
            "oracle_answer": self._oracle_answer,
            "user_answer": answer,
            "correct": reward == 1.0,
        }

        return obs, reward, True, False, info

    def _score_answer(self, answer: str) -> float:
        """Score the answer based on answer format."""
        answer_format = self.QUESTION_TYPES[self._current_question_type][
            "answer_format"
        ]

        if answer_format == "multiple_choice":
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

    def _get_state_text(self) -> str:
        """Generate text description of the current game state.

        Returns a text representation that contains the same information as the rendered image.
        """
        # Create a text grid representation
        grid = [["." for _ in range(self._grid_size)] for _ in range(self._grid_size)]

        # Mark walls
        for wall in self._walls:
            grid[wall[0]][wall[1]] = "#"

        # Mark beans
        for bean in self._beans:
            grid[bean[0]][bean[1]] = "*"

        # Mark ghosts
        for ghost in self._ghosts:
            pos = ghost.position
            if ghost.name == "Pinky":
                grid[pos[0]][pos[1]] = "P"
            elif ghost.name == "Blinky":
                grid[pos[0]][pos[1]] = "B"

        # Mark Pacman (override anything else at this position)
        grid[self._pacman_position[0]][self._pacman_position[1]] = "C"

        # Build the text description
        grid_str = "\n".join(["".join(row) for row in grid])

        state_text = f"""Grid Size: {self._grid_size}x{self._grid_size}
Grid (C=Pacman, P=Pinky, B=Blinky, *=bean, #=wall, .=empty):
{grid_str}"""

        return state_text

    def _generate_qa(self) -> None:
        """Generate question and oracle answer based on current state."""
        q_type = self._current_question_type
        self._options = None

        if q_type == 0:
            self._generate_q0_pacman_position()
        elif q_type == 1:
            self._generate_q1_bean_count()
        elif q_type == 2:
            self._generate_q2_closer_ghost()
        elif q_type == 3:
            self._generate_q3_beans_in_direction()
        elif q_type == 4:
            self._generate_q4_movement_result()
        elif q_type == 5:
            self._generate_q5_pinky_direction_change()
        elif q_type == 6:
            self._generate_q6_blinky_direction_change()
        elif q_type == 7:
            self._generate_q7_pinky_next_move()
        elif q_type == 8:
            self._generate_q8_blinky_next_move()
        elif q_type == 9:
            self._generate_q9_optimal_direction()

    def _generate_q0_pacman_position(self) -> None:
        """Q0: What is Pac-Man's position and direction?"""
        self._current_question = "What is Pac-Man's position and direction?"
        correct = (self._pacman_position[0], self._pacman_position[1], self._direction)

        # Generate distractors
        options_data = [correct]
        while len(options_data) < 8:
            r = self.np_random.integers(0, self._grid_size)
            c = self.np_random.integers(0, self._grid_size)
            d = self.DIRECTIONS[self.np_random.integers(0, 4)]
            if (r, c, d) not in options_data:
                options_data.append((r, c, d))

        self.np_random.shuffle(options_data)

        self._options = []
        for i, (r, c, d) in enumerate(options_data):
            letter = chr(ord("A") + i)
            self._options.append(f"{letter}. ({r}, {c}), {d}")
            if (r, c, d) == correct:
                self._oracle_answer = letter

    def _generate_q1_bean_count(self) -> None:
        """Q1: Count beans in 5x5 grid around Pac-Man."""
        self._current_question = "Now how many beans are visible there in the 5 by 5 grid around the Pac-man center?"

        pacman_row, pacman_col = self._pacman_position
        bean_count = 0
        ghost_positions = {g.position for g in self._ghosts}

        for row in range(pacman_row - 2, pacman_row + 3):
            for col in range(pacman_col - 2, pacman_col + 3):
                if (
                    0 <= row < self._grid_size
                    and 0 <= col < self._grid_size
                    and (row, col) in self._beans
                    and (row, col) not in ghost_positions
                ):
                    bean_count += 1

        self._oracle_answer = str(bean_count)
        self._options = None

    def _generate_q2_closer_ghost(self) -> None:
        """Q2: Which ghost is closer to Pac-Man?"""
        self._current_question = "Which ghost is closer to Pac-Man, Pinky or Blinky?"

        pinky = next(g for g in self._ghosts if g.name == "Pinky")
        blinky = next(g for g in self._ghosts if g.name == "Blinky")

        pinky_dist = abs(pinky.position[0] - self._pacman_position[0]) + abs(
            pinky.position[1] - self._pacman_position[1]
        )
        blinky_dist = abs(blinky.position[0] - self._pacman_position[0]) + abs(
            blinky.position[1] - self._pacman_position[1]
        )

        self._options = [
            "A. Pinky is closer to Pac-Man",
            "B. Blinky is closer to Pac-Man",
            "C. Both ghosts are equidistant from Pac-Man",
        ]

        if pinky_dist < blinky_dist:
            self._oracle_answer = "A"
        elif blinky_dist < pinky_dist:
            self._oracle_answer = "B"
        else:
            self._oracle_answer = "C"

    def _generate_q3_beans_in_direction(self) -> None:
        """Q3: How many beans in current direction until wall?"""
        self._current_question = "Assuming the ghosts don't move, how many beans can Pac-Man eat if it moves in its current direction until hitting a wall?"

        row, col = self._pacman_position
        dr, dc = self.DIR_MAP[self._direction]
        bean_count = 0

        while True:
            row += dr
            col += dc
            if (row, col) in self._walls or not (
                0 <= row < self._grid_size and 0 <= col < self._grid_size
            ):
                break
            if (row, col) in self._beans:
                bean_count += 1

        self._oracle_answer = str(bean_count)
        self._options = None

    def _generate_q4_movement_result(self) -> None:
        """Q4: What happens after moving in specified directions?"""
        dir1 = self.DIRECTIONS[self.np_random.integers(0, 4)]
        dir2 = self.DIRECTIONS[self.np_random.integers(0, 4)]
        num1 = self.np_random.integers(1, 4)
        num2 = self.np_random.integers(1, 4)

        self._current_question = f"Assuming Pac-Man and both ghosts move one step at a time, what would happen if Pac-Man moves {dir1} {num1} times, then {dir2} {num2} times?"

        # Simulate movement
        result, beans_eaten, caught_by = self._simulate_movement(
            [(dir1, num1), (dir2, num2)]
        )
        # Generate options
        beans_list = [beans_eaten]
        for _ in range(5):
            while True:
                rb = self.np_random.integers(0, beans_eaten + 6)
                if rb != beans_eaten and rb not in beans_list:
                    beans_list.append(rb)
                    break

        self.np_random.shuffle(beans_list[1:])
        correct_idx = self.np_random.integers(0, 6)

        self._options = []
        for i in range(6):
            letter = chr(ord("A") + i)
            if caught_by is None:
                if i == correct_idx:
                    b = beans_eaten
                elif i > correct_idx:
                    b = beans_list[i] if i < len(beans_list) else beans_list[-1]
                else:
                    b = beans_list[i + 1] if i + 1 < len(beans_list) else beans_list[-1]
                self._options.append(
                    f"{letter}. It will eat {b} beans, and the score will become {self._score + b}"
                )
                if i == correct_idx:
                    self._oracle_answer = letter
            else:
                b = beans_list[i] if i < len(beans_list) else beans_list[-1]
                self._options.append(
                    f"{letter}. It will eat {b} beans, and the score will become {self._score + b}"
                )

        self._options.append("G. It will be caught by Pinky (the pink ghost)")
        self._options.append("H. It will be caught by Blinky (the red ghost)")

        if caught_by == "Pinky":
            self._oracle_answer = "G"
        elif caught_by == "Blinky":
            self._oracle_answer = "H"

    def _generate_q5_pinky_direction_change(self) -> None:
        """Q5: Will Pinky's direction change after Pac-Man moves?"""
        dir0 = self.DIRECTIONS[self.np_random.integers(0, 4)]
        num1 = self.np_random.integers(1, 4)

        self._current_question = f"Assuming Pinky doesn't move, if Pac-Man moves {dir0} {num1} times, will Pinky's next movement direction change?"

        pinky = next(g for g in self._ghosts if g.name == "Pinky")
        original_dir = self._get_ghost_direction(pinky, "Pinky")

        # Simulate Pac-Man movement
        test_pos = self._pacman_position
        test_dir = self._direction
        for _ in range(num1):
            dr, dc = self.DIR_MAP[dir0]
            new_pos = (test_pos[0] + dr, test_pos[1] + dc)
            if (
                new_pos not in self._walls
                and 0 <= new_pos[0] < self._grid_size
                and 0 <= new_pos[1] < self._grid_size
            ):
                test_pos = new_pos
                test_dir = dir0

        # Calculate new direction
        new_dir = self._get_ghost_direction_at(
            pinky.position, test_pos, test_dir, "Pinky"
        )

        other_dirs = [d for d in self.DIRECTIONS if d != original_dir]
        self.np_random.shuffle(other_dirs)

        self._options = [
            f"A. Pinky's direction remains unchanged, still {original_dir}",
            f"B. Pinky's direction changes to {other_dirs[0]}",
            f"C. Pinky's direction changes to {other_dirs[1]}",
            f"D. Pinky's direction changes to {other_dirs[2]}",
        ]

        if new_dir == original_dir:
            self._oracle_answer = "A"
        else:
            for i, opt in enumerate(self._options):
                if new_dir in opt:
                    self._oracle_answer = chr(ord("A") + i)
                    break

    def _generate_q6_blinky_direction_change(self) -> None:
        """Q6: Will Blinky's direction change after Pac-Man moves?"""
        dir0 = self.DIRECTIONS[self.np_random.integers(0, 4)]
        num1 = self.np_random.integers(1, 4)

        self._current_question = f"Assuming Blinky doesn't move, if Pac-Man moves {dir0} {num1} times, will Blinky's next movement direction change?"

        blinky = next(g for g in self._ghosts if g.name == "Blinky")
        original_dir = self._get_ghost_direction(blinky, "Blinky")

        # Simulate Pac-Man movement
        test_pos = self._pacman_position
        for _ in range(num1):
            dr, dc = self.DIR_MAP[dir0]
            new_pos = (test_pos[0] + dr, test_pos[1] + dc)
            if (
                new_pos not in self._walls
                and 0 <= new_pos[0] < self._grid_size
                and 0 <= new_pos[1] < self._grid_size
            ):
                test_pos = new_pos

        # Calculate new direction (Blinky targets Pac-Man directly)
        new_dir = self._get_ghost_direction_at(
            blinky.position, test_pos, self._direction, "Blinky"
        )

        other_dirs = [d for d in self.DIRECTIONS if d != original_dir]
        self.np_random.shuffle(other_dirs)

        self._options = [
            f"A. Blinky's direction remains unchanged, still {original_dir}",
            f"B. Blinky's direction changes to {other_dirs[0]}",
            f"C. Blinky's direction changes to {other_dirs[1]}",
            f"D. Blinky's direction changes to {other_dirs[2]}",
        ]

        if new_dir == original_dir:
            self._oracle_answer = "A"
        else:
            for i, opt in enumerate(self._options):
                if new_dir in opt:
                    self._oracle_answer = chr(ord("A") + i)
                    break

    def _generate_q7_pinky_next_move(self) -> None:
        """Q7: Where will Pinky move next if Pac-Man stays still?"""
        self._current_question = (
            "If Pac-Man stays still, where will Pinky move in the next turn?"
        )

        pinky = next(g for g in self._ghosts if g.name == "Pinky")
        direction = self._get_ghost_direction(pinky, "Pinky")

        self._options = [
            "A. Pinky will move one step UP",
            "B. Pinky will move one step DOWN",
            "C. Pinky will move one step RIGHT",
            "D. Pinky will move one step LEFT",
        ]

        dir_to_letter = {"UP": "A", "DOWN": "B", "RIGHT": "C", "LEFT": "D"}
        self._oracle_answer = dir_to_letter.get(direction, "A")

    def _generate_q8_blinky_next_move(self) -> None:
        """Q8: Where will Blinky move next if Pac-Man stays still?"""
        self._current_question = (
            "If Pac-Man stays still, where will Blinky move in the next turn?"
        )

        blinky = next(g for g in self._ghosts if g.name == "Blinky")
        direction = self._get_ghost_direction(blinky, "Blinky")

        self._options = [
            "A. Blinky will move one step UP",
            "B. Blinky will move one step DOWN",
            "C. Blinky will move one step RIGHT",
            "D. Blinky will move one step LEFT",
        ]

        dir_to_letter = {"UP": "A", "DOWN": "B", "RIGHT": "C", "LEFT": "D"}
        self._oracle_answer = dir_to_letter.get(direction, "A")

    def _generate_q9_optimal_direction(self) -> None:
        """Q9: Which direction should Pac-Man move to eat most beans safely?"""
        self._current_question = "If Pac-Man and both ghosts move one step at a time, in which direction should Pac-Man move continuously until hitting a wall to eat the most beans without being caught by a ghost? (When moving in more than one direction is optimal, the priority order is UP > DOWN > LEFT > RIGHT)"

        best_dir = None
        max_beans = -1
        all_caught = True

        for direction in self.DIRECTIONS:
            result, beans, caught = self._simulate_movement([(direction, 100)])
            if caught is None and beans > max_beans:
                max_beans = beans
                best_dir = direction
                all_caught = False
            elif caught is None and beans == max_beans:
                # Priority order
                if best_dir is None or self.DIRECTIONS.index(
                    direction
                ) < self.DIRECTIONS.index(best_dir):
                    best_dir = direction

        self._options = [
            "A. Pac-Man should move UP",
            "B. Pac-Man should move DOWN",
            "C. Pac-Man should move RIGHT",
            "D. Pac-Man should move LEFT",
            "E. Pac-Man will be caught by a ghost regardless of direction",
        ]

        if all_caught:
            self._oracle_answer = "E"
        else:
            dir_to_letter = {"UP": "A", "DOWN": "B", "RIGHT": "C", "LEFT": "D"}
            self._oracle_answer = dir_to_letter.get(best_dir, "E")

    def _get_ghost_direction(self, ghost: Ghost, ghost_name: str) -> str:
        """Get the direction a ghost will move."""
        return self._get_ghost_direction_at(
            ghost.position, self._pacman_position, self._direction, ghost_name
        )

    def _get_ghost_direction_at(
        self,
        ghost_pos: tuple[int, int],
        pacman_pos: tuple[int, int],
        pacman_dir: str,
        ghost_name: str,
    ) -> str:
        """Get ghost direction for a given state."""
        if ghost_name == "Pinky":
            target = self._get_pinky_target_at(pacman_pos, pacman_dir)
        else:
            target = pacman_pos

        path = self._bfs(ghost_pos, target)
        if len(path) >= 2:
            dr = path[1][0] - path[0][0]
            dc = path[1][1] - path[0][1]
            for d, (ddr, ddc) in self.DIR_MAP.items():
                if (ddr, ddc) == (dr, dc):
                    return d
        return self._direction

    def _get_pinky_target(self) -> tuple[int, int]:
        """Calculate Pinky's target: 4 cells ahead of Pac-Man."""
        return self._get_pinky_target_at(self._pacman_position, self._direction)

    def _get_pinky_target_at(
        self, pos: tuple[int, int], direction: str
    ) -> tuple[int, int]:
        """Calculate Pinky's target for given position and direction."""
        row, col = pos
        target = (row, col)
        dr, dc = self.DIR_MAP.get(direction, (0, 1))

        for _ in range(4):
            next_cell = (target[0] + dr, target[1] + dc)
            if (
                0 <= next_cell[0] < self._grid_size
                and 0 <= next_cell[1] < self._grid_size
                and next_cell not in self._walls
            ):
                target = next_cell
            else:
                break

        return target

    def _simulate_movement(
        self, moves: list[tuple[str, int]]
    ) -> tuple[str, int, str | None]:
        """Simulate Pac-Man and ghost movements.

        Returns:
            (result_type, beans_eaten, caught_by_ghost_name or None)
        """
        # Deep copy state
        test_pos = self._pacman_position
        test_dir = self._direction
        test_beans = self._beans.copy()
        ghost_positions = {g.name: g.position for g in self._ghosts}
        beans_eaten = 0

        for direction, count in moves:
            for _ in range(count):
                # Move Pac-Man
                dr, dc = self.DIR_MAP.get(direction, (0, 0))
                new_pos = (test_pos[0] + dr, test_pos[1] + dc)

                if (
                    new_pos not in self._walls
                    and 0 <= new_pos[0] < self._grid_size
                    and 0 <= new_pos[1] < self._grid_size
                ):
                    test_pos = new_pos
                    test_dir = direction

                    # Eat bean
                    if test_pos in test_beans:
                        test_beans.remove(test_pos)
                        beans_eaten += 1

                # Move ghosts
                for g_name, g_pos in ghost_positions.items():
                    if g_name == "Pinky":
                        target = self._get_pinky_target_at(test_pos, test_dir)
                    else:
                        target = test_pos

                    path = self._bfs(g_pos, target)
                    if len(path) >= 2:
                        ghost_positions[g_name] = path[1]

                    # Check collision
                    if ghost_positions[g_name] == test_pos:
                        return ("caught", beans_eaten, g_name)

        return ("complete", beans_eaten, None)

    def _bfs(
        self, start: tuple[int, int], goal: tuple[int, int]
    ) -> list[tuple[int, int]]:
        """BFS to find shortest path."""
        queue: deque[list[tuple[int, int]]] = deque()
        queue.append([start])
        visited: set[tuple[int, int]] = {start}

        while queue:
            path = queue.popleft()
            current = path[-1]

            if current == goal:
                return path

            # Priority order: UP, DOWN, LEFT, RIGHT
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (current[0] + dr, current[1] + dc)
                if (
                    0 <= neighbor[0] < self._grid_size
                    and 0 <= neighbor[1] < self._grid_size
                    and neighbor not in visited
                    and neighbor not in self._walls
                ):
                    visited.add(neighbor)
                    queue.append(path + [neighbor])

        return [start]

    def _create_outer_walls(self) -> set[tuple[int, int]]:
        """Create outer walls."""
        walls = set()
        for row in range(self._grid_size):
            for col in range(self._grid_size):
                if (
                    row == 0
                    or row == self._grid_size - 1
                    or col == 0
                    or col == self._grid_size - 1
                ):
                    walls.add((row, col))
        return walls

    def _add_internal_walls(self) -> None:
        """Add internal walls."""
        total_cells = self._grid_size * self._grid_size
        num_internal_walls = int(total_cells * self._wall_ratio)

        available = [
            (row, col)
            for row in range(1, self._grid_size - 1)
            for col in range(1, self._grid_size - 1)
            if (row, col) not in self._walls
        ]
        random.shuffle(available)

        internal_walls: set[tuple[int, int]] = set()
        for pos in available:
            if len(internal_walls) >= num_internal_walls:
                break

            row, col = pos
            neighbors = [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]
            if sum(1 for n in neighbors if n in internal_walls) < 2:
                internal_walls.add(pos)

        self._walls.update(internal_walls)

    def _initialize_beans(self) -> None:
        """Initialize beans."""
        self._beans = set(
            (row, col)
            for row in range(self._grid_size)
            for col in range(self._grid_size)
            if (row, col) not in self._walls
        )

    def _get_random_start_position(self) -> tuple[int, int]:
        """Get random start position."""
        available = [
            (row, col)
            for row in range(self._grid_size)
            for col in range(self._grid_size)
            if (row, col) not in self._walls
        ]
        return random.choice(available)

    def _add_ghosts(self) -> None:
        """Add ghosts."""
        for name in ["Pinky", "Blinky"]:
            available = [
                (row, col)
                for row in range(self._grid_size)
                for col in range(self._grid_size)
                if (row, col) not in self._walls
                and (row, col) != self._pacman_position
                and all(g.position != (row, col) for g in self._ghosts)
            ]
            if available:
                pos = random.choice(available)
                self._ghosts.append(Ghost(name, pos, self))

    def render(self) -> Image.Image:
        """Render the game state."""
        img_width = self._margin + self._grid_size * self._cell_size
        img_height = (
            self._score_height + self._margin + self._grid_size * self._cell_size
        )

        img = Image.new("RGB", (img_width, img_height), self.COLORS["background"])
        draw = ImageDraw.Draw(img)

        font_path = self.assets_dir / "DejaVuSans.ttf"
        if font_path.exists():
            font = ImageFont.truetype(str(font_path), 12)
        else:
            font = ImageFont.load_default()

        # Score
        draw.text(
            (10, 2), f"Score: {self._score}", fill=self.COLORS["score"], font=font
        )

        # Coordinates
        for i in range(self._grid_size):
            x = self._margin + i * self._cell_size + self._cell_size // 2
            draw.text(
                (x, self._score_height + 2),
                str(i),
                fill=self.COLORS["text"],
                font=font,
                anchor="mt",
            )
            y = (
                self._score_height
                + self._margin
                + i * self._cell_size
                + self._cell_size // 2
            )
            draw.text((2, y), str(i), fill=self.COLORS["text"], font=font, anchor="lm")

        # Walls
        for row, col in self._walls:
            x = self._margin + col * self._cell_size
            y = self._score_height + self._margin + row * self._cell_size
            draw.rectangle(
                [x, y, x + self._cell_size, y + self._cell_size],
                fill=self.COLORS["wall"],
            )

        # Beans
        for row, col in self._beans:
            cx = self._margin + col * self._cell_size + self._cell_size // 2
            cy = (
                self._score_height
                + self._margin
                + row * self._cell_size
                + self._cell_size // 2
            )
            r = self._cell_size // 6
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=self.COLORS["bean"])

        # Ghosts
        for ghost in self._ghosts:
            row, col = ghost.position
            x = self._margin + col * self._cell_size
            y = self._score_height + self._margin + row * self._cell_size
            ghost_img = self._ghost_images.get(ghost.name)
            if ghost_img:
                img.paste(ghost_img, (x, y))

        # Pac-Man
        row, col = self._pacman_position
        x = self._margin + col * self._cell_size
        y = self._score_height + self._margin + row * self._cell_size
        if self._pacman_image:
            rotated = self._rotate_image(self._pacman_image, self._direction)
            img.paste(rotated, (x, y))

        return img

    def _rotate_image(self, image: Image.Image, direction: str) -> Image.Image:
        """Rotate image based on direction."""
        if direction == "LEFT":
            return image.rotate(180)
        elif direction == "UP":
            return image.rotate(90)
        elif direction == "DOWN":
            return image.rotate(-90)
        return image
